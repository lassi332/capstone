"""
Movement-Based Autism Spectrum Disorder (ASD) Classification Harness
=====================================================================

Production-Grade Clinical Machine Learning & Diagnostic Evaluation Pipeline
for Tabular 3D Full-Body Skeletal Kinematics.

Key Architectural Guarantees:
1. Strict Group-Aware Data Partitioning (Zero Data Leakage on child_id):
   - 50% Training Partition (50 children: 25 ASD, 25 TD)
   - 25% Validation Partition (25 children: 12-13 ASD, 12-13 TD)
   - 25% Held-Out Final Test Cohort (25 children: 12-13 ASD, 12-13 TD)
2. Leak-Free Preprocessing: Imputers and Scalers fit EXCLUSIVELY on the training partition.
3. Optuna Hyperparameter Optimization with 5-Fold GroupKFold Cross-Validation.
4. Candidate Models: CatBoost (Primary), LightGBM, XGBoost, Random Forest, Soft-Voting Ensemble.
5. Dual-Level Diagnostic Evaluation (Trial/Frame-Level & Aggregated Child-Level).
6. Probability Calibration (Platt Scaling & Isotonic Regression) with ECE metrics.
7. Clinical Metrics with 95% Confidence Intervals & TreeExplainer SHAP Interpretability.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

# Gradient Boosted Decision Trees
import catboost as cb
import lightgbm as lgb
import optuna

# Optional XGBoost import with graceful fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception):
    xgb = None
    XGB_AVAILABLE = False

# SHAP for model explainability
import shap

# Configure Optuna logging level
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress minor non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# 1. LOGGING & CONFIGURATION
# =====================================================================

def setup_logger(output_dir: Path) -> logging.Logger:
    """Configures structured console and file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline_execution.log"
    
    logger = logging.getLogger("ASD_Kinematic_Harness")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if re-initialized
    if not logger.handlers:
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        
        c_format = logging.Formatter("[%(asctime)s | %(levelname)-7s] %(message)s", datefmt="%H:%M:%S")
        f_format = logging.Formatter("[%(asctime)s | %(levelname)-7s | %(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger


@dataclass
class HarnessConfig:
    """Global configuration dataclass for the clinical machine learning harness."""
    # Data Sources
    data_path: Optional[str] = "Dataset/Final dataset.xlsx"
    use_synthetic_if_missing: bool = True
    
    # Cohort Settings
    total_children: int = 100
    n_asd: int = 50
    n_td: int = 50
    min_frames_per_child: int = 150
    max_frames_per_child: int = 250
    sampling_rate_hz: float = 30.0
    
    # Partition Ratios (Must sum to 1.0)
    train_ratio: float = 0.50
    val_ratio: float = 0.25
    test_ratio: float = 0.25
    
    # Cross Validation & Tuning
    n_cv_folds: int = 5
    optuna_trials_primary: int = 20
    optuna_trials_benchmark: int = 15
    optuna_seed: int = 42
    
    # Preprocessing
    imputation_strategy: str = "median"
    scaling_strategy: str = "robust"  # "robust" or "standard"
    
    # Calibration & Thresholding
    calibration_method: str = "sigmoid"  # "sigmoid" (Platt) or "isotonic"
    bootstrap_iterations: int = 1000
    
    # Reproducibility & Output
    random_state: int = 42
    output_dir: Path = field(default_factory=lambda: Path("./asd_kinematics_output"))
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.models_dir = self.output_dir / "models"
        
        for d in [self.output_dir, self.figures_dir, self.tables_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 2. REAL & SYNTHETIC DATA LOADERS
# =====================================================================

class ClinicalDatasetLoader:
    """
    Loads and structures tabular 3D kinematic datasets:
    - Ingests real motion capture dataset from Excel/CSV (e.g. 100 children: 50 ASD, 50 TD, 800 trials).
    - Accurately maps subject identifiers (child_id) to enforce strict zero-leakage group boundaries.
    """

    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_dataset(self) -> Tuple[pd.DataFrame, bool]:
        """
        Loads the clinical dataset from disk, or falls back to synthetic generation.
        Returns:
            df: DataFrame containing kinematic features, child_id, and binary label (1=ASD, 0=TD).
            is_precomputed_features: Boolean indicating if features are already extracted.
        """
        if self.config.data_path and os.path.exists(self.config.data_path):
            self.logger.info(f"Loading real clinical kinematic dataset from: {self.config.data_path}")
            start_time = time.time()
            df = pd.read_excel(self.config.data_path)
            
            # Map child_id and label for the 100-child cohort (8 rows per child)
            if "child_id" not in df.columns:
                n_rows = len(df)
                self.logger.info(f"Mapping child identifiers for {n_rows} records (8 movement trials per child)...")
                child_ids = []
                labels = []
                
                # 400 ASD trials (50 children x 8)
                for i in range(50):
                    for r in range(8):
                        child_ids.append(f"ASD_{i+1:03d}")
                        labels.append(1)
                # 400 TD control trials (50 children x 8)
                for j in range(50):
                    for r in range(8):
                        child_ids.append(f"TD_{j+1:03d}")
                        labels.append(0)
                        
                df["child_id"] = child_ids
                df["label"] = labels
            elif "label" not in df.columns and "class" in df.columns:
                df["label"] = df["class"].map({"A": 1, "T": 0, 1: 1, 0: 0})
                
            elapsed = time.time() - start_time
            self.logger.info(
                f"Successfully loaded real cohort dataset: {len(df):,} records, "
                f"{len(df['child_id'].unique())} children ({sum(df.groupby('child_id')['label'].first()==1)} ASD, "
                f"{sum(df.groupby('child_id')['label'].first()==0)} TD), {len(df.columns)-2} kinematic features in {elapsed:.2f}s."
            )
            return df, True
        else:
            self.logger.info("Real dataset path not found; utilizing Synthetic Kinematics Generator.")
            generator = SyntheticKinematicsGenerator(self.config, self.logger)
            raw_df = generator.generate_cohort()
            return raw_df, False


class SyntheticKinematicsGenerator:
    """
    Generates realistic 3D full-body skeletal kinematic trajectories for a 100-child cohort
    (50 ASD, 50 Typically Developing), embedding established biomechanical phenotypes:
    - Altered movement smoothness and elevated sub-movement jerk in ASD [1, 2, 6].
    - Increased postural sway and Center-of-Mass (CoM) excursion variance [2, 9].
    - Bilateral upper-limb movement asymmetry [2].
    - Intermittent marker occlusion / missing coordinate tracking artifacts [10].
    """
    
    JOINTS = [
        "head", "neck", "spine_torso", "pelvis",
        "shoulder_left", "elbow_left", "wrist_left",
        "shoulder_right", "elbow_right", "wrist_right",
        "hip_left", "knee_left", "ankle_left",
        "hip_right", "knee_right", "ankle_right"
    ]
    
    BASE_SKELETON = {
        "pelvis": np.array([0.0, 0.0, 0.65]),
        "spine_torso": np.array([0.0, 0.0, 0.85]),
        "neck": np.array([0.0, 0.0, 1.05]),
        "head": np.array([0.0, 0.0, 1.25]),
        "shoulder_left": np.array([-0.18, 0.0, 1.02]),
        "elbow_left": np.array([-0.28, -0.05, 0.78]),
        "wrist_left": np.array([-0.30, -0.15, 0.55]),
        "shoulder_right": np.array([0.18, 0.0, 1.02]),
        "elbow_right": np.array([0.28, -0.05, 0.78]),
        "wrist_right": np.array([0.30, -0.15, 0.55]),
        "hip_left": np.array([-0.10, 0.0, 0.60]),
        "knee_left": np.array([-0.11, 0.05, 0.35]),
        "ankle_left": np.array([-0.11, 0.0, 0.08]),
        "hip_right": np.array([0.10, 0.0, 0.60]),
        "knee_right": np.array([0.11, 0.05, 0.35]),
        "ankle_right": np.array([0.11, 0.0, 0.08]),
    }

    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.rng = np.random.RandomState(config.random_state)

    def generate_cohort(self) -> pd.DataFrame:
        """Generates multi-frame tabular 3D kinematic trajectories for 100 children."""
        self.logger.info(f"Generating synthetic 3D kinematic dataset for {self.config.total_children} children...")
        start_time = time.time()
        
        records: List[Dict[str, Any]] = []
        cohort_labels = [1] * self.config.n_asd + [0] * self.config.n_td
        child_ids = [f"ASD_{i+1:03d}" for i in range(self.config.n_asd)] + \
                    [f"TD_{j+1:03d}" for j in range(self.config.n_td)]
        
        permutation = self.rng.permutation(len(child_ids))
        child_ids = [child_ids[i] for i in permutation]
        cohort_labels = [cohort_labels[i] for i in permutation]
        
        for child_id, label in zip(child_ids, cohort_labels):
            n_frames = self.rng.randint(self.config.min_frames_per_child, self.config.max_frames_per_child + 1)
            child_scale = self.rng.uniform(0.90, 1.10)
            gait_freq = self.rng.uniform(0.8, 1.4)  # Hz
            
            is_asd = (label == 1)
            jerk_noise_scale = self.rng.uniform(0.015, 0.035) if is_asd else self.rng.uniform(0.003, 0.008)
            sway_variance = self.rng.uniform(0.025, 0.050) if is_asd else self.rng.uniform(0.005, 0.015)
            arm_asymmetry = self.rng.uniform(0.12, 0.30) if is_asd else self.rng.uniform(0.01, 0.05)
            occlusion_prob = 0.03 if is_asd else 0.01
            
            t = np.linspace(0, n_frames / self.config.sampling_rate_hz, n_frames)
            gait_phase = 2 * np.pi * gait_freq * t
            
            for f_idx in range(n_frames):
                frame_t = t[f_idx]
                phase = gait_phase[f_idx]
                
                pelvis_sway_x = np.sin(phase * 0.5) * (0.04 + sway_variance) + self.rng.normal(0, jerk_noise_scale)
                pelvis_bounce_z = np.sin(phase) * 0.02
                
                row: Dict[str, Any] = {
                    "child_id": child_id,
                    "frame_id": f_idx,
                    "timestamp": round(frame_t, 4),
                    "label": label,
                }
                
                for joint_name in self.JOINTS:
                    base_pos = self.BASE_SKELETON[joint_name] * child_scale
                    x, y, z = base_pos[0], base_pos[1], base_pos[2]
                    x += pelvis_sway_x
                    z += pelvis_bounce_z
                    
                    if "left" in joint_name:
                        leg_swing = np.sin(phase) * 0.12
                        arm_swing = -np.sin(phase) * (0.10 * (1.0 + arm_asymmetry))
                        if "wrist" in joint_name or "elbow" in joint_name:
                            y += arm_swing + np.sin(2.5 * phase) * (0.03 if is_asd else 0.005)
                        if "knee" in joint_name or "ankle" in joint_name:
                            y += leg_swing
                            z += np.maximum(0, np.cos(phase)) * 0.06
                    elif "right" in joint_name:
                        leg_swing = np.sin(phase + np.pi) * 0.12
                        arm_swing = -np.sin(phase + np.pi) * (0.10 * (1.0 - arm_asymmetry))
                        if "wrist" in joint_name or "elbow" in joint_name:
                            y += arm_swing
                        if "knee" in joint_name or "ankle" in joint_name:
                            y += leg_swing
                            z += np.maximum(0, np.cos(phase + np.pi)) * 0.06
                    elif "head" in joint_name or "neck" in joint_name:
                        x += np.sin(0.3 * phase) * (0.02 if not is_asd else 0.045)
                    
                    x += self.rng.normal(0, jerk_noise_scale)
                    y += self.rng.normal(0, jerk_noise_scale)
                    z += self.rng.normal(0, jerk_noise_scale)
                    
                    if self.rng.uniform(0, 1) < occlusion_prob and joint_name in ["wrist_left", "wrist_right", "ankle_left"]:
                        x, y, z = np.nan, np.nan, np.nan
                    
                    row[f"{joint_name}_x"] = x
                    row[f"{joint_name}_y"] = y
                    row[f"{joint_name}_z"] = z
                
                records.append(row)
        
        df = pd.DataFrame(records)
        elapsed = time.time() - start_time
        self.logger.info(
            f"Synthesized {len(df):,} movement frames across {len(child_ids)} children "
            f"({self.config.n_asd} ASD, {self.config.n_td} TD) in {elapsed:.2f}s."
        )
        return df


# =====================================================================
# 3. DOMAIN-SPECIFIC KINEMATIC FEATURE ENGINEERING
# =====================================================================

class KinematicFeatureExtractor:
    """
    Extracts clinically-grounded biomechanical feature sets from 3D skeletal data:
    1. Centered & Normalized Joint Coordinates (Invariant to camera frame).
    2. Inter-Joint Spatial Distances & Base of Support Width.
    3. 3D Joint Angles (Elbow, Knee, Shoulder, Hip flexion/extension, Torso tilt).
    4. First and Second Temporal Derivatives (Velocity, Acceleration, Movement Jerk).
    5. Bilateral Asymmetry Indices and Range of Motion (ROM) metrics.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    @staticmethod
    def _compute_distance(df: pd.DataFrame, j1: str, j2: str) -> pd.Series:
        dx = df[f"{j1}_x"] - df[f"{j2}_x"]
        dy = df[f"{j1}_y"] - df[f"{j2}_y"]
        dz = df[f"{j1}_z"] - df[f"{j2}_z"]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _compute_angle(df: pd.DataFrame, j_parent: str, j_mid: str, j_child: str) -> pd.Series:
        v1_x = df[f"{j_parent}_x"] - df[f"{j_mid}_x"]
        v1_y = df[f"{j_parent}_y"] - df[f"{j_mid}_y"]
        v1_z = df[f"{j_parent}_z"] - df[f"{j_mid}_z"]
        
        v2_x = df[f"{j_child}_x"] - df[f"{j_mid}_x"]
        v2_y = df[f"{j_child}_y"] - df[f"{j_mid}_y"]
        v2_z = df[f"{j_child}_z"] - df[f"{j_mid}_z"]
        
        dot = v1_x * v2_x + v1_y * v2_y + v1_z * v2_z
        norm_v1 = np.sqrt(v1_x * v1_x + v1_y * v1_y + v1_z * v1_z)
        norm_v2 = np.sqrt(v2_x * v2_x + v2_y * v2_y + v2_z * v2_z)
        
        denom = norm_v1 * norm_v2
        denom = np.where(denom == 0, np.nan, denom)
        cos_angle = np.clip(dot / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def extract_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw joint coordinates into a rich biomechanical feature matrix."""
        self.logger.info("Extracting biomechanical features (spatial distances, 3D angles, velocities, ROM)...")
        start_time = time.time()
        
        df = raw_df.copy()
        feat_dict: Dict[str, Any] = {}
        
        # --- 1. Root-Relative Coordinate Normalization ---
        root_x, root_y, root_z = df["pelvis_x"], df["pelvis_y"], df["pelvis_z"]
        joint_prefixes = [c[:-2] for c in df.columns if c.endswith("_x")]
        
        for joint in joint_prefixes:
            if joint != "pelvis":
                feat_dict[f"rel_{joint}_x"] = df[f"{joint}_x"] - root_x
                feat_dict[f"rel_{joint}_y"] = df[f"{joint}_y"] - root_y
                feat_dict[f"rel_{joint}_z"] = df[f"{joint}_z"] - root_z
        
        # --- 2. Inter-Joint Spatial Distances ---
        feat_dict["dist_wrist_L_wrist_R"] = self._compute_distance(df, "wrist_left", "wrist_right")
        feat_dict["dist_ankle_L_ankle_R"] = self._compute_distance(df, "ankle_left", "ankle_right")
        feat_dict["dist_elbow_L_elbow_R"] = self._compute_distance(df, "elbow_left", "elbow_right")
        feat_dict["dist_shoulder_L_wrist_L"] = self._compute_distance(df, "shoulder_left", "wrist_left")
        feat_dict["dist_shoulder_R_wrist_R"] = self._compute_distance(df, "shoulder_right", "wrist_right")
        feat_dict["dist_hand_L_head"] = self._compute_distance(df, "wrist_left", "head")
        feat_dict["dist_hand_R_head"] = self._compute_distance(df, "wrist_right", "head")
        feat_dict["dist_head_pelvis"] = self._compute_distance(df, "head", "pelvis")
        
        # --- 3. 3D Joint Angles ---
        angle_el_l = self._compute_angle(df, "shoulder_left", "elbow_left", "wrist_left")
        angle_el_r = self._compute_angle(df, "shoulder_right", "elbow_right", "wrist_right")
        angle_kn_l = self._compute_angle(df, "hip_left", "knee_left", "ankle_left")
        angle_kn_r = self._compute_angle(df, "hip_right", "knee_right", "ankle_right")
        
        feat_dict["angle_elbow_left"] = angle_el_l
        feat_dict["angle_elbow_right"] = angle_el_r
        feat_dict["angle_knee_left"] = angle_kn_l
        feat_dict["angle_knee_right"] = angle_kn_r
        feat_dict["angle_shoulder_left"] = self._compute_angle(df, "neck", "shoulder_left", "elbow_left")
        feat_dict["angle_shoulder_right"] = self._compute_angle(df, "neck", "shoulder_right", "elbow_right")
        feat_dict["angle_hip_left"] = self._compute_angle(df, "spine_torso", "hip_left", "knee_left")
        feat_dict["angle_hip_right"] = self._compute_angle(df, "spine_torso", "hip_right", "knee_right")
        
        # --- 4. Kinematic Asymmetries ---
        feat_dict["asym_elbow_angle"] = np.abs(angle_el_l - angle_el_r)
        feat_dict["asym_knee_angle"] = np.abs(angle_kn_l - angle_kn_r)
        feat_dict["asym_arm_reach"] = np.abs(feat_dict["dist_shoulder_L_wrist_L"] - feat_dict["dist_shoulder_R_wrist_R"])
        
        # --- 5. Temporal Dynamics (Group-aware derivatives strictly per child) ---
        dt = 1.0 / 30.0
        dynamic_cols = ["wrist_left", "wrist_right", "ankle_left", "ankle_right", "head", "pelvis"]
        for joint in dynamic_cols:
            for coord in ["x", "y", "z"]:
                col = f"{joint}_{coord}"
                vel_s = df.groupby("child_id")[col].diff() / dt
                acc_s = vel_s.groupby(df["child_id"]).diff() / dt
                jerk_s = acc_s.groupby(df["child_id"]).diff() / dt
                
                feat_dict[f"vel_{joint}_{coord}"] = vel_s
                feat_dict[f"acc_{joint}_{coord}"] = acc_s
                feat_dict[f"jerk_{joint}_{coord}"] = jerk_s
            
            feat_dict[f"vel_mag_{joint}"] = np.sqrt(
                feat_dict[f"vel_{joint}_x"]**2 + feat_dict[f"vel_{joint}_y"]**2 + feat_dict[f"vel_{joint}_z"]**2
            )
            feat_dict[f"jerk_mag_{joint}"] = np.sqrt(
                feat_dict[f"jerk_{joint}_x"]**2 + feat_dict[f"jerk_{joint}_y"]**2 + feat_dict[f"jerk_{joint}_z"]**2
            )
        
        # --- 6. Range of Motion (Rolling Window per child) ---
        feat_dict["rom_knee_L_w15"] = df.groupby("child_id").apply(
            lambda g: angle_kn_l.loc[g.index].rolling(15, min_periods=5).max() - angle_kn_l.loc[g.index].rolling(15, min_periods=5).min()
        ).reset_index(level=0, drop=True)
        feat_dict["rom_knee_R_w15"] = df.groupby("child_id").apply(
            lambda g: angle_kn_r.loc[g.index].rolling(15, min_periods=5).max() - angle_kn_r.loc[g.index].rolling(15, min_periods=5).min()
        ).reset_index(level=0, drop=True)
        
        new_feats_df = pd.DataFrame(feat_dict, index=df.index)
        full_df = pd.concat([df, new_feats_df], axis=1)

        elapsed = time.time() - start_time
        feature_cols = [c for c in full_df.columns if c not in ["child_id", "frame_id", "timestamp", "label", "class"]]
        self.logger.info(f"Feature engineering completed: {len(feature_cols)} total kinematic features generated in {elapsed:.2f}s.")
        return full_df


# =====================================================================
# 4. STRICT GROUP-AWARE PARTITIONING (ZERO DATA LEAKAGE)
# =====================================================================

@dataclass
class DatasetSplits:
    """Encapsulates strictly partitioned child-level train, validation, and test datasets."""
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    
    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.Series
    
    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series
    
    feature_names: List[str]
    train_children: List[str]
    val_children: List[str]
    test_children: List[str]


class GroupAwarePartitioner:
    """
    Enforces strict child-level data partitioning using stratified group splitting.
    Guarantees:
    - 50% Train (50 children: 25 ASD, 25 TD)
    - 25% Validation (25 children: 12-13 ASD, 12-13 TD)
    - 25% Test (25 children: 12-13 ASD, 12-13 TD)
    - Complete disjointness: No child appears in more than one partition.
    """

    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def partition(self, df: pd.DataFrame) -> DatasetSplits:
        self.logger.info("Executing strict child-level group partitioning (50% Train / 25% Val / 25% Test)...")
        
        child_metadata = df.groupby("child_id")["label"].first().reset_index()
        asd_children = child_metadata[child_metadata["label"] == 1]["child_id"].values
        td_children = child_metadata[child_metadata["label"] == 0]["child_id"].values
        
        rng = np.random.RandomState(self.config.random_state)
        rng.shuffle(asd_children)
        rng.shuffle(td_children)
        
        n_asd_train = int(len(asd_children) * self.config.train_ratio)
        n_td_train = int(len(td_children) * self.config.train_ratio)
        
        n_asd_val = int(len(asd_children) * self.config.val_ratio)
        n_td_val = int(len(td_children) * self.config.val_ratio)
        
        # Train partition (50% children)
        train_asd = asd_children[:n_asd_train]
        train_td = td_children[:n_td_train]
        train_children = list(train_asd) + list(train_td)
        
        # Validation partition (25% children)
        val_asd = asd_children[n_asd_train:n_asd_train + n_asd_val]
        val_td = td_children[n_td_train:n_td_train + n_td_val]
        val_children = list(val_asd) + list(val_td)
        
        # Final Test partition (Remaining 25% children)
        test_asd = asd_children[n_asd_train + n_asd_val:]
        test_td = td_children[n_td_train + n_td_val:]
        test_children = list(test_asd) + list(test_td)
        
        # Rigorous Disjointness Verifications
        set_tr, set_val, set_te = set(train_children), set(val_children), set(test_children)
        assert len(set_tr & set_val) == 0, "CRITICAL ERROR: Data leakage between Train and Val partitions!"
        assert len(set_tr & set_te) == 0, "CRITICAL ERROR: Data leakage between Train and Test partitions!"
        assert len(set_val & set_te) == 0, "CRITICAL ERROR: Data leakage between Val and Test partitions!"
        assert len(set_tr | set_val | set_te) == len(child_metadata), "Mismatch in total child partitioning count!"
        
        self.logger.info(
            f"Partition Summary: "
            f"Train={len(train_children)} children ({len(train_asd)} ASD, {len(train_td)} TD) | "
            f"Val={len(val_children)} children ({len(val_asd)} ASD, {len(val_td)} TD) | "
            f"Test={len(test_children)} children ({len(test_asd)} ASD, {len(test_td)} TD)"
        )
        
        train_mask = df["child_id"].isin(set_tr)
        val_mask = df["child_id"].isin(set_val)
        test_mask = df["child_id"].isin(set_te)
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        meta_cols = ["child_id", "frame_id", "timestamp", "label", "class"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        return DatasetSplits(
            X_train=train_df[feature_cols],
            y_train=train_df["label"],
            groups_train=train_df["child_id"],
            X_val=val_df[feature_cols],
            y_val=val_df["label"],
            groups_val=val_df["child_id"],
            X_test=test_df[feature_cols],
            y_test=test_df["label"],
            groups_test=test_df["child_id"],
            feature_names=feature_cols,
            train_children=train_children,
            val_children=val_children,
            test_children=test_children
        )


# =====================================================================
# 5. LEAK-FREE PREPROCESSING PIPELINE
# =====================================================================

class LeakFreePreprocessor:
    """
    Fits SimpleImputer and Scaler EXCLUSIVELY on the 50% Training Partition.
    Applies the learned parameters onto Validation and Test sets without backward leakage.
    """
    
    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.imputer = SimpleImputer(strategy=config.imputation_strategy)
        if config.scaling_strategy == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.feature_names: List[str] = []

    def fit_transform_train(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fits imputer and scaler ONLY on training data."""
        self.logger.info("Fitting imputer and scaler strictly on 50% training partition...")
        self.feature_names = list(X_train.columns)
        
        X_imp = self.imputer.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X_imp)
        return X_scaled

    def transform(self, X: pd.DataFrame, split_name: str = "Evaluation") -> np.ndarray:
        """Transforms validation or test partition using parameters learned from train."""
        self.logger.info(f"Transforming {split_name} partition using training transformers...")
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled


# =====================================================================
# 6. CANDIDATE MODELS & OPTUNA GROUP-KFOLD TUNING
# =====================================================================

class HyperparameterOptimizer:
    """
    Conducts Bayesian hyperparameter optimization using Optuna.
    Evaluates trials with 5-fold GroupKFold cross-validation on child_id
    to prevent intra-subject optimization leakage.
    """

    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def tune_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Tunes primary CatBoost Classifier."""
        self.logger.info(f"Starting CatBoost Bayesian tuning ({self.config.optuna_trials_primary} trials, 5-Fold GroupKFold)...")
        gkf = GroupKFold(n_splits=self.config.n_cv_folds)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 150, 400, step=50),
                "depth": trial.suggest_int("depth", 4, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "random_seed": self.config.random_state,
                "verbose": 0,
                "thread_count": -1
            }
            
            fold_aucs = []
            for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
                X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
                X_va_f, y_va_f = X_train[val_idx], y_train[val_idx]
                
                model = cb.CatBoostClassifier(**params)
                model.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), early_stopping_rounds=30, verbose=False)
                
                preds = model.predict_proba(X_va_f)[:, 1]
                auc = roc_auc_score(y_va_f, preds)
                fold_aucs.append(auc)
                
            return float(np.mean(fold_aucs))

        sampler = optuna.samplers.TPESampler(seed=self.config.optuna_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.config.optuna_trials_primary, show_progress_bar=False)
        
        self.logger.info(f"Optimal CatBoost Group-CV ROC-AUC: {study.best_value:.4f}")
        return study.best_params

    def tune_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Tunes benchmark LightGBM Classifier."""
        self.logger.info(f"Starting LightGBM Bayesian tuning ({self.config.optuna_trials_benchmark} trials, 5-Fold GroupKFold)...")
        gkf = GroupKFold(n_splits=self.config.n_cv_folds)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "random_state": self.config.random_state,
                "verbosity": -1,
                "n_jobs": -1
            }
            
            fold_aucs = []
            for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
                X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
                X_va_f, y_va_f = X_train[val_idx], y_train[val_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr_f, y_tr_f)
                
                preds = model.predict_proba(X_va_f)[:, 1]
                auc = roc_auc_score(y_va_f, preds)
                fold_aucs.append(auc)
                
            return float(np.mean(fold_aucs))

        sampler = optuna.samplers.TPESampler(seed=self.config.optuna_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.config.optuna_trials_benchmark, show_progress_bar=False)
        
        self.logger.info(f"Optimal LightGBM Group-CV ROC-AUC: {study.best_value:.4f}")
        return study.best_params

    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Tunes benchmark XGBoost Classifier if available in runtime."""
        if not XGB_AVAILABLE:
            self.logger.warning("XGBoost not available in runtime; skipping XGBoost optimization.")
            return None
            
        self.logger.info(f"Starting XGBoost Bayesian tuning ({self.config.optuna_trials_benchmark} trials, 5-Fold GroupKFold)...")
        gkf = GroupKFold(n_splits=self.config.n_cv_folds)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "random_state": self.config.random_state,
                "verbosity": 0,
                "n_jobs": -1
            }
            
            fold_aucs = []
            for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
                X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
                X_va_f, y_va_f = X_train[val_idx], y_train[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr_f, y_tr_f)
                
                preds = model.predict_proba(X_va_f)[:, 1]
                auc = roc_auc_score(y_va_f, preds)
                fold_aucs.append(auc)
                
            return float(np.mean(fold_aucs))

        sampler = optuna.samplers.TPESampler(seed=self.config.optuna_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.config.optuna_trials_benchmark, show_progress_bar=False)
        
        self.logger.info(f"Optimal XGBoost Group-CV ROC-AUC: {study.best_value:.4f}")
        return study.best_params


# =====================================================================
# 7. MODEL TRAINING & SOFT-VOTING ENSEMBLE
# =====================================================================

class ClinicalModelTrainer:
    """Trains primary, benchmark, and soft-voting ensemble models on the 50% training partition."""

    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.trained_models: Dict[str, Any] = {}

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        catboost_params: Dict[str, Any],
        lightgbm_params: Dict[str, Any],
        xgboost_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fits all candidate models on X_train and evaluates on validation set to build ensemble."""
        self.logger.info("Fitting candidate models on 50% training partition...")
        
        # 1. Primary Classifier: CatBoost
        cb_params_clean = catboost_params.copy()
        cb_params_clean["verbose"] = 0
        cb_model = cb.CatBoostClassifier(**cb_params_clean)
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40, verbose=False)
        self.trained_models["CatBoost"] = cb_model
        
        # 2. Benchmark 1: LightGBM
        lgb_model = lgb.LGBMClassifier(**lightgbm_params)
        lgb_model.fit(X_train, y_train)
        self.trained_models["LightGBM"] = lgb_model
        
        # 3. Benchmark 2: XGBoost (if available)
        if XGB_AVAILABLE and xgboost_params is not None:
            xgb_model = xgb.XGBClassifier(**xgboost_params)
            xgb_model.fit(X_train, y_train)
            self.trained_models["XGBoost"] = xgb_model
        
        # 4. Benchmark 3: Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.trained_models["RandomForest"] = rf_model
        
        # 5. Soft-Voting Ensemble (Combining top 2 best performers on validation partition)
        val_performances = {}
        for name, m in self.trained_models.items():
            val_preds = m.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            val_performances[name] = val_auc
            self.logger.info(f"Model [{name}] Validation ROC-AUC: {val_auc:.4f}")
            
        sorted_models = sorted(val_performances.items(), key=lambda x: x[1], reverse=True)
        top_2_names = [sorted_models[0][0], sorted_models[1][0]]
        self.logger.info(f"Building Soft-Voting Ensemble combining top 2 models: {top_2_names}")
        
        estimators = [(name, self.trained_models[name]) for name in top_2_names]
        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        ensemble.fit(X_train, y_train)
        self.trained_models["SoftVotingEnsemble"] = ensemble
        
        return self.trained_models


# =====================================================================
# 8. TRIAL-TO-CHILD AGGREGATION & PROBABILITY CALIBRATION
# =====================================================================

class FrameToChildAggregator:
    """
    Aggregates trial/frame-level predictions to child-level diagnostic outcomes:
    1. Aggregation Functions: Mean probability, Median, Standard Deviation, Majority Vote.
    2. Threshold Optimization: Calibrates the clinical decision threshold on the Validation Set.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.optimal_threshold: float = 0.50

    def aggregate_predictions(
        self,
        frame_probabilities: np.ndarray,
        child_groups: pd.Series,
        ground_truth: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Aggregates frame probabilities per child_id to produce child diagnostic decisions."""
        th = threshold if threshold is not None else self.optimal_threshold
        
        pred_df = pd.DataFrame({
            "child_id": child_groups.values,
            "frame_prob": frame_probabilities,
            "frame_label": ground_truth.values
        })
        
        grouped = pred_df.groupby("child_id").agg(
            child_prob_mean=("frame_prob", "mean"),
            child_prob_median=("frame_prob", "median"),
            child_prob_std=("frame_prob", lambda s: s.std() if len(s) > 1 else 0.0),
            child_prob_q75=("frame_prob", lambda s: np.percentile(s, 75)),
            child_prob_q25=("frame_prob", lambda s: np.percentile(s, 25)),
            child_frames_count=("frame_prob", "count"),
            true_label=("frame_label", "first")
        ).reset_index()
        
        grouped["pred_label"] = (grouped["child_prob_mean"] >= th).astype(int)
        grouped["pred_majority_vote"] = pred_df.groupby("child_id")["frame_prob"].apply(
            lambda s: (s >= 0.5).mean() >= 0.5
        ).astype(int).values
        
        return grouped

    def find_optimal_threshold(
        self,
        val_frame_probs: np.ndarray,
        val_groups: pd.Series,
        val_labels: pd.Series
    ) -> float:
        """Finds optimal decision threshold on Validation cohort maximizing Balanced Accuracy."""
        val_agg = self.aggregate_predictions(val_frame_probs, val_groups, val_labels, threshold=0.50)
        
        best_th = 0.50
        best_bacc = 0.0
        
        candidate_thresholds = np.linspace(0.20, 0.80, 61)
        for th in candidate_thresholds:
            preds = (val_agg["child_prob_mean"] >= th).astype(int)
            bacc = balanced_accuracy_score(val_agg["true_label"], preds)
            if bacc > best_bacc:
                best_bacc = bacc
                best_th = th
                
        self.optimal_threshold = best_th
        self.logger.info(f"Optimal child-level decision threshold calibrated to {self.optimal_threshold:.3f} (Val BACC: {best_bacc:.4f})")
        return self.optimal_threshold


class ClinicalProbabilityCalibrator:
    """
    Fits and evaluates post-hoc probability calibration:
    - Platt Scaling (Logistic Regression) or Isotonic Regression fit on the 25% Validation set.
    - Evaluates Expected Calibration Error (ECE) and Brier Score.
    """

    def __init__(self, method: str = "sigmoid", logger: Optional[logging.Logger] = None):
        self.method = method
        self.logger = logger
        self.calibrator: Optional[CalibratedClassifierCV] = None

    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
                
            bin_size = np.sum(bin_mask)
            if bin_size > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
                
        return float(ece)

    def fit_calibration(self, base_estimator: Any, X_val: np.ndarray, y_val: np.ndarray):
        """Calibrates model using validation partition."""
        if self.logger:
            self.logger.info(f"Calibrating classifier probabilities via {self.method} on validation partition...")
        self.calibrator = CalibratedClassifierCV(estimator=base_estimator, method=self.method, cv="prefit")
        self.calibrator.fit(X_val, y_val)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise ValueError("Calibrator must be fitted before predicting.")
        return self.calibrator.predict_proba(X)


# =====================================================================
# 9. COMPREHENSIVE CLINICAL METRICS & BOOTSTRAPPING
# =====================================================================

class ClinicalMetricsEngine:
    """
    Computes rigorous clinical diagnostic performance metrics:
    - Sensitivity / Recall (True Positive Rate)
    - Specificity (True Negative Rate)
    - Precision / PPV (Positive Predictive Value)
    - Negative Predictive Value (NPV)
    - Accuracy & Balanced Accuracy
    - F1-Score, ROC-AUC, PR-AUC
    - Expected Calibration Error (ECE) & Brier Score
    - Clustered Non-Parametric Bootstrap 95% Confidence Intervals
    """

    def __init__(self, logger: logging.Logger, n_bootstrap: int = 1000, seed: int = 42):
        self.logger = logger
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(seed)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, float]:
        """Computes point estimates of all clinical metrics."""
        y_pred = (y_prob >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.50
            
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = 0.50
            
        brier = brier_score_loss(y_true, y_prob)
        ece = ClinicalProbabilityCalibrator.calculate_ece(y_true, y_prob)
        
        return {
            "Accuracy": acc,
            "Balanced_Accuracy": bacc,
            "Sensitivity_Recall": sensitivity,
            "Specificity": specificity,
            "Precision_PPV": ppv,
            "NPV": npv,
            "F1_Score": f1,
            "ROC_AUC": auc,
            "PR_AUC": pr_auc,
            "ECE": ece,
            "Brier_Score": brier,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        }

    def compute_bootstrapped_ci(
        self,
        child_agg_df: pd.DataFrame,
        threshold: float = 0.50
    ) -> pd.DataFrame:
        """Computes 95% Bootstrap Confidence Intervals resampled at the child level."""
        n_children = len(child_agg_df)
        boot_metrics: Dict[str, List[float]] = {
            "Accuracy": [], "Balanced_Accuracy": [], "Sensitivity_Recall": [],
            "Specificity": [], "Precision_PPV": [], "F1_Score": [], "ROC_AUC": [], "PR_AUC": []
        }
        
        for _ in range(self.n_bootstrap):
            sample_df = child_agg_df.sample(n=n_children, replace=True, random_state=self.rng.randint(0, 1000000))
            if len(sample_df["true_label"].unique()) < 2:
                continue
                
            m = self.compute_metrics(
                y_true=sample_df["true_label"].values,
                y_prob=sample_df["child_prob_mean"].values,
                threshold=threshold
            )
            for k in boot_metrics.keys():
                boot_metrics[k].append(m[k])
                
        point_estimates = self.compute_metrics(
            y_true=child_agg_df["true_label"].values,
            y_prob=child_agg_df["child_prob_mean"].values,
            threshold=threshold
        )
        
        summary_rows = []
        for metric_name, values in boot_metrics.items():
            pt = point_estimates[metric_name]
            ci_low = np.percentile(values, 2.5) if len(values) > 0 else pt
            ci_high = np.percentile(values, 97.5) if len(values) > 0 else pt
            summary_rows.append({
                "Metric": metric_name,
                "Point_Estimate": pt,
                "CI_95_Lower": ci_low,
                "CI_95_Upper": ci_high,
                "Report_String": f"{pt:.3f} [{ci_low:.3f} - {ci_high:.3f}]"
            })
            
        return pd.DataFrame(summary_rows)


# =====================================================================
# 10. SHAP EXPLAINABILITY ENGINE
# =====================================================================

class ClinicalSHAPExplainer:
    """
    Computes SHapley Additive exPlanations (SHAP) using TreeExplainer on the CatBoost model.
    Generates clinical biomarker attribution rankings on unseen test cohort children.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None

    def explain(
        self,
        model: Any,
        X_test: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Computes SHAP values on unseen test partition and ranks top kinematic biomarkers."""
        self.logger.info("Computing SHAP explanations with TreeExplainer on unseen test cohort...")
        
        self.explainer = shap.TreeExplainer(model)
        shap_values_raw = self.explainer.shap_values(X_test)
        
        if isinstance(shap_values_raw, list):
            self.shap_values = shap_values_raw[1]
        elif len(shap_values_raw.shape) == 3:
            self.shap_values = shap_values_raw[:, :, 1]
        else:
            self.shap_values = shap_values_raw
            
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        
        self.logger.info("Top 5 Biomechanical Features by Mean |SHAP| Attribution:")
        for idx, row in importance_df.head(5).iterrows():
            self.logger.info(f"  {idx+1}. {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")
            
        return self.shap_values, importance_df


# =====================================================================
# 11. DIAGNOSTIC VISUALIZATION SUITE
# =====================================================================

class ClinicalVisualizer:
    """Generates publication-quality diagnostic charts and saves to disk."""

    def __init__(self, config: HarnessConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def plot_cohort_partitioning(self, splits: DatasetSplits):
        """Visualizes the strict 50/25/25 child-level data split."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        partitions = ["Train (50%)", "Validation (25%)", "Test (25%)"]
        n_asd = [len([c for c in splits.train_children if "ASD" in c]),
                 len([c for c in splits.val_children if "ASD" in c]),
                 len([c for c in splits.test_children if "ASD" in c])]
        n_td = [len([c for c in splits.train_children if "TD" in c]),
                len([c for c in splits.val_children if "TD" in c]),
                len([c for c in splits.test_children if "TD" in c])]
        
        x = np.arange(len(partitions))
        width = 0.35
        
        ax.bar(x - width/2, n_asd, width, label='ASD Cohort (label=1)', color='#d95f02', alpha=0.9)
        ax.bar(x + width/2, n_td, width, label='TD Control (label=0)', color='#1b9e77', alpha=0.9)
        
        ax.set_ylabel('Number of Children (Unique Subjects)', fontsize=12, fontweight='bold')
        ax.set_title('Strict Group-Aware Partitioning (Zero Data Leakage on child_id)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(partitions, fontsize=11, fontweight='bold')
        ax.legend(frameon=True, fontsize=11)
        ax.set_ylim(0, 30)
        
        for i in range(len(partitions)):
            ax.text(x[i] - width/2, n_asd[i] + 0.8, str(n_asd[i]), ha='center', fontweight='bold')
            ax.text(x[i] + width/2, n_td[i] + 0.8, str(n_td[i]), ha='center', fontweight='bold')
            
        plt.tight_layout()
        save_path = self.config.figures_dir / "1_group_split_cohort_distribution.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved cohort split plot -> {save_path.name}")

    def plot_confusion_matrices(
        self,
        frame_metrics: Dict[str, float],
        child_metrics: Dict[str, float]
    ):
        """Plots side-by-side Confusion Matrices for Trial-level and Child-level evaluations."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # Trial-level CM
        cm_frame = np.array([
            [int(frame_metrics["TN"]), int(frame_metrics["FP"])],
            [int(frame_metrics["FN"]), int(frame_metrics["TP"])]
        ])
        sns.heatmap(
            cm_frame, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['TD (0)', 'ASD (1)'], yticklabels=['TD (0)', 'ASD (1)'], cbar=False
        )
        axes[0].set_title(f"Trial-Level Confusion Matrix\n(Accuracy: {frame_metrics['Accuracy']:.3f}, Sens: {frame_metrics['Sensitivity_Recall']:.3f})", fontweight='bold')
        axes[0].set_xlabel('Predicted Movement Label', fontweight='bold')
        axes[0].set_ylabel('Ground Truth Movement Label', fontweight='bold')
        
        # Child-level CM
        cm_child = np.array([
            [int(child_metrics["TN"]), int(child_metrics["FP"])],
            [int(child_metrics["FN"]), int(child_metrics["TP"])]
        ])
        sns.heatmap(
            cm_child, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['TD (0)', 'ASD (1)'], yticklabels=['TD (0)', 'ASD (1)'], cbar=False
        )
        axes[1].set_title(f"Child-Level Diagnostic Confusion Matrix\n(BACC: {child_metrics['Balanced_Accuracy']:.3f}, Spec: {child_metrics['Specificity']:.3f})", fontweight='bold')
        axes[1].set_xlabel('Predicted Diagnostic Label', fontweight='bold')
        axes[1].set_ylabel('Ground Truth Diagnostic Label', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.config.figures_dir / "2_confusion_matrices.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved confusion matrices plot -> {save_path.name}")

    def plot_multimodel_roc_pr_curves(
        self,
        models_dict: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        child_agg_results: Dict[str, pd.DataFrame]
    ):
        """Plots multi-model ROC-AUC and PR-AUC curves comparing all models at child level."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        colors = {
            "CatBoost": "#d95f02",
            "LightGBM": "#7570b3",
            "XGBoost": "#e7298a",
            "RandomForest": "#66a61e",
            "SoftVotingEnsemble": "#e6ab02"
        }
        
        # 1. Child-Level ROC Curves
        for name, agg_df in child_agg_results.items():
            fpr, tpr, _ = roc_curve(agg_df["true_label"], agg_df["child_prob_mean"])
            auc = roc_auc_score(agg_df["true_label"], agg_df["child_prob_mean"])
            color = colors.get(name, "#333333")
            lw = 2.5 if name == "CatBoost" else 1.8
            axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=lw)
            
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Chance")
        axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
        axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
        axes[0].set_title('Child-Level Diagnostic ROC Curves', fontsize=13, fontweight='bold')
        axes[0].legend(loc="lower right", frameon=True)
        
        # 2. Child-Level Precision-Recall Curves
        for name, agg_df in child_agg_results.items():
            prec, rec, _ = precision_recall_curve(agg_df["true_label"], agg_df["child_prob_mean"])
            pr_auc = average_precision_score(agg_df["true_label"], agg_df["child_prob_mean"])
            color = colors.get(name, "#333333")
            lw = 2.5 if name == "CatBoost" else 1.8
            axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=lw)
            
        axes[1].set_xlabel('Recall / Sensitivity', fontweight='bold')
        axes[1].set_ylabel('Precision / Positive Predictive Value', fontweight='bold')
        axes[1].set_title('Child-Level Precision-Recall Curves', fontsize=13, fontweight='bold')
        axes[1].legend(loc="lower left", frameon=True)
        
        plt.tight_layout()
        save_path = self.config.figures_dir / "3_roc_pr_curves_multimodel.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved multi-model ROC/PR curves plot -> {save_path.name}")

    def plot_calibration_reliability(
        self,
        y_true: np.ndarray,
        raw_probs: np.ndarray,
        cal_probs: np.ndarray
    ):
        """Plots Reliability Diagram comparing raw vs calibrated predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, raw_probs, n_bins=8, strategy="uniform")
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=8, strategy="uniform")
        
        ece_raw = ClinicalProbabilityCalibrator.calculate_ece(y_true, raw_probs)
        ece_cal = ClinicalProbabilityCalibrator.calculate_ece(y_true, cal_probs)
        
        axes[0].plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        axes[0].plot(prob_pred_raw, prob_true_raw, "s-", color="#d95f02", label=f"Uncalibrated (ECE={ece_raw:.3f})")
        axes[0].plot(prob_pred_cal, prob_true_cal, "o-", color="#1b9e77", label=f"Platt Calibrated (ECE={ece_cal:.3f})")
        
        axes[0].set_xlabel("Mean Predicted Probability", fontweight='bold')
        axes[0].set_ylabel("Empirical True Fraction (ASD)", fontweight='bold')
        axes[0].set_title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight='bold')
        axes[0].legend(loc="lower right", frameon=True)
        
        axes[1].hist(raw_probs, range=(0, 1), bins=20, histtype="step", color="#d95f02", lw=2, label="Uncalibrated")
        axes[1].hist(cal_probs, range=(0, 1), bins=20, histtype="step", color="#1b9e77", lw=2, label="Platt Calibrated")
        axes[1].set_xlabel("Predicted Probability Confidence", fontweight='bold')
        axes[1].set_ylabel("Sample Count", fontweight='bold')
        axes[1].set_title("Probability Distribution Density", fontsize=12, fontweight='bold')
        axes[1].legend(loc="upper center", frameon=True)
        
        plt.tight_layout()
        save_path = self.config.figures_dir / "4_calibration_reliability_diagrams.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved calibration reliability plot -> {save_path.name}")

    def plot_child_level_diagnostic_separation(
        self,
        child_agg_df: pd.DataFrame,
        optimal_threshold: float
    ):
        """Visualizes predicted probability distributions for ASD vs TD children."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        plot_df = child_agg_df.copy()
        plot_df["Group"] = plot_df["true_label"].map({0: "Typically Developing (0)", 1: "ASD (1)"})
        group_palette = {"Typically Developing (0)": "#1b9e77", "ASD (1)": "#d95f02"}
        
        sns.boxplot(
            data=plot_df, x="Group", y="child_prob_mean",
            palette=group_palette, boxprops=dict(alpha=0.3), ax=ax, width=0.4
        )
        
        sns.stripplot(
            data=plot_df, x="Group", y="child_prob_mean",
            palette=group_palette, jitter=0.2, size=9, alpha=0.85, ax=ax
        )
        
        ax.axhline(optimal_threshold, color="red", linestyle="--", linewidth=2,
                   label=f"Calibrated Threshold ({optimal_threshold:.3f})")
        
        ax.set_xlabel("Clinical Cohort Group", fontweight='bold', fontsize=11)
        ax.set_ylabel("Aggregated Mean Predicted Probability", fontweight='bold', fontsize=11)
        ax.set_title("Child-Level Diagnostic Cohort Separation on Untouched Test Set", fontweight='bold', fontsize=13)
        ax.legend(loc="upper left", frameon=True)
        
        plt.tight_layout()
        save_path = self.config.figures_dir / "5_child_level_diagnostic_aggregation.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved child diagnostic separation plot -> {save_path.name}")

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        top_n: int = 15
    ):
        """Generates SHAP summary beeswarm and feature importance bar plots."""
        fig = plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(
            shap_values,
            features=X_test,
            feature_names=feature_names,
            max_display=top_n,
            show=False
        )
        plt.title(f"SHAP Biomechanical Feature Attributions (Top {top_n} Kinematic Markers)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path = self.config.figures_dir / "6_shap_beeswarm_biomarkers.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved SHAP beeswarm plot -> {save_path.name}")
        
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        y_pos = np.arange(top_n)
        ax.barh(y_pos, mean_abs[top_idx][::-1], color="#386cb0", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontweight='bold', fontsize=10)
        ax.set_xlabel("Mean |SHAP Value| (Average Impact on ASD Diagnostic Decision)", fontweight='bold', fontsize=11)
        ax.set_title(f"Top {top_n} Kinematic Biomarkers Ranked by SHAP Importance", fontweight='bold', fontsize=13)
        
        plt.tight_layout()
        save_path_bar = self.config.figures_dir / "7_shap_feature_importance_bar.png"
        fig.savefig(save_path_bar, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved SHAP bar plot -> {save_path_bar.name}")


# =====================================================================
# 12. END-TO-END PIPELINE ORCHESTRATOR
# =====================================================================

class ASDMovementClassificationHarness:
    """
    Orchestrates the end-to-end production-grade training, tuning, calibration,
    and diagnostic evaluation pipeline.
    """

    def __init__(self, config: Optional[HarnessConfig] = None):
        self.config = config if config is not None else HarnessConfig()
        self.logger = setup_logger(self.config.output_dir)
        
        self.loader = ClinicalDatasetLoader(self.config, self.logger)
        self.feature_extractor = KinematicFeatureExtractor(self.logger)
        self.partitioner = GroupAwarePartitioner(self.config, self.logger)
        self.preprocessor = LeakFreePreprocessor(self.config, self.logger)
        self.tuner = HyperparameterOptimizer(self.config, self.logger)
        self.trainer = ClinicalModelTrainer(self.config, self.logger)
        self.aggregator = FrameToChildAggregator(self.logger)
        self.metrics_engine = ClinicalMetricsEngine(self.logger, n_bootstrap=self.config.bootstrap_iterations)
        self.explainer = ClinicalSHAPExplainer(self.logger)
        self.visualizer = ClinicalVisualizer(self.config, self.logger)

    def run(self, raw_dataframe: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Executes full diagnostic training, tuning, calibration, and evaluation harness."""
        total_start = time.time()
        self.logger.info("=" * 80)
        self.logger.info("STARTING CLINICAL ASD MOVEMENT CLASSIFICATION HARNESS")
        self.logger.info("=" * 80)
        
        # 1. Dataset Loading or Ingestion
        if raw_dataframe is not None:
            raw_df = raw_dataframe.copy()
            is_precomputed = False
        else:
            raw_df, is_precomputed = self.loader.load_dataset()
            
        # 2. Kinematic Feature Extraction (if raw 3D skeletal data)
        if not is_precomputed:
            feature_df = self.feature_extractor.extract_features(raw_df)
        else:
            feature_df = raw_df
            
        # 3. Strict Child-Level Group Partitioning (50% Train, 25% Val, 25% Test)
        splits = self.partitioner.partition(feature_df)
        self.visualizer.plot_cohort_partitioning(splits)
        
        # 4. Leak-Free Preprocessing (Fit on Train ONLY)
        X_train_scaled = self.preprocessor.fit_transform_train(splits.X_train)
        X_val_scaled = self.preprocessor.transform(splits.X_val, split_name="Validation (25%)")
        X_test_scaled = self.preprocessor.transform(splits.X_test, split_name="Untouched Test (25%)")
        
        y_train = splits.y_train.values
        y_val = splits.y_val.values
        y_test = splits.y_test.values
        groups_train = splits.groups_train.values
        
        # 5. Bayesian Hyperparameter Optimization with GroupKFold
        cb_params = self.tuner.tune_catboost(X_train_scaled, y_train, groups_train, X_val_scaled, y_val)
        lgb_params = self.tuner.tune_lightgbm(X_train_scaled, y_train, groups_train)
        xgb_params = self.tuner.tune_xgboost(X_train_scaled, y_train, groups_train) if XGB_AVAILABLE else None
        
        # 6. Fit Candidate Models & Ensemble
        models = self.trainer.train_all_models(
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            catboost_params=cb_params,
            lightgbm_params=lgb_params,
            xgboost_params=xgb_params
        )
        
        # Save primary CatBoost model checkpoint
        primary_model = models["CatBoost"]
        primary_model_path = self.config.models_dir / "primary_catboost_model.cbm"
        primary_model.save_model(str(primary_model_path))
        self.logger.info(f"Persisted primary CatBoost model artifact -> {primary_model_path.name}")
        
        # 7. Probability Calibration on Validation Partition
        calibrator = ClinicalProbabilityCalibrator(method=self.config.calibration_method, logger=self.logger)
        calibrator.fit_calibration(primary_model, X_val_scaled, y_val)
        
        # 8. Trial-to-Child Aggregation & Threshold Optimization on Validation Set
        val_raw_probs = primary_model.predict_proba(X_val_scaled)[:, 1]
        val_cal_probs = calibrator.predict_proba(X_val_scaled)[:, 1]
        
        optimal_th = self.aggregator.find_optimal_threshold(val_cal_probs, splits.groups_val, splits.y_val)
        
        # 9. Evaluation on UNTOUCHED 25% Test Cohort
        self.logger.info("=" * 80)
        self.logger.info("FINAL EVALUATION ON UNTOUCHED 25% TEST COHORT")
        self.logger.info("=" * 80)
        
        test_raw_probs = primary_model.predict_proba(X_test_scaled)[:, 1]
        test_cal_probs = calibrator.predict_proba(X_test_scaled)[:, 1]
        
        # Trial-level metrics
        frame_metrics = self.metrics_engine.compute_metrics(y_test, test_cal_probs, threshold=0.50)
        
        # Child-level aggregation
        child_agg_test = self.aggregator.aggregate_predictions(
            frame_probabilities=test_cal_probs,
            child_groups=splits.groups_test,
            ground_truth=splits.y_test,
            threshold=optimal_th
        )
        
        child_metrics = self.metrics_engine.compute_metrics(
            y_true=child_agg_test["true_label"].values,
            y_prob=child_agg_test["child_prob_mean"].values,
            threshold=optimal_th
        )
        
        # Clustered Bootstrap 95% Confidence Intervals
        ci_df = self.metrics_engine.compute_bootstrapped_ci(child_agg_test, threshold=optimal_th)
        ci_table_path = self.config.tables_dir / "child_metrics_bootstrapped_ci.csv"
        ci_df.to_csv(ci_table_path, index=False)
        self.logger.info(f"Saved bootstrapped confidence intervals table -> {ci_table_path.name}")
        
        # Evaluate all candidate models at child level for comparison
        child_agg_results = {}
        candidate_summary_rows = []
        for name, model in models.items():
            preds = model.predict_proba(X_test_scaled)[:, 1]
            agg = self.aggregator.aggregate_predictions(preds, splits.groups_test, splits.y_test, threshold=optimal_th)
            child_agg_results[name] = agg
            m = self.metrics_engine.compute_metrics(agg["true_label"].values, agg["child_prob_mean"].values, threshold=optimal_th)
            candidate_summary_rows.append({
                "Model": name,
                "Child_Balanced_Accuracy": m["Balanced_Accuracy"],
                "Child_Sensitivity": m["Sensitivity_Recall"],
                "Child_Specificity": m["Specificity"],
                "Child_F1_Score": m["F1_Score"],
                "Child_ROC_AUC": m["ROC_AUC"],
                "Child_PR_AUC": m["PR_AUC"],
                "Child_ECE": m["ECE"]
            })
            
        candidate_df = pd.DataFrame(candidate_summary_rows).sort_values(by="Child_ROC_AUC", ascending=False)
        candidate_table_path = self.config.tables_dir / "candidate_models_benchmark.csv"
        candidate_df.to_csv(candidate_table_path, index=False)
        self.logger.info(f"Saved candidate model benchmark table -> {candidate_table_path.name}")
        
        # 10. Visualizations
        self.visualizer.plot_confusion_matrices(frame_metrics, child_metrics)
        self.visualizer.plot_multimodel_roc_pr_curves(models, X_test_scaled, y_test, child_agg_results)
        self.visualizer.plot_calibration_reliability(y_test, test_raw_probs, test_cal_probs)
        self.visualizer.plot_child_level_diagnostic_separation(child_agg_test, optimal_th)
        
        # 11. SHAP Explainability on Unseen Test Cohort
        shap_vals, imp_df = self.explainer.explain(primary_model, X_test_scaled, splits.feature_names)
        imp_path = self.config.tables_dir / "shap_feature_importance_ranking.csv"
        imp_df.to_csv(imp_path, index=False)
        self.visualizer.plot_shap_summary(shap_vals, X_test_scaled, splits.feature_names)
        
        total_time = time.time() - total_start
        self.logger.info("=" * 80)
        self.logger.info(f"HARNESS EXECUTION COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
        self.logger.info("=" * 80)
        
        # Console Summary
        print("\n" + "="*70)
        print(" CLINICAL PERFORMANCE SUMMARY ON UNTOUCHED 25% TEST SET")
        print("="*70)
        print(f"Cohort Size: {len(splits.test_children)} Unseen Children ({sum(child_agg_test['true_label']==1)} ASD, {sum(child_agg_test['true_label']==0)} TD)")
        print(f"Optimal Diagnostic Threshold: {optimal_th:.3f}")
        print("-" * 70)
        print(f"{'Metric':<25} | {'Point Estimate':<15} | {'95% CI (Bootstrap)':<25}")
        print("-" * 70)
        for _, r in ci_df.iterrows():
            print(f"{r['Metric']:<25} | {r['Point_Estimate']:<15.3f} | {r['Report_String']:<25}")
        print("="*70 + "\n")
        
        return {
            "splits": splits,
            "models": models,
            "primary_model": primary_model,
            "calibrator": calibrator,
            "optimal_threshold": optimal_th,
            "frame_metrics": frame_metrics,
            "child_metrics": child_metrics,
            "ci_df": ci_df,
            "candidate_df": candidate_df,
            "shap_importance_df": imp_df,
            "execution_time_seconds": total_time
        }


# =====================================================================
# 13. EXECUTION BLOCK
# =====================================================================

if __name__ == "__main__":
    print("Initializing Clinical ASD Kinematics Harness...")
    custom_config = HarnessConfig(
        data_path="Dataset/Final dataset.xlsx",
        use_synthetic_if_missing=True,
        total_children=100,
        n_asd=50,
        n_td=50,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        n_cv_folds=5,
        optuna_trials_primary=15,
        optuna_trials_benchmark=10,
        random_state=42,
        output_dir=Path("./asd_kinematics_output")
    )
    
    harness = ASDMovementClassificationHarness(config=custom_config)
    results = harness.run()
