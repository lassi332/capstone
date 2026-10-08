#!/usr/bin/env python3
"""
Interactive Clinical ASD Diagnostic Classifier & Inference CLI
===============================================================

Use this interactive CLI tool to test, evaluate, and classify individual children or custom movement files
using the trained CatBoost clinical model, leak-free RobustScaler preprocessor, and Platt probability calibrator.

Examples of How to Run:
-----------------------
1. Classify a random child from the cohort:
   python3 classify_child.py --random

2. Classify a specific child by ID (e.g. ASD_005, ASD_012, TD_008, TD_025):
   python3 classify_child.py --child-id ASD_005

3. Classify all 26 unseen test children (held-out test evaluation):
   python3 classify_child.py --test-cohort

4. Classify all 100 children in the cohort:
   python3 classify_child.py --all

5. Launch interactive menu mode:
   python3 classify_child.py --interactive
"""

import os
import sys
import json
import joblib
import argparse
from typing import Tuple, Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
import catboost as cb

MODEL_PATH = "asd_kinematics_output/models/primary_catboost_model.cbm"
PREPROCESSOR_PATH = "asd_kinematics_output/models/preprocessor.joblib"
CALIBRATOR_PATH = "asd_kinematics_output/models/calibrator.joblib"
METADATA_PATH = "asd_kinematics_output/models/metadata.json"
DATASET_PATH = "Dataset/Final dataset.xlsx"


class ClinicalPredictor:
    """Encapsulates the preprocessor, base model, calibrator, and decision logic."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run asd_movement_classification_harness.py first.")
        
        self.model = cb.CatBoostClassifier()
        self.model.load_model(MODEL_PATH)
        
        self.preprocessor = joblib.load(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else None
        self.calibrator = joblib.load(CALIBRATOR_PATH) if os.path.exists(CALIBRATOR_PATH) else None
        
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                self.metadata = json.load(f)
            self.threshold = self.metadata.get("optimal_threshold", 0.630)
            self.test_children = set(self.metadata.get("test_children", []))
        else:
            self.metadata = {}
            self.threshold = 0.630
            self.test_children = set()

    def predict_dataframe(self, df_records: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns raw probabilities and calibrated probabilities for input records."""
        meta_cols = ["child_id", "label", "class", "frame_id", "timestamp"]
        feature_cols = [c for c in df_records.columns if c not in meta_cols]
        
        X_raw = df_records[feature_cols]
        if self.preprocessor is not None:
            X_scaled = self.preprocessor.transform(X_raw)
        else:
            X_scaled = X_raw.values
            
        raw_probs = self.model.predict_proba(X_scaled)[:, 1]
        
        if self.calibrator is not None:
            cal_probs = self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            cal_probs = raw_probs
            
        return raw_probs, cal_probs


def load_dataset(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads dataset and assigns child_id and binary labels."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    
    df = pd.read_excel(dataset_path)
    if "child_id" not in df.columns:
        child_ids = []
        labels = []
        for i in range(50):
            for r in range(8):
                child_ids.append(f"ASD_{i+1:03d}")
                labels.append(1)
        for j in range(50):
            for r in range(8):
                child_ids.append(f"TD_{j+1:03d}")
                labels.append(0)
        df["child_id"] = child_ids
        df["label"] = labels
    return df


def classify_child(child_id: str, df: pd.DataFrame, predictor: ClinicalPredictor, threshold: float = None):
    """Performs trial-level and aggregated diagnostic classification for a specific child."""
    th = threshold if threshold is not None else predictor.threshold
    child_df = df[df["child_id"] == child_id].copy()
    
    if child_df.empty:
        print(f"\n[ERROR] Child ID '{child_id}' not found in dataset!")
        available = df["child_id"].unique()
        print(f"Available Child IDs: {list(available[:4])} ... {list(available[-4:])}")
        return

    is_in_test_set = child_id in predictor.test_children
    set_badge = "[HELD-OUT 25% UNSEEN TEST COHORT]" if is_in_test_set else "[TRAIN/VAL COHORT]"
    
    true_label = child_df["label"].iloc[0]
    true_diagnosis = "ASD (Autism Spectrum Disorder, label=1)" if true_label == 1 else "TD (Typically Developing Peer, label=0)"
    
    raw_probs, cal_probs = predictor.predict_dataframe(child_df)
    mean_prob = float(np.mean(cal_probs))
    std_prob = float(np.std(cal_probs))
    
    pred_label = 1 if mean_prob >= th else 0
    pred_diagnosis = "ASD (Autism Spectrum Disorder, label=1)" if pred_label == 1 else "TD (Typically Developing Peer, label=0)"
    is_correct = (pred_label == true_label)
    
    confidence = (mean_prob if pred_label == 1 else (1.0 - mean_prob)) * 100.0
    
    print("\n" + "="*75)
    print(f" CLINICAL DIAGNOSTIC CLASSIFICATION REPORT: {child_id} {set_badge}")
    print("="*75)
    print(f"• Ground Truth Diagnosis    : {true_diagnosis}")
    print(f"• Model Diagnostic Decision : {pred_diagnosis}")
    print(f"• Diagnostic Status         : {'CORRECT [MATCH]' if is_correct else 'INCORRECT [MISMATCH]'}")
    print(f"• Aggregated ASD Probability : {mean_prob:.4f} (Calibrated Threshold = {th:.3f})")
    print(f"• Diagnostic Confidence     : {confidence:.2f}%")
    print(f"• Movement Trial Volatility  : +/- {std_prob:.4f} (std across {len(cal_probs)} trials)")
    print("-" * 75)
    print("Trial-by-Trial Movement Predictions (8 Movement Sequences):")
    for idx, (p_raw, p_cal) in enumerate(zip(raw_probs, cal_probs)):
        trial_status = "ASD Positive (+)" if p_cal >= th else "TD Control (-)"
        print(f"   Trial {idx+1}: Calibrated P(ASD) = {p_cal:.4f} (Raw = {p_raw:.4f}) -> {trial_status}")
    print("="*75 + "\n")


def classify_cohort_group(df: pd.DataFrame, predictor: ClinicalPredictor, only_test: bool = False):
    """Evaluates and reports metrics for a cohort group."""
    if only_test and predictor.test_children:
        eval_df = df[df["child_id"].isin(predictor.test_children)].copy()
        title = "HELD-OUT 25% UNSEEN TEST COHORT (26 CHILDREN)"
    else:
        eval_df = df.copy()
        title = "FULL COHORT DIAGNOSTIC CLASSIFICATION (100 CHILDREN)"
        
    results = []
    for child_id, group in eval_df.groupby("child_id"):
        _, cal_probs = predictor.predict_dataframe(group)
        mean_prob = float(np.mean(cal_probs))
        true_label = group["label"].iloc[0]
        pred_label = 1 if mean_prob >= predictor.threshold else 0
        results.append({
            "child_id": child_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "mean_prob": mean_prob,
            "correct": (true_label == pred_label)
        })
        
    res_df = pd.DataFrame(results)
    total_acc = (res_df["correct"].mean()) * 100.0
    asd_subset = res_df[res_df["true_label"] == 1]
    td_subset = res_df[res_df["true_label"] == 0]
    
    sens = (asd_subset["correct"].mean()) * 100.0
    spec = (td_subset["correct"].mean()) * 100.0
    bacc = (sens + spec) / 2.0
    
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)
    print(f"• Total Children Evaluated : {len(res_df)} children")
    print(f"• Overall Accuracy         : {total_acc:.2f}% ({res_df['correct'].sum()}/{len(res_df)})")
    print(f"• Balanced Accuracy        : {bacc:.2f}%")
    print(f"• Clinical Sensitivity     : {sens:.2f}% ({asd_subset['correct'].sum()}/{len(asd_subset)} ASD identified)")
    print(f"• Clinical Specificity     : {spec:.2f}% ({td_subset['correct'].sum()}/{len(td_subset)} TD identified)")
    print("="*70 + "\n")


def interactive_menu(df: pd.DataFrame, predictor: ClinicalPredictor):
    """Interactive console prompt for hands-on experimentation."""
    while True:
        print("\n" + "="*50)
        print(" CLINICAL ASD KINEMATIC DIAGNOSTIC TOOL")
        print("="*50)
        print("1. Classify a Random Child from Cohort")
        print("2. Classify Specific Child by ID (e.g. ASD_005, TD_018)")
        print("3. Classify Held-Out 25% Unseen Test Cohort (26 children)")
        print("4. Classify Entire 100-Child Dataset")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()
        
        if choice == "1":
            chosen = np.random.choice(df["child_id"].unique())
            print(f"\n[INFO] Randomly selected: {chosen}")
            classify_child(chosen, df, predictor)
        elif choice == "2":
            cid = input("Enter Child ID (e.g. ASD_001 to ASD_050, TD_001 to TD_050): ").strip()
            classify_child(cid, df, predictor)
        elif choice == "3":
            classify_cohort_group(df, predictor, only_test=True)
        elif choice == "4":
            classify_cohort_group(df, predictor, only_test=False)
        elif choice in ["5", "exit", "quit", "q"]:
            print("Exiting diagnostic tool. Goodbye!")
            break
        else:
            print("Invalid selection. Please enter 1, 2, 3, 4, or 5.")


def main():
    parser = argparse.ArgumentParser(description="Clinical ASD Kinematics Diagnostic Classifier")
    parser.add_argument("--random", action="store_true", help="Classify a random child from dataset")
    parser.add_argument("--child-id", type=str, default=None, help="Specific child ID (e.g., ASD_005 or TD_018)")
    parser.add_argument("--test-cohort", action="store_true", help="Classify all 26 unseen test children")
    parser.add_argument("--all", action="store_true", help="Classify all 100 children and print summary")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive menu mode")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold override")
    
    args = parser.parse_args()
    predictor = ClinicalPredictor()
    df = load_dataset()
    
    if args.random:
        chosen = np.random.choice(df["child_id"].unique())
        classify_child(chosen, df, predictor, threshold=args.threshold)
    elif args.child_id:
        classify_child(args.child_id, df, predictor, threshold=args.threshold)
    elif args.test_cohort:
        classify_cohort_group(df, predictor, only_test=True)
    elif args.all:
        classify_cohort_group(df, predictor, only_test=False)
    else:
        interactive_menu(df, predictor)


if __name__ == "__main__":
    main()
