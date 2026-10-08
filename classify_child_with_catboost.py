#!/usr/bin/env python3
"""
Classify a Child Using ONLY the Trained CatBoost Model
======================================================

This standalone script loads the saved CatBoost model artifact (`primary_catboost_model.cbm`)
and the fitted preprocessor (`preprocessor.joblib`), feeds in the 3D kinematic records
for a child, and outputs:
  - Movement trial probabilities
  - Aggregated diagnostic score
  - Final classification: ASD (label=1) vs Typically Developing (label=0)
  - Diagnostic confidence percentage

Usage Examples:
---------------
1. Classify a specific child by ID (e.g. ASD_005 or TD_012):
   python3 classify_with_catboost.py --child-id ASD_005
   python3 classify_with_catboost.py --child-id TD_012

2. Classify a random child from the dataset:
   python3 classify_with_catboost.py --random

3. Classify custom data from an Excel/CSV file:
   python3 classify_with_catboost.py --file path/to/my_child_data.xlsx
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import catboost as cb

MODEL_PATH = "asd_kinematics_output/models/primary_catboost_model.cbm"
PREPROCESSOR_PATH = "asd_kinematics_output/models/preprocessor.joblib"
DEFAULT_DATASET = "Dataset/Final dataset.xlsx"
DECISION_THRESHOLD = 0.50  # Standard decision threshold (or 0.630 if using calibrated probabilities)


def load_catboost_pipeline():
    """Loads CatBoost model and the fitted preprocessor."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"CatBoost model not found at '{MODEL_PATH}'. Run harness first.")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at '{PREPROCESSOR_PATH}'. Run harness first.")

    model = cb.CatBoostClassifier()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def load_dataset_with_ids(path=DEFAULT_DATASET):
    """Loads dataset and assigns child_id if not present."""
    df = pd.read_excel(path)
    if "child_id" not in df.columns:
        child_ids = []
        labels = []
        for i in range(50):
            for _ in range(8):
                child_ids.append(f"ASD_{i+1:03d}")
                labels.append(1)
        for j in range(50):
            for _ in range(8):
                child_ids.append(f"TD_{j+1:03d}")
                labels.append(0)
        df["child_id"] = child_ids
        df["label"] = labels
    return df


def classify_child(child_records: pd.DataFrame, model: cb.CatBoostClassifier, preprocessor, threshold: float = DECISION_THRESHOLD, child_name: str = "Input Child"):
    """
    Feeds a child's movement records into CatBoost and prints diagnostic results.
    """
    meta_cols = ["child_id", "label", "class", "frame_id", "timestamp"]
    feature_cols = [c for c in child_records.columns if c not in meta_cols]
    
    # Extract features and scale using preprocessor
    X_raw = child_records[feature_cols]
    X_scaled = preprocessor.transform(X_raw)
    
    # Predict with CatBoost
    trial_probs = model.predict_proba(X_scaled)[:, 1]
    
    # Compute aggregate diagnostic probability (mean across all trials)
    mean_prob = float(np.mean(trial_probs))
    std_prob = float(np.std(trial_probs))
    
    # Diagnostic decision
    is_asd = (mean_prob >= threshold)
    pred_label = 1 if is_asd else 0
    diagnosis = "Autism Spectrum Disorder (ASD, label=1)" if is_asd else "Typically Developing Peer (TD, label=0)"
    confidence = (mean_prob if is_asd else (1.0 - mean_prob)) * 100.0
    
    print("\n" + "=" * 70)
    print(f" CATBOOST DIAGNOSTIC REPORT: {child_name}")
    print("=" * 70)
    if "label" in child_records.columns:
        ground_truth = "ASD (label=1)" if child_records["label"].iloc[0] == 1 else "TD (label=0)"
        status = "[MATCH]" if pred_label == child_records["label"].iloc[0] else "[MISMATCH]"
        print(f"• Ground Truth          : {ground_truth}")
        print(f"• CatBoost Decision     : {diagnosis}  {status}")
    else:
        print(f"• CatBoost Decision     : {diagnosis}")
        
    print(f"• Aggregate ASD Score   : {mean_prob:.4f}  (Decision Threshold: {threshold:.2f})")
    print(f"• Classification Confidence: {confidence:.2f}%")
    print(f"• Trial Consistency     : +/- {std_prob:.4f} (variation across {len(trial_probs)} trials)")
    print("-" * 70)
    print("Individual Movement Trial Scores:")
    for idx, p in enumerate(trial_probs):
        tag = "ASD (+)" if p >= threshold else "TD (-)"
        print(f"  Trial {idx + 1:02d}: P(ASD) = {p:.4f}  ->  {tag}")
    print("=" * 70 + "\n")
    
    return {
        "predicted_label": pred_label,
        "diagnosis": diagnosis,
        "mean_probability": mean_prob,
        "confidence": confidence,
        "trial_probabilities": trial_probs.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a child using only the CatBoost model.")
    parser.add_argument("--child-id", type=str, default=None, help="Child ID to classify (e.g. ASD_005 or TD_012)")
    parser.add_argument("--random", action="store_true", help="Classify a random child from the dataset")
    parser.add_argument("--file", type=str, default=None, help="Path to custom Excel/CSV movement file")
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD, help=f"Decision threshold (default: {DECISION_THRESHOLD})")
    args = parser.parse_args()

    print("[INFO] Loading CatBoost model and preprocessor...")
    model, preprocessor = load_catboost_pipeline()

    if args.file:
        print(f"[INFO] Loading custom data from: {args.file}")
        df = pd.read_excel(args.file) if args.file.endswith(".xlsx") else pd.read_csv(args.file)
        classify_child(df, model, preprocessor, threshold=args.threshold, child_name=os.path.basename(args.file))
    else:
        df = load_dataset_with_ids(DEFAULT_DATASET)
        if args.child_id:
            child_df = df[df["child_id"] == args.child_id]
            if child_df.empty:
                print(f"[ERROR] Child ID '{args.child_id}' not found! Examples: ASD_001..ASD_050, TD_001..TD_050")
                sys.exit(1)
            classify_child(child_df, model, preprocessor, threshold=args.threshold, child_name=args.child_id)
        else:
            # Default to random child if no arguments passed
            random_child = np.random.choice(df["child_id"].unique())
            print(f"[INFO] Selecting random child: {random_child}")
            child_df = df[df["child_id"] == random_child]
            classify_child(child_df, model, preprocessor, threshold=args.threshold, child_name=random_child)


if __name__ == "__main__":
    main()
