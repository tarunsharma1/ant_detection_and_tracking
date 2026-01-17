#!/usr/bin/env python3
"""Trajectory classifier for identifying excavating vs non-excavating ants.

This script trains a binary classifier on precomputed trajectory features to
distinguish excavating ants from non-excavating ants. Labels are sourced from
manual annotations, and features are loaded from the comprehensive trajectory
analysis dataset. Evaluation uses leave-one-video-out cross-validation to
measure how well the classifier generalises to unseen videos. Performance is
reported with and without the ``nest_proximity`` feature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANNOTATION_FILE = (
    "/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/"
    "excavating_annotation_beer-tree-08-01-2024_to_08-10-2024.csv"
)

FEATURE_FILE = (
    "/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/"
    "beer-tree-08-01-2024_to_08-10-2024/comprehensive_trajectory_analysis.csv"
)

MODEL_OUTPUT_PATH = (
    "/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/models/"
    "trajectory_classifier_with_nest.joblib"
)

PREDICTION_OUTPUT_PATH = (
    "/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/models/"
    "trajectory_classifier_predictions.csv"
)

SCORE_HISTOGRAM_PATH = (
    "/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/models/"
    "trajectory_classifier_score_histogram.png"
)

# Core features that are always considered
BASE_FEATURE_COLUMNS = [
    "efficiency",
    "straight_distance",
    "avg_angle_change",
    "total_length",
]

NEST_PROXIMITY_FEATURE = "nest_proximity"


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> pd.DataFrame:
    """Load and clean the manual excavation annotations."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")

    annotations = pd.read_csv(path)

    required_columns = {"video_id", "ant_id", "excavating"}
    missing = required_columns - set(annotations.columns)
    if missing:
        raise ValueError(
            "Annotation file is missing required columns: " + ", ".join(sorted(missing))
        )

    annotations = annotations.dropna(subset=["video_id", "ant_id", "excavating"])
    annotations["excavating"] = (
        annotations["excavating"].astype(str).str.strip().str.lower()
    )

    valid_mask = annotations["excavating"].isin({"yes", "no"})
    filtered = annotations.loc[valid_mask].copy()
    filtered["label"] = filtered["excavating"].map({"yes": 1, "no": 0})

    if filtered.empty:
        raise ValueError("No valid annotations found after filtering for Yes/No values.")

    filtered["video_id"] = filtered["video_id"].astype(str)
    filtered["ant_id"] = filtered["ant_id"].astype(int)

    return filtered


def load_features(path: str, ant_ids: List[int] | None = None) -> pd.DataFrame:
    """Load trajectory features, optionally filtering to specific ant IDs."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory feature file not found: {path}")

    features = pd.read_csv(path)

    required_columns = {
        "trajectory_index",
        "ant_id",
        "video_id",
        "trajectory_length",
        "efficiency",
        "straight_distance",
        "avg_angle_change",
        "total_length",
    }

    missing = required_columns - set(features.columns)
    if missing:
        raise ValueError(
            "Feature file is missing required columns: " + ", ".join(sorted(missing))
        )

    features["video_id"] = features["video_id"].astype(str)
    features["ant_id"] = features["ant_id"].astype(int)

    if ant_ids is not None:
        subset = features.loc[features["ant_id"].isin(ant_ids)].copy()
        if subset.empty:
            raise ValueError(
                "No matching entries found in feature file for the annotated ants."
            )
        return subset

    return features


def prepare_dataset(annotation_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Merge annotations with trajectory features on video and ant ID."""

    merged = annotation_df.merge(
        feature_df,
        on=["video_id", "ant_id"],
        how="inner",
        suffixes=("_ann", "_feat"),
    )

    if merged.empty:
        raise ValueError("Merged dataset is empty. Check that IDs match between files.")

    merged = merged.dropna(subset=BASE_FEATURE_COLUMNS)

    return merged


# ---------------------------------------------------------------------------
# Modelling utilities
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    video_id: str
    n_samples: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def build_classifier() -> Pipeline:
    """Create a scikit-learn pipeline for classification."""

    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    return pipeline


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> Dict[str, float]:
    """Compute evaluation metrics for a single fold."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }

    if y_prob is not None and np.unique(y_true).size == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


def leave_one_video_out_cv(
    dataset: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[List[FoldResult], pd.DataFrame]:
    """Perform leave-one-video-out cross validation."""

    fold_results: List[FoldResult] = []
    videos = sorted(dataset["video_id"].unique())

    for video_id in videos:
        train_df = dataset[dataset["video_id"] != video_id]
        test_df = dataset[dataset["video_id"] == video_id]

        if test_df.empty:
            continue

        if train_df["label"].nunique() < 2:
            print(
                f"⚠️ Skipping fold for video {video_id}: training data lacks both classes."
            )
            continue

        model = build_classifier()
        model.fit(train_df[feature_columns], train_df["label"])

        y_pred = model.predict(test_df[feature_columns])
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(test_df[feature_columns])[:, 1]

        metrics = evaluate_fold(test_df["label"].values, y_pred, y_prob)

        fold_results.append(
            FoldResult(
                video_id=str(video_id),
                n_samples=len(test_df),
                accuracy=metrics["accuracy"],
                balanced_accuracy=metrics["balanced_accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                roc_auc=metrics["roc_auc"],
            )
        )

    results_df = pd.DataFrame([result.__dict__ for result in fold_results])

    return fold_results, results_df


def summarise_results(results_df: pd.DataFrame) -> pd.Series:
    """Compute mean performance across folds, ignoring NaNs."""

    summary = results_df.drop(columns=["video_id", "n_samples"], errors="ignore").mean()
    summary.name = "mean"
    return summary


def print_evaluation_summary(title: str, results_df: pd.DataFrame) -> None:
    """Pretty-print per-fold metrics and aggregated summary."""

    if results_df.empty:
        print(f"❌ {title}: no evaluation results available.")
        return

    summary = summarise_results(results_df)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("-" * 80)
    print("Summary (mean across folds):")
    print(summary.to_frame().T.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def plot_score_distribution(scores: pd.Series, output_path: str) -> None:
    """Plot histogram of excavation scores in 0.1 bins and save to disk."""

    clean_scores = scores.dropna()

    if clean_scores.empty:
        print("⚠️ No valid scores available for plotting histogram.")
        return

    bins = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clean_scores, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Excavation probability score")
    ax.set_ylabel("Number of trajectories")
    ax.set_title("Distribution of Excavation Classifier Scores")
    ax.set_xticks(bins)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved score distribution histogram to {output_path}")


def run_inference_on_all_trajectories(
    full_feature_df: pd.DataFrame,
    model: Pipeline,
    feature_columns: List[str],
    output_path: str,
) -> None:
    """Run model inference on all trajectories and save predictions."""

    predictions_df = full_feature_df.copy()

    # Ensure required feature columns exist
    missing = [col for col in feature_columns if col not in predictions_df.columns]
    if missing:
        raise ValueError(
            "Feature dataframe missing required columns for prediction: "
            + ", ".join(missing)
        )

    # Prepare column for scores
    score_column = "excavating_score"
    predictions_df[score_column] = np.nan

    # Determine which rows have all required features present
    available_mask = predictions_df[feature_columns].notna().all(axis=1)

    if available_mask.any():
        available_features = predictions_df.loc[available_mask, feature_columns]
        scores = model.predict_proba(available_features)[:, 1]
        predictions_df.loc[available_mask, score_column] = scores
    else:
        print("⚠️ No trajectories had complete feature data for inference. Skipping scores.")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)
    print(f"✅ Saved trajectory predictions (with scores) to {output_path}")

    if available_mask.any():
        plot_score_distribution(predictions_df[score_column], SCORE_HISTOGRAM_PATH)


# ---------------------------------------------------------------------------
# Threshold analysis utilities
# ---------------------------------------------------------------------------

def evaluate_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    """Evaluate precision/recall metrics at different probability thresholds."""

    records = []
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        accuracy = accuracy_score(labels, preds)

        records.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    print("🔍 Loading annotations and features...")
    annotations = load_annotations(ANNOTATION_FILE)
    full_feature_df = load_features(FEATURE_FILE)
    features = full_feature_df.loc[
        full_feature_df["ant_id"].isin(annotations["ant_id"])
    ].copy()

    if features.empty:
        raise ValueError(
            "No annotated trajectories found in the feature file. Check inputs."
        )

    dataset = prepare_dataset(annotations, features)

    print(f"✅ Merged dataset contains {len(dataset)} trajectories")
    print(
        dataset["label"].value_counts()
        .rename(index={1: "Excavating", 0: "Non-excavating"})
        .to_string()
    )
    print(f"Unique videos in dataset: {sorted(dataset['video_id'].unique())}")

    available_base_features = [
        feature for feature in BASE_FEATURE_COLUMNS if feature in dataset.columns
    ]

    if not available_base_features:
        raise ValueError("No base feature columns were found in the dataset.")

    print(f"Using base feature columns: {available_base_features}")

    features_without_nest = available_base_features.copy()
    features_with_nest = available_base_features.copy()

    if NEST_PROXIMITY_FEATURE in dataset.columns:
        features_with_nest.append(NEST_PROXIMITY_FEATURE)
    else:
        print(
            "⚠️ nest_proximity feature not found in dataset. Skipping comparison using it."
        )

    # Evaluate without nest proximity
    print("\n🚫 Evaluating without nest proximity feature...")
    dataset_without_nest = dataset.dropna(subset=features_without_nest)
    _, results_without_nest = leave_one_video_out_cv(
        dataset_without_nest, features_without_nest
    )
    print_evaluation_summary(
        "Performance WITHOUT nest_proximity", results_without_nest
    )

    # Evaluate with nest proximity (if available)
    if NEST_PROXIMITY_FEATURE in features_with_nest:
        print("\n✅ Evaluating with nest proximity feature...")
        dataset_with_nest = dataset.dropna(subset=features_with_nest)
        _, results_with_nest = leave_one_video_out_cv(
            dataset_with_nest, features_with_nest
        )
        print_evaluation_summary(
            "Performance WITH nest_proximity", results_with_nest
        )

        if not results_without_nest.empty and not results_with_nest.empty:
            comparison = (
                summarise_results(results_with_nest)
                - summarise_results(results_without_nest)
            )
            print("\n📈 Difference (with - without):")
            print(
                comparison.to_frame().T.to_string(
                    index=False, float_format=lambda x: f"{x:+.3f}"
                )
            )

        if not dataset_with_nest.empty:
            print("\n💾 Training final model with nest proximity feature on full dataset...")
            final_model = build_classifier()
            final_model.fit(
                dataset_with_nest[features_with_nest],
                dataset_with_nest["label"],
            )

            # Analyse precision/recall at different thresholds on the full dataset
            print("\n🎯 Threshold analysis on full dataset (with nest proximity)")
            thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.05)]
            probabilities = final_model.predict_proba(
                dataset_with_nest[features_with_nest]
            )[:, 1]
            threshold_metrics = evaluate_thresholds(
                probabilities,
                dataset_with_nest["label"].values,
                thresholds,
            )

            if not threshold_metrics.empty:
                print(
                    threshold_metrics.to_string(
                        index=False, float_format=lambda x: f"{x:.3f}"
                    )
                )
                best_row = threshold_metrics.loc[
                    threshold_metrics["f1"].idxmax()
                ]
                print(
                    "\n🏆 Best threshold by F1: "
                    f"{best_row['threshold']:.2f} (Precision={best_row['precision']:.3f}, "
                    f"Recall={best_row['recall']:.3f}, F1={best_row['f1']:.3f})"
                )
            else:
                print("⚠️ Unable to compute threshold metrics (no valid predictions).")

            output_dir = os.path.dirname(MODEL_OUTPUT_PATH)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            joblib.dump(final_model, MODEL_OUTPUT_PATH)
            print(f"✅ Saved final model to {MODEL_OUTPUT_PATH}")

            run_inference_on_all_trajectories(
                full_feature_df,
                final_model,
                features_with_nest,
                PREDICTION_OUTPUT_PATH,
            )


if __name__ == "__main__":
    main()


