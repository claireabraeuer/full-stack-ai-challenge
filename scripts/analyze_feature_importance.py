"""Analyze and visualize feature importance from trained XGBoost model.

Shows which features are most important for ticket categorization.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_feature_importance():
    """Analyze feature importance from XGBoost model."""
    logger.info("Loading trained XGBoost model...")

    # Load model and preprocessor
    models_dir = Path("models/categorization")
    model = joblib.load(models_dir / "xgboost_category.pkl")
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")

    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importances from XGBoost
    importances = model.feature_importances_

    # Create DataFrame for analysis
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # Save to JSON
    output_path = Path("data/feature_importance.json")
    importance_data = {
        "total_features": len(feature_names),
        "top_20_features": importance_df.head(20).to_dict("records"),
        "feature_type_summary": {
            "tfidf_features": int(
                importance_df["feature"].str.startswith("tfidf__").sum()
            ),
            "categorical_features": int(
                importance_df["feature"].str.startswith("cat__").sum()
            ),
            "numerical_features": int(
                importance_df["feature"].str.startswith("num__").sum()
            ),
            "boolean_features": int(
                importance_df["feature"].str.startswith("bool__").sum()
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(importance_data, f, indent=2)

    logger.info(f"✓ Feature importance saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal Features: {len(feature_names)}")
    print("\nTop 20 Most Important Features:\n")

    for idx, row in importance_df.head(20).iterrows():
        feature_type = row["feature"].split("__")[0]
        feature_name = "__".join(row["feature"].split("__")[1:])
        print(f"{idx+1:2d}. [{feature_type:5s}] {feature_name:40s} {row['importance']:.6f}")

    print("\nFeature Type Summary:")
    for feat_type, count in importance_data["feature_type_summary"].items():
        print(f"  {feat_type:25s}: {count:4d} features")

    # Create visualization
    create_importance_plot(importance_df.head(20))

    return importance_df


def create_importance_plot(top_features_df):
    """Create bar plot of top features."""
    plt.figure(figsize=(12, 8))

    # Shorten feature names for readability
    labels = []
    for feat in top_features_df["feature"]:
        parts = feat.split("__")
        if len(parts) > 1:
            feat_type = parts[0]
            feat_name = "__".join(parts[1:])
            # Truncate long TF-IDF terms
            if len(feat_name) > 30:
                feat_name = feat_name[:27] + "..."
            labels.append(f"[{feat_type}] {feat_name}")
        else:
            labels.append(feat)

    y_pos = np.arange(len(labels))

    plt.barh(y_pos, top_features_df["importance"].values, color="steelblue")
    plt.yticks(y_pos, labels, fontsize=9)
    plt.xlabel("Feature Importance (Gain)", fontsize=11)
    plt.title("Top 20 Most Important Features for Ticket Categorization", fontsize=13, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path = Path("data/feature_importance.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Feature importance plot saved to {output_path}")

    print(f"\n✓ Visualization saved to {output_path}")


if __name__ == "__main__":
    analyze_feature_importance()
