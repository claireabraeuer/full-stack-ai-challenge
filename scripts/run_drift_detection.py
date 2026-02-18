"""Simulate drift detection on historical predictions."""

import json
from pathlib import Path

import pandas as pd

from src.data import load_splits
from src.models.categorization import load_production_model
from src.models.monitoring import DriftDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Simulate drift detection on test set."""
    logger.info("Loading data and model...")

    # Load data
    train_df, val_df, test_df = load_splits("data/splits")

    # Load production model
    model, metadata, loader = load_production_model()

    logger.info(
        f"Loaded model: {metadata['model_type']} version {metadata['version']}"
    )

    # Initialize drift detector with baseline metrics
    detector = DriftDetector(
        baseline_metrics=metadata.get("baseline_metrics"),
        confidence_threshold=0.85,
        accuracy_drop_threshold=0.05,
        window_size=100,
    )

    # Simulate predictions on test set (first 100 samples)
    logger.info("Simulating predictions on test set...")

    for idx, row in test_df.head(100).iterrows():
        ticket_df = pd.DataFrame([row])

        # Run prediction
        result = loader.predict(model, ticket_df, return_confidence=True)

        # Record for drift detection
        detector.record_prediction(
            ticket_id=row["ticket_id"],
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            true_category=row["category"],  # Ground truth available
        )

    # Run drift analysis
    logger.info("\n" + "=" * 50)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("=" * 50)

    results = detector.analyze(historical_df=train_df)

    # Print results
    conf_drift = results["confidence_drift"]
    logger.info(
        f"\n1. Confidence Drift: {conf_drift['drift_detected']}\n"
        f"   Mean confidence: {conf_drift.get('mean_confidence', 'N/A'):.3f}\n"
        f"   Threshold: {conf_drift['threshold']}"
    )

    acc_drift = results["accuracy_drift"]
    logger.info(
        f"\n2. Accuracy Drift: {acc_drift['drift_detected']}\n"
        f"   Rolling accuracy: {acc_drift.get('rolling_accuracy', 'N/A'):.3f}\n"
        f"   Baseline accuracy: {acc_drift.get('baseline_accuracy', 'N/A'):.3f}\n"
        f"   Drop: {acc_drift.get('drop', 0):.1%}"
    )

    if "distribution_drift" in results:
        dist_drift = results["distribution_drift"]
        logger.info(
            f"\n3. Distribution Drift:\n"
            f"   New categories: {len(dist_drift['new_categories'])}\n"
            f"   Surging categories: {len(dist_drift['surging_categories'])}"
        )

        if dist_drift["surging_categories"]:
            logger.info("   Top surges:")
            for surge in dist_drift["surging_categories"][:3]:
                logger.info(
                    f"     - {surge['category']}: "
                    f"{surge['baseline_pct']:.1f}% → {surge['recent_pct']:.1f}% "
                    f"({surge['increase_factor']:.1f}x)"
                )

    # Get statistics
    stats = detector.get_stats()
    logger.info(f"\n4. Prediction Statistics:")
    logger.info(f"   Total predictions: {stats['predictions_count']}")
    logger.info(f"   Mean confidence: {stats['confidence']['mean']:.3f}")
    logger.info(
        f"   Accuracy: {stats.get('accuracy', {}).get('rolling_accuracy', 'N/A'):.3f}"
    )

    # Save results
    output_path = Path("data/drift_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_path}")
    logger.info(
        "✓ Prometheus metrics exposed at /metrics endpoint (when API is built)"
    )


if __name__ == "__main__":
    main()
