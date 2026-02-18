"""Run anomaly detection on ticket data.

This script loads all ticket data (train+val+test), runs the AnomalyDetector,
and saves results to JSON.
"""

import json
from pathlib import Path

import pandas as pd

from src.data import load_splits
from src.models.anomaly import AnomalyDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run anomaly detection on full dataset."""
    # Load data splits
    logger.info("Loading ticket data...")
    train_df, val_df, test_df = load_splits("data/splits")

    # Combine for full historical view
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Loaded {len(full_df):,} tickets total")

    # Initialize detector with default thresholds
    detector = AnomalyDetector(
        volume_window=7,
        volume_z_threshold=2.5,
        baseline_days=30,
        recent_days=7,
        sentiment_threshold=0.3,
    )

    # Run detection
    logger.info("Running anomaly detection...")
    results = detector.analyze(full_df)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("ANOMALY DETECTION RESULTS")
    logger.info("=" * 50)

    logger.info(
        f"\n1. Volume Anomalies: {results['volume_anomalies']['total_flagged']} days flagged"
    )
    if results["volume_anomalies"]["anomalies"]:
        logger.info("   Recent volume spikes/drops:")
        for anomaly in results["volume_anomalies"]["anomalies"][:3]:
            logger.info(
                f"   - {anomaly['date']}: {anomaly['ticket_count']} tickets "
                f"(expected {anomaly['expected']:.0f}, z={anomaly['z_score']:.2f})"
            )

    logger.info(
        f"\n2. New Categories: {len(results['new_categories']['new_categories'])} truly new"
    )
    if results["new_categories"]["new_categories"]:
        logger.info(f"   Categories: {results['new_categories']['new_categories']}")

    logger.info(
        f"\n3. Surging Categories: {len(results['new_categories']['surging_categories'])} with 2x+ increase"
    )
    if results["new_categories"]["surging_categories"]:
        for surge in results["new_categories"]["surging_categories"][:3]:
            logger.info(
                f"   - {surge['category']}: "
                f"{surge['baseline_pct']:.1f}% → {surge['recent_pct']:.1f}% "
                f"({surge['increase_factor']:.1f}x)"
            )

    logger.info("\n4. Sentiment Shifts:")
    shifts = results["sentiment_shifts"]
    logger.info(
        f"   Sentiment: {shifts['prior_sentiment_avg']:.2f} → "
        f"{shifts['recent_sentiment_avg']:.2f} "
        f"(delta={shifts['sentiment_delta']:.2f}, "
        f"anomaly={shifts['sentiment_anomaly']})"
    )
    logger.info(
        f"   Satisfaction: {shifts['prior_satisfaction_avg']:.2f} → "
        f"{shifts['recent_satisfaction_avg']:.2f} "
        f"(delta={shifts['satisfaction_delta']:.2f}, "
        f"anomaly={shifts['satisfaction_anomaly']})"
    )

    logger.info("\n5. Top Emerging Problems:")
    emerging = results["emerging_problems"]
    if not emerging.empty:
        for idx, row in emerging.head(5).iterrows():
            logger.info(
                f"   - {row['product']} / {row['category']}: "
                f"score={row['anomaly_score']:.3f} "
                f"(tickets={row['ticket_count']}, "
                f"escalations={row['escalation_rate']:.1%}, "
                f"bugs={row['bug_rate']:.1%})"
            )
    else:
        logger.info("   No emerging problems detected")

    # Save results to JSON
    output_path = Path("data/anomaly_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving results to {output_path}...")
    with open(output_path, "w") as f:
        # Convert DataFrame to dict for JSON serialization
        output = results.copy()
        output["emerging_problems"] = results["emerging_problems"].to_dict(
            orient="records"
        )
        json.dump(output, f, indent=2)

    logger.info("✓ Anomaly detection complete!")
    logger.info(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
