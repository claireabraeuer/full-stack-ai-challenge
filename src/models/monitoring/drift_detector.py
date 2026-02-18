"""ML model drift detection using statistical methods.

Monitors three types of drift:
1. Prediction distribution drift (category proportions)
2. Confidence degradation (low prediction confidence)
3. Performance drift (accuracy drop when labels available)
"""

from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics (activate dormant dependency)
PREDICTION_CONFIDENCE = Gauge(
    "model_prediction_confidence_mean",
    "Rolling mean prediction confidence (last 100 predictions)",
)

PREDICTION_ACCURACY = Gauge(
    "model_prediction_accuracy_rolling",
    "Rolling accuracy when ground truth available (last 100 predictions)",
)

DRIFT_ALERTS_TOTAL = Counter(
    "model_drift_alerts_total", "Total drift alerts fired", ["alert_type"]
)


class DriftDetector:
    """Statistical drift detector for ML models.

    Detects three types of drift:
    1. Prediction distribution drift (category proportions change)
    2. Confidence degradation (prediction confidence drops)
    3. Performance drift (accuracy drops when labels available)

    Uses simple statistics - no ML training required.
    """

    def __init__(
        self,
        baseline_metrics: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.85,
        accuracy_drop_threshold: float = 0.05,
        window_size: int = 100,
        baseline_window_days: int = 30,
        recent_window_days: int = 7,
    ):
        """Initialize drift detector.

        Args:
            baseline_metrics: Dict with test_accuracy, test_f1_macro from training
            confidence_threshold: Alert if mean confidence drops below this
            accuracy_drop_threshold: Alert if accuracy drops more than this
            window_size: Rolling window size for recent predictions
            baseline_window_days: Historical baseline period (days)
            recent_window_days: Recent period to compare (days)
        """
        self.baseline_metrics = baseline_metrics or {}
        self.confidence_threshold = confidence_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.window_size = window_size
        self.baseline_window_days = baseline_window_days
        self.recent_window_days = recent_window_days

        # Rolling prediction log (in-memory for simplicity)
        self.predictions = deque(maxlen=window_size)

        logger.info(
            f"Initialized DriftDetector (confidence_threshold={confidence_threshold}, "
            f"accuracy_drop_threshold={accuracy_drop_threshold}, window_size={window_size})"
        )

        if baseline_metrics:
            logger.info(
                f"Baseline: accuracy={baseline_metrics.get('test_accuracy', 'N/A'):.4f}"
            )

    def record_prediction(
        self,
        ticket_id: str,
        predicted_category: str,
        confidence: float,
        true_category: Optional[str] = None,
    ):
        """Record a prediction for drift monitoring.

        Args:
            ticket_id: Unique ticket identifier
            predicted_category: Model prediction
            confidence: Prediction confidence score (0-1)
            true_category: Ground truth label (if available)
        """
        self.predictions.append(
            {
                "timestamp": datetime.now(),
                "ticket_id": ticket_id,
                "predicted_category": predicted_category,
                "confidence": confidence,
                "true_category": true_category,
                "correct": (
                    predicted_category == true_category if true_category else None
                ),
            }
        )

        # Update Prometheus metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update Prometheus gauges with current statistics."""
        if not self.predictions:
            return

        # Mean confidence
        confidences = [p["confidence"] for p in self.predictions]
        mean_conf = np.mean(confidences)
        PREDICTION_CONFIDENCE.set(mean_conf)

        # Rolling accuracy (only if we have ground truth)
        labeled = [p for p in self.predictions if p["true_category"] is not None]
        if labeled:
            correct = [p["correct"] for p in labeled]
            rolling_acc = np.mean(correct)
            PREDICTION_ACCURACY.set(rolling_acc)

    def detect_confidence_drift(self) -> Dict[str, Any]:
        """Detect if prediction confidence has degraded.

        Returns:
            Dict with:
            - mean_confidence: Current rolling mean
            - threshold: Configured threshold
            - drift_detected: Boolean alert flag
        """
        if len(self.predictions) < 10:
            return {
                "mean_confidence": None,
                "threshold": self.confidence_threshold,
                "drift_detected": False,
                "warning": "Insufficient predictions (need at least 10)",
            }

        confidences = [p["confidence"] for p in self.predictions]
        mean_conf = float(np.mean(confidences))

        drift = mean_conf < self.confidence_threshold

        if drift:
            DRIFT_ALERTS_TOTAL.labels(alert_type="confidence_low").inc()
            logger.error(
                f"DRIFT ALERT: Mean confidence {mean_conf:.3f} "
                f"below threshold {self.confidence_threshold}"
            )

        return {
            "mean_confidence": mean_conf,
            "std_confidence": float(np.std(confidences)),
            "threshold": self.confidence_threshold,
            "drift_detected": drift,
        }

    def detect_accuracy_drift(self) -> Dict[str, Any]:
        """Detect if model accuracy has dropped (requires labels).

        Returns:
            Dict with:
            - rolling_accuracy: Current accuracy
            - baseline_accuracy: Baseline from training
            - drop: Accuracy drop (negative = improvement)
            - drift_detected: Boolean alert flag
        """
        baseline_acc = self.baseline_metrics.get("test_accuracy")

        if baseline_acc is None:
            return {
                "rolling_accuracy": None,
                "baseline_accuracy": None,
                "drop": None,
                "drift_detected": False,
                "warning": "No baseline accuracy provided",
            }

        # Filter predictions with ground truth
        labeled = [p for p in self.predictions if p["true_category"] is not None]

        if len(labeled) < 10:
            return {
                "rolling_accuracy": None,
                "baseline_accuracy": baseline_acc,
                "drop": None,
                "drift_detected": False,
                "warning": f"Insufficient labeled predictions (have {len(labeled)}, need 10+)",
            }

        correct = [p["correct"] for p in labeled]
        rolling_acc = float(np.mean(correct))
        drop = baseline_acc - rolling_acc

        drift = drop > self.accuracy_drop_threshold

        if drift:
            DRIFT_ALERTS_TOTAL.labels(alert_type="accuracy_drop").inc()
            logger.error(
                f"DRIFT ALERT: Accuracy dropped {drop:.1%} "
                f"(baseline={baseline_acc:.3f}, current={rolling_acc:.3f})"
            )

        return {
            "rolling_accuracy": rolling_acc,
            "baseline_accuracy": baseline_acc,
            "drop": drop,
            "drift_detected": drift,
        }

    def detect_distribution_drift(self, historical_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if prediction category distribution has shifted.

        Compares recent predictions vs historical training distribution.
        Reuses the same pattern as AnomalyDetector.detect_new_categories().

        Args:
            historical_df: Historical training data with 'category' column

        Returns:
            Dict with:
            - new_categories: Categories in predictions but not in training
            - surging_categories: Categories with 2x+ proportion increase
        """
        if len(self.predictions) < 10:
            return {
                "new_categories": [],
                "surging_categories": [],
                "warning": "Insufficient predictions",
            }

        # Historical distribution from training data
        historical_dist = historical_df["category"].value_counts(normalize=True)

        # Recent predictions distribution
        recent_cats = [p["predicted_category"] for p in self.predictions]
        recent_series = pd.Series(recent_cats)
        recent_dist = recent_series.value_counts(normalize=True)

        # Truly new categories (not seen in training)
        new_cats = list(set(recent_dist.index) - set(historical_dist.index))

        # Surging categories (2x+ increase)
        surging = []
        for cat in recent_dist.index:
            base_pct = historical_dist.get(cat, 0)
            if base_pct > 0 and recent_dist[cat] / base_pct > 2.0:
                surging.append(
                    {
                        "category": cat,
                        "baseline_pct": float(base_pct * 100),
                        "recent_pct": float(recent_dist[cat] * 100),
                        "increase_factor": float(recent_dist[cat] / base_pct),
                    }
                )

        if new_cats or surging:
            DRIFT_ALERTS_TOTAL.labels(alert_type="distribution_shift").inc()
            logger.warning(
                f"Distribution drift: {len(new_cats)} new categories, "
                f"{len(surging)} surging categories"
            )

        return {
            "new_categories": new_cats,
            "surging_categories": surging,
        }

    def analyze(self, historical_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run all drift detectors and return structured results.

        Args:
            historical_df: Historical training data for distribution comparison

        Returns:
            Dict with results from all detectors + timestamp
        """
        logger.info(
            f"Running drift detection on {len(self.predictions)} predictions..."
        )

        results = {
            "confidence_drift": self.detect_confidence_drift(),
            "accuracy_drift": self.detect_accuracy_drift(),
            "timestamp": datetime.now().isoformat(),
            "predictions_analyzed": len(self.predictions),
        }

        if historical_df is not None:
            results["distribution_drift"] = self.detect_distribution_drift(
                historical_df
            )

        logger.info("Drift detection complete")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics of recent predictions.

        Returns:
            Dict with min/max/mean confidence, accuracy, category distribution
        """
        if not self.predictions:
            return {"predictions_count": 0}

        confidences = [p["confidence"] for p in self.predictions]
        categories = [p["predicted_category"] for p in self.predictions]
        labeled = [p for p in self.predictions if p["true_category"] is not None]

        stats = {
            "predictions_count": len(self.predictions),
            "confidence": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
            },
            "category_distribution": pd.Series(categories).value_counts().to_dict(),
        }

        if labeled:
            correct = [p["correct"] for p in labeled]
            stats["accuracy"] = {
                "labeled_count": len(labeled),
                "rolling_accuracy": float(np.mean(correct)),
            }

        return stats
