"""Statistical anomaly detector for ticket monitoring.

This module provides simple, interpretable anomaly detection using basic
statistical methods (z-scores, rolling averages, proportion comparisons).
No ML models required - works immediately on any dataset.
"""

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Statistical anomaly detector for ticket monitoring.

    Detects four types of anomalies:
    1. Volume spikes/drops (z-score on rolling window)
    2. New/emerging categories (proportion comparison)
    3. Sentiment deterioration (rolling average comparison)
    4. Emerging problems (multi-signal scoring)

    All methods use simple statistics - no ML training required.
    """

    def __init__(
        self,
        volume_window: int = 7,
        volume_z_threshold: float = 2.5,
        baseline_days: int = 30,
        recent_days: int = 7,
        sentiment_threshold: float = 0.3,
    ):
        """Initialize anomaly detector with configurable thresholds.

        Args:
            volume_window: Rolling window size (days) for volume anomalies
            volume_z_threshold: Z-score threshold for flagging volume anomalies
            baseline_days: Historical baseline period (days) for comparison
            recent_days: Recent period (days) to analyze
            sentiment_threshold: Threshold for sentiment/satisfaction drops
        """
        self.volume_window = volume_window
        self.volume_z_threshold = volume_z_threshold
        self.baseline_days = baseline_days
        self.recent_days = recent_days
        self.sentiment_threshold = sentiment_threshold

        logger.info(
            f"Initialized AnomalyDetector (volume_window={volume_window}, "
            f"z_threshold={volume_z_threshold}, baseline={baseline_days}d, "
            f"recent={recent_days}d, sentiment_threshold={sentiment_threshold})"
        )

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all anomaly detectors and return structured results.

        Args:
            df: DataFrame with ticket data (must have created_at column)

        Returns:
            Dictionary with results from all detectors:
            - volume_anomalies: Unusual daily ticket counts
            - new_categories: Emerging issue types
            - sentiment_shifts: Customer satisfaction deterioration
            - emerging_problems: Multi-signal incident indicators
            - timestamp: Analysis timestamp
        """
        logger.info(f"Running anomaly detection on {len(df):,} tickets...")

        # Ensure created_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
            df = df.copy()
            df["created_at"] = pd.to_datetime(df["created_at"])

        results = {
            "volume_anomalies": self.detect_volume_anomalies(df),
            "new_categories": self.detect_new_categories(df),
            "sentiment_shifts": self.detect_sentiment_shifts(df),
            "emerging_problems": self.detect_emerging_problems(df),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Anomaly detection complete")
        return results

    def detect_volume_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect days with abnormal ticket volumes using z-scores.

        Method: Calculate z-score for each day's count using rolling mean/std.
        Flag days where |z-score| > threshold (default 2.5 sigma).

        Args:
            df: DataFrame with created_at column

        Returns:
            Dict with:
            - anomalies: List of dicts with date, z_score, count, expected, top_categories
            - total_flagged: Number of anomalous days detected
        """
        # Resample to daily counts
        daily = df.set_index("created_at").resample("D").size()

        # Rolling statistics
        rolling_mean = daily.rolling(self.volume_window, min_periods=3).mean()
        rolling_std = daily.rolling(self.volume_window, min_periods=3).std()

        # Z-scores (handle zero std)
        z_scores = (daily - rolling_mean) / rolling_std.replace(0, np.nan)

        # Find anomalies
        anomalies = z_scores[z_scores.abs() > self.volume_z_threshold]

        # Build detailed results
        results = []
        for date in anomalies.index:
            day_df = df[df["created_at"].dt.date == date.date()]
            cat_counts = day_df["category"].value_counts()

            results.append(
                {
                    "date": date.date().isoformat(),
                    "z_score": float(anomalies[date]),
                    "ticket_count": int(daily[date]),
                    "expected": float(rolling_mean[date]),
                    "deviation": int(daily[date] - rolling_mean[date]),
                    "top_categories": cat_counts.head(3).to_dict(),
                }
            )

        logger.info(f"Volume anomalies: {len(results)} days flagged")
        return {"anomalies": results, "total_flagged": len(results)}

    def detect_new_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find categories that are new or surging (2x increase).

        Method: Compare category distributions in recent vs baseline periods.
        Flag categories that:
        1. Appear in recent period but not in baseline (truly new)
        2. Have 2x+ proportion increase (surging)

        Args:
            df: DataFrame with created_at and category columns

        Returns:
            Dict with:
            - new_categories: List of category names (not in baseline)
            - surging_categories: List of dicts with category, baseline_pct,
              recent_pct, increase_factor
            - baseline_period_days: Baseline window size
            - recent_period_days: Recent window size
        """
        cutoff = df["created_at"].max()
        recent_mask = df["created_at"] >= cutoff - pd.Timedelta(days=self.recent_days)
        baseline_mask = (
            df["created_at"]
            >= cutoff - pd.Timedelta(days=self.baseline_days + self.recent_days)
        ) & ~recent_mask

        baseline_dist = df[baseline_mask]["category"].value_counts(normalize=True)
        recent_dist = df[recent_mask]["category"].value_counts(normalize=True)

        # Truly new categories
        new_cats = list(set(recent_dist.index) - set(baseline_dist.index))

        # Surging categories (2x+ increase)
        surging = []
        for cat in recent_dist.index:
            base_pct = baseline_dist.get(cat, 0)
            if base_pct > 0 and recent_dist[cat] / base_pct > 2.0:
                surging.append(
                    {
                        "category": cat,
                        "baseline_pct": float(base_pct * 100),
                        "recent_pct": float(recent_dist[cat] * 100),
                        "increase_factor": float(recent_dist[cat] / base_pct),
                    }
                )

        logger.info(
            f"New categories: {len(new_cats)} new, {len(surging)} surging (2x+)"
        )
        return {
            "new_categories": new_cats,
            "surging_categories": surging,
            "baseline_period_days": self.baseline_days,
            "recent_period_days": self.recent_days,
        }

    def detect_sentiment_shifts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect drops in sentiment or satisfaction scores.

        Method: Compare recent vs prior rolling window averages.
        Flag if drop exceeds threshold (default 0.3).

        Args:
            df: DataFrame with customer_sentiment, satisfaction_score, created_at

        Returns:
            Dict with:
            - sentiment_delta: Change in sentiment score
            - satisfaction_delta: Change in satisfaction score
            - sentiment_anomaly: Boolean flag (drop > threshold)
            - satisfaction_anomaly: Boolean flag (drop > threshold)
            - recent/prior averages for both metrics
        """
        # Sentiment mapping (customize based on actual values in data)
        SENTIMENT_MAP = {
            "positive": 1.0,
            "neutral": 0.0,
            "frustrated": -1.0,
            "angry": -2.0,
        }

        df_work = df.copy()
        df_work["sent_num"] = df_work["customer_sentiment"].map(SENTIMENT_MAP).fillna(0)
        df_work = df_work.sort_values("created_at")

        # Aggregate by day
        daily_sent = df_work.set_index("created_at").resample("D")["sent_num"].mean()
        daily_sat = (
            df_work.set_index("created_at").resample("D")["satisfaction_score"].mean()
        )

        # Compare recent vs prior windows
        window = self.volume_window
        if len(daily_sent) < 2 * window:
            logger.warning(
                f"Insufficient data for sentiment shift detection "
                f"(need {2*window} days, have {len(daily_sent)})"
            )
            return {
                "sentiment_delta": 0.0,
                "satisfaction_delta": 0.0,
                "sentiment_anomaly": False,
                "satisfaction_anomaly": False,
                "recent_sentiment_avg": 0.0,
                "recent_satisfaction_avg": 0.0,
                "prior_sentiment_avg": 0.0,
                "prior_satisfaction_avg": 0.0,
                "warning": "Insufficient data",
            }

        recent_sent = daily_sent.iloc[-window:].mean()
        prior_sent = daily_sent.iloc[-2 * window : -window].mean()
        recent_sat = daily_sat.iloc[-window:].mean()
        prior_sat = daily_sat.iloc[-2 * window : -window].mean()

        sent_delta = recent_sent - prior_sent
        sat_delta = recent_sat - prior_sat

        # Flag anomalies (negative delta = worsening)
        sent_anomaly = float(prior_sent - recent_sent) > self.sentiment_threshold
        sat_anomaly = float(prior_sat - recent_sat) > self.sentiment_threshold

        logger.info(
            f"Sentiment shifts: sentiment_delta={sent_delta:.3f}, "
            f"satisfaction_delta={sat_delta:.3f}"
        )

        return {
            "sentiment_delta": float(sent_delta),
            "satisfaction_delta": float(sat_delta),
            "sentiment_anomaly": sent_anomaly,
            "satisfaction_anomaly": sat_anomaly,
            "recent_sentiment_avg": float(recent_sent),
            "recent_satisfaction_avg": float(recent_sat),
            "prior_sentiment_avg": float(prior_sent),
            "prior_satisfaction_avg": float(prior_sat),
        }

    def detect_emerging_problems(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find product+category combinations showing incident signals.

        Method: Group by product+category, aggregate escalations, bugs,
        recurrence, affected users. Compute weighted anomaly score.
        Return top 10 by score.

        Args:
            df: DataFrame with product, category, escalated, bug_report_filed,
                similar_issues_last_30_days, affected_users, created_at

        Returns:
            DataFrame with columns:
            - product, category
            - ticket_count, escalation_rate, bug_rate, avg_similar_issues,
              avg_affected_users
            - anomaly_score (0-1)
            Sorted by anomaly_score descending, limited to top 10
        """
        cutoff = df["created_at"].max()
        recent = df[df["created_at"] >= cutoff - pd.Timedelta(days=self.recent_days)]

        if len(recent) == 0:
            logger.warning("No recent tickets for emerging problem detection")
            return pd.DataFrame()

        # Group by product + category
        grouped = (
            recent.groupby(["product", "category"])
            .agg(
                ticket_count=("ticket_id", "count"),
                escalation_rate=("escalated", "mean"),
                bug_rate=("bug_report_filed", "mean"),
                avg_similar_issues=("similar_issues_last_30_days", "mean"),
                avg_affected_users=("affected_users", "mean"),
            )
            .reset_index()
        )

        # Composite anomaly score (0-1 scale)
        # Equal weighting: 25% each for escalations, bugs, recurrence, volume
        grouped["anomaly_score"] = (
            grouped["escalation_rate"] * 0.25
            + grouped["bug_rate"] * 0.25
            + (grouped["avg_similar_issues"] / 50).clip(0, 1) * 0.25
            + (grouped["ticket_count"] / grouped["ticket_count"].max()) * 0.25
        )

        # Return top 10
        top_problems = grouped.sort_values("anomaly_score", ascending=False).head(10)

        logger.info(
            f"Emerging problems: analyzed {len(grouped)} product+category "
            f"combinations, returning top 10"
        )

        return top_problems
