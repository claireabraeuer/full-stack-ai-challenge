"""Data quality validation and monitoring."""

from typing import Any

import pandas as pd
from loguru import logger


class DataQualityReport:
    """Data quality validation report."""

    def __init__(self) -> None:
        """Initialize empty report."""
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.metrics: dict[str, Any] = {}

    def add_error(self, message: str) -> None:
        """Add an error to the report."""
        self.errors.append(message)
        logger.error(f"Data Quality Error: {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning to the report."""
        self.warnings.append(message)
        logger.warning(f"Data Quality Warning: {message}")

    def add_info(self, message: str) -> None:
        """Add info to the report."""
        self.info.append(message)
        logger.info(f"Data Quality Info: {message}")

    def is_valid(self) -> bool:
        """Check if data passed validation (no errors)."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """Generate summary of the report."""
        return (
            f"Data Quality Report:\n"
            f"  Errors: {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Info: {len(self.info)}\n"
        )


class DataValidator:
    """Validate data quality for support tickets."""

    def __init__(
        self,
        max_missing_ratio: float = 0.1,
        min_rows: int = 1000,
    ) -> None:
        """Initialize validator.

        Args:
            max_missing_ratio: Maximum allowed ratio of missing values per column
            min_rows: Minimum number of rows required
        """
        self.max_missing_ratio = max_missing_ratio
        self.min_rows = min_rows

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """Run all validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Data quality report
        """
        report = DataQualityReport()

        # Run all checks
        self._check_row_count(df, report)
        self._check_duplicates(df, report)
        self._check_missing_values(df, report)
        self._check_target_variables(df, report)
        self._check_data_types(df, report)
        self._check_value_ranges(df, report)
        self._check_temporal_consistency(df, report)
        self._check_class_imbalance(df, report)

        logger.info(report.summary())
        return report

    def _check_row_count(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check if dataset has minimum number of rows."""
        if len(df) < self.min_rows:
            report.add_error(
                f"Insufficient data: {len(df)} rows (minimum: {self.min_rows})"
            )
        else:
            report.add_info(f"Dataset has {len(df):,} rows")

    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for duplicate ticket IDs."""
        if "ticket_id" in df.columns:
            duplicates = df["ticket_id"].duplicated().sum()
            if duplicates > 0:
                report.add_error(f"Found {duplicates} duplicate ticket IDs")
            else:
                report.add_info("No duplicate ticket IDs found")

    def _check_missing_values(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for missing values in critical columns."""
        critical_columns = [
            "ticket_id",
            "category",
            "subcategory",
            "description",
            "resolution",
        ]

        for col in critical_columns:
            if col not in df.columns:
                report.add_error(f"Critical column missing: {col}")
                continue

            missing_ratio = df[col].isnull().sum() / len(df)
            report.metrics[f"missing_ratio_{col}"] = missing_ratio

            if missing_ratio > 0:
                report.add_error(
                    f"Critical column '{col}' has {missing_ratio*100:.1f}% missing values"
                )

        # Check optional columns
        optional_missing = {}
        for col in df.columns:
            if col not in critical_columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > self.max_missing_ratio:
                    optional_missing[col] = missing_ratio

        if optional_missing:
            report.add_warning(
                f"Columns with >{self.max_missing_ratio*100}% missing: "
                f"{list(optional_missing.keys())}"
            )

    def _check_target_variables(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check target variables for classification."""
        targets = ["category", "subcategory"]

        for target in targets:
            if target not in df.columns:
                report.add_error(f"Target variable missing: {target}")
                continue

            n_classes = df[target].nunique()
            report.metrics[f"n_classes_{target}"] = n_classes
            report.add_info(f"Target '{target}' has {n_classes} unique classes")

            # Check for very rare classes
            value_counts = df[target].value_counts()
            rare_classes = value_counts[value_counts < 10]

            if len(rare_classes) > 0:
                report.add_warning(
                    f"Target '{target}' has {len(rare_classes)} classes with <10 samples"
                )

    def _check_data_types(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check if columns have expected data types."""
        datetime_cols = ["created_at", "updated_at", "resolved_at"]
        numeric_cols = ["resolution_time_hours", "satisfaction_score", "account_monthly_value"]

        for col in datetime_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                report.add_warning(f"Column '{col}' should be datetime type")

        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                report.add_warning(f"Column '{col}' should be numeric type")

    def _check_value_ranges(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check if values are within expected ranges."""
        # Satisfaction score should be 1-5
        if "satisfaction_score" in df.columns:
            invalid = df[
                (df["satisfaction_score"] < 1) | (df["satisfaction_score"] > 5)
            ].shape[0]
            if invalid > 0:
                report.add_error(
                    f"satisfaction_score has {invalid} values outside range [1,5]"
                )

        # Resolution time should be positive
        if "resolution_time_hours" in df.columns:
            invalid = df[df["resolution_time_hours"] < 0].shape[0]
            if invalid > 0:
                report.add_error(
                    f"resolution_time_hours has {invalid} negative values"
                )

    def _check_temporal_consistency(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """Check temporal consistency."""
        if all(col in df.columns for col in ["created_at", "resolved_at"]):
            inconsistent = df[df["resolved_at"] < df["created_at"]].shape[0]
            if inconsistent > 0:
                report.add_error(
                    f"{inconsistent} tickets resolved before creation"
                )

    def _check_class_imbalance(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for severe class imbalance."""
        if "category" in df.columns:
            value_counts = df["category"].value_counts()
            max_class = value_counts.iloc[0]
            min_class = value_counts.iloc[-1]
            imbalance_ratio = max_class / min_class

            report.metrics["class_imbalance_ratio"] = imbalance_ratio

            if imbalance_ratio > 100:
                report.add_warning(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    "Consider resampling techniques."
                )
            elif imbalance_ratio > 10:
                report.add_info(
                    f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
                )
