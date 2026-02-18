"""Data loading utilities for support tickets."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.data.schemas import SupportTicket, TicketStats


class TicketDataLoader:
    """Load and parse support ticket data from JSON."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the data loader.

        Args:
            file_path: Path to the JSON file containing ticket data
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

    def load_raw(self) -> list[dict[str, Any]]:
        """Load raw JSON data.

        Returns:
            List of ticket dictionaries
        """
        logger.info(f"Loading data from {self.file_path}")
        with open(self.file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # If JSON is wrapped in a top-level key
            data = list(data.values())[0] if len(data) == 1 else data

        logger.info(f"Loaded {len(data):,} tickets")
        return data

    def load_as_dataframe(
        self, validate: bool = True, handle_errors: str = "raise"
    ) -> pd.DataFrame:
        """Load data as pandas DataFrame with optional validation.

        Args:
            validate: Whether to validate each ticket against schema
            handle_errors: How to handle validation errors ('raise', 'skip', 'coerce')

        Returns:
            DataFrame with ticket data
        """
        raw_data = self.load_raw()

        if validate:
            logger.info("Validating ticket data against schema")
            validated_data = []
            errors = []

            for i, ticket in enumerate(raw_data):
                try:
                    validated_ticket = SupportTicket(**ticket)
                    validated_data.append(validated_ticket.model_dump())
                except Exception as e:
                    errors.append((i, ticket.get("ticket_id", "unknown"), str(e)))
                    if handle_errors == "raise":
                        raise
                    elif handle_errors == "skip":
                        continue
                    # 'coerce' - add the original data despite validation error

            if errors:
                logger.warning(f"Validation errors in {len(errors)} tickets")
                if handle_errors == "skip":
                    logger.info(f"Skipped {len(errors)} invalid tickets")
                    raw_data = validated_data
                elif handle_errors == "coerce":
                    logger.info("Kept all tickets despite validation errors")
            else:
                raw_data = validated_data

        df = pd.DataFrame(raw_data)

        # Convert datetime strings to datetime objects
        datetime_cols = ["created_at", "updated_at", "resolved_at"]
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        logger.info(f"Created DataFrame with shape {df.shape}")
        return df

    def compute_statistics(self, df: pd.DataFrame) -> TicketStats:
        """Compute statistics about the ticket dataset.

        Args:
            df: DataFrame containing ticket data

        Returns:
            Statistics about the dataset
        """
        stats = TicketStats(
            total_tickets=len(df),
            date_range=(df["created_at"].min(), df["created_at"].max()),
            categories=df["category"].value_counts().to_dict(),
            subcategories=df["subcategory"].value_counts().to_dict(),
            products=df["product"].value_counts().to_dict(),
            customer_tiers=df["customer_tier"].value_counts().to_dict(),
            avg_resolution_time_hours=df["resolution_time_hours"].mean(),
            avg_satisfaction_score=df["satisfaction_score"].mean(),
            missing_values=df.isnull().sum().to_dict(),
        )

        logger.info(f"Dataset spans from {stats.date_range[0]} to {stats.date_range[1]}")
        logger.info(f"Average resolution time: {stats.avg_resolution_time_hours:.2f} hours")
        logger.info(f"Average satisfaction: {stats.avg_satisfaction_score:.2f}/5.0")

        return stats
