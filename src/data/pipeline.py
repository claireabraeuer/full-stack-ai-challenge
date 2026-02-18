"""Main data pipeline for loading, validating, and splitting ticket data."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config import settings
from src.data.loader import TicketDataLoader
from src.data.schemas import DataSplit, TicketStats
from src.data.splitter import DataSplitter
from src.data.validator import DataValidator


class DataPipeline:
    """End-to-end data pipeline."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        validate_quality: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            data_path: Path to raw JSON data file
            output_dir: Directory for output files
            validate_quality: Whether to run data quality checks
        """
        self.data_path = Path(data_path or settings.ticket_data_path)
        self.output_dir = Path(output_dir or "data/splits")
        self.validate_quality = validate_quality

        self.loader = TicketDataLoader(self.data_path)
        self.validator = DataValidator()
        self.splitter = DataSplitter()

    def run(
        self,
        save_splits: bool = True,
        save_stats: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the complete data pipeline.

        Args:
            save_splits: Whether to save train/val/test splits
            save_stats: Whether to save dataset statistics

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 80)
        logger.info("Starting Data Pipeline")
        logger.info("=" * 80)

        # Step 1: Load data
        logger.info("\n[1/5] Loading data...")
        df = self.loader.load_as_dataframe(validate=False, handle_errors="coerce")

        # Step 2: Validate data quality
        if self.validate_quality:
            logger.info("\n[2/5] Validating data quality...")
            report = self.validator.validate(df)

            if not report.is_valid():
                logger.error("Data validation failed!")
                logger.error(f"Errors: {report.errors}")
                raise ValueError("Data quality validation failed")

            logger.info("✓ Data quality validation passed")
        else:
            logger.info("\n[2/5] Skipping data quality validation")

        # Step 3: Compute statistics
        logger.info("\n[3/5] Computing dataset statistics...")
        stats = self.loader.compute_statistics(df)

        if save_stats:
            stats_path = self.output_dir / "dataset_stats.json"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                f.write(stats.model_dump_json(indent=2))
            logger.info(f"✓ Saved statistics to {stats_path}")

        # Step 4: Split data
        logger.info("\n[4/5] Splitting data into train/val/test...")
        train_df, val_df, test_df = self.splitter.split(df)

        # Step 5: Save splits
        if save_splits:
            logger.info("\n[5/5] Saving data splits...")
            split_info = self.splitter.save_splits(
                train_df, val_df, test_df, output_dir=self.output_dir
            )
            logger.info(f"✓ Split info: {split_info}")
        else:
            logger.info("\n[5/5] Skipping save (save_splits=False)")

        logger.info("\n" + "=" * 80)
        logger.info("Data Pipeline Complete!")
        logger.info("=" * 80)

        return train_df, val_df, test_df


def main() -> None:
    """Run the data pipeline as a standalone script."""
    pipeline = DataPipeline()
    train_df, val_df, test_df = pipeline.run()

    logger.info(f"\nFinal shapes:")
    logger.info(f"  Train: {train_df.shape}")
    logger.info(f"  Val:   {val_df.shape}")
    logger.info(f"  Test:  {test_df.shape}")


if __name__ == "__main__":
    main()
