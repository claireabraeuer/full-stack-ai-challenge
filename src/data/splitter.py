"""Data splitting utilities for train/validation/test sets."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data.schemas import DataSplit


class DataSplitter:
    """Split data into train, validation, and test sets."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = None,
        stratify_column: Optional[str] = "category",
    ) -> None:
        """Initialize the data splitter.

        Args:
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            random_seed: Random seed for reproducibility
            stratify_column: Column to stratify split by (e.g., 'category')
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed or settings.random_seed
        self.stratify_column = stratify_column

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train, validation, and test sets.

        Args:
            df: Input dataframe to split

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(
            f"Splitting data with ratios - train: {self.train_ratio}, "
            f"val: {self.val_ratio}, test: {self.test_ratio}"
        )

        stratify = df[self.stratify_column] if self.stratify_column else None

        # First split: separate out test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=stratify,
        )

        # Second split: separate train and validation from remaining data
        # Adjust val ratio relative to the train+val set
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)

        stratify_train_val = (
            train_val_df[self.stratify_column] if self.stratify_column else None
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=stratify_train_val,
        )

        logger.info(f"Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Val set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

        # Verify stratification worked
        if self.stratify_column:
            self._log_stratification_stats(df, train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _log_stratification_stats(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Log statistics to verify stratification."""
        logger.info(f"Stratification by '{self.stratify_column}':")

        overall_dist = df[self.stratify_column].value_counts(normalize=True)

        for category in overall_dist.index[:5]:  # Show top 5 categories
            overall_pct = overall_dist[category] * 100
            train_pct = (
                train_df[self.stratify_column].value_counts(normalize=True).get(category, 0)
                * 100
            )
            val_pct = (
                val_df[self.stratify_column].value_counts(normalize=True).get(category, 0) * 100
            )
            test_pct = (
                test_df[self.stratify_column].value_counts(normalize=True).get(category, 0)
                * 100
            )

            logger.debug(
                f"  {category}: Overall={overall_pct:.1f}%, "
                f"Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%"
            )

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> DataSplit:
        """Save train/val/test splits to Parquet files.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Directory to save splits (default: from config)

        Returns:
            DataSplit object with split information
        """
        if output_dir is None:
            output_dir = Path("data/splits")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as Parquet for efficient storage and fast loading
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"
        test_path = output_dir / "test.parquet"

        logger.info(f"Saving splits to {output_dir}")
        train_df.to_parquet(train_path, index=False, compression="snappy")
        val_df.to_parquet(val_path, index=False, compression="snappy")
        test_df.to_parquet(test_path, index=False, compression="snappy")

        logger.info(f"✓ Saved train set to {train_path}")
        logger.info(f"✓ Saved val set to {val_path}")
        logger.info(f"✓ Saved test set to {test_path}")

        split_info = DataSplit(
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            stratify_column=self.stratify_column or "none",
            random_seed=self.random_seed,
        )

        # Save split metadata
        metadata_path = output_dir / "split_info.json"
        with open(metadata_path, "w") as f:
            f.write(split_info.model_dump_json(indent=2))

        logger.info(f"✓ Saved split metadata to {metadata_path}")

        return split_info


def load_splits(
    splits_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split train/val/test data.

    Args:
        splits_dir: Directory containing split files (default: from config)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if splits_dir is None:
        splits_dir = Path("data/splits")

    splits_dir = Path(splits_dir)

    logger.info(f"Loading splits from {splits_dir}")

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    logger.info(f"Loaded train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")

    return train_df, val_df, test_df
