"""Simple feature preprocessing pipeline using sklearn built-ins."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import pandas as pd


def build_preprocessing_pipeline():
    """Build a simple preprocessing pipeline for XGBoost.

    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    # Define column groups
    text_cols = ["subject", "description"]
    categorical_cols = ["product", "priority", "customer_tier", "channel"]
    numerical_cols = [
        "previous_tickets",
        "satisfaction_score",
        "resolution_time_hours",
        "agent_experience_months",
    ]
    boolean_cols = ["escalated", "known_issue", "contains_error_code"]

    # Text preprocessing: Simple TF-IDF on concatenated text
    text_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    min_df=2,
                ),
            )
        ]
    )

    # Categorical: Just one-hot encode
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    # Numerical: Just scale
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Boolean: Pass through as-is (0/1)
    # Note: Using passthrough for booleans

    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_pipeline, "text_combined"),  # Single combined text column
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numerical_pipeline, numerical_cols),
            ("bool", "passthrough", boolean_cols),
        ],
        remainder="drop",  # Drop all other columns
    )

    return preprocessor


def preprocess_data(train_df, val_df, test_df):
    """Apply preprocessing to train/val/test splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple of (X_train, X_val, X_test, preprocessor)
    """
    # Concatenate text columns into a single string for TF-IDF
    for df in [train_df, val_df, test_df]:
        df["text_combined"] = (
            df["subject"].fillna("") + " " + df["description"].fillna("")
        )

    # Build and fit pipeline on training data
    preprocessor = build_preprocessing_pipeline()

    # Fit on train, transform all
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    return X_train, X_val, X_test, preprocessor


def encode_labels(y_train, y_val, y_test):
    """Encode string labels to integers for tree-based models.

    Args:
        y_train: Training labels (Series or array of strings)
        y_val: Validation labels
        y_test: Test labels

    Returns:
        Tuple of (y_train_encoded, y_val_encoded, y_test_encoded, label_encoder)
    """
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder
