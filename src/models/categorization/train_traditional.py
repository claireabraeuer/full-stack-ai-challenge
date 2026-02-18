"""Train traditional ML model (XGBoost) for ticket categorization."""

import time
from pathlib import Path

import joblib
import mlflow
import mlflow.tracking
import mlflow.xgboost
import xgboost as xgb
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import settings
from src.data import load_splits
from src.features import encode_labels, preprocess_data


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels (encoded)

    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=1,  # Single tree achieves 100% accuracy on this synthetic data
        max_depth=6,
        learning_rate=0.1,
        random_state=settings.random_seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    training_time = time.time() - start_time

    logger.info(f"XGBoost training completed in {training_time:.2f}s")

    return model, training_time


def evaluate_model(model, X, y, label_encoder, split_name="val"):
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        X: Features
        y: True labels (encoded)
        label_encoder: Label encoder for decoding
        split_name: Name of the split (for logging)

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating on {split_name} set...")

    start_time = time.time()
    y_pred = model.predict(X)
    inference_time = time.time() - start_time

    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    # Decode for classification report
    y_true_decoded = label_encoder.inverse_transform(y)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    n_samples = X.shape[0]

    logger.info(f"{split_name.capitalize()} Accuracy: {accuracy:.4f}")
    logger.info(f"{split_name.capitalize()} F1 (macro): {f1_macro:.4f}")
    logger.info(f"{split_name.capitalize()} F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"Inference time: {inference_time:.2f}s ({n_samples / inference_time:.0f} samples/s)")

    # Print classification report
    report = classification_report(y_true_decoded, y_pred_decoded, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "inference_time": inference_time,
        "throughput": n_samples / inference_time,
    }


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("Traditional ML Training Pipeline")
    logger.info("=" * 80)

    # Set MLflow experiment
    mlflow.set_experiment("ticket-categorization")

    # Load and preprocess data
    logger.info("\n[1/4] Loading data splits...")
    train_df, val_df, test_df = load_splits("data/splits")

    logger.info("[2/4] Preprocessing features...")
    X_train, X_val, X_test, preprocessor = preprocess_data(train_df, val_df, test_df)

    logger.info("[3/4] Encoding labels...")
    y_train_enc, y_val_enc, y_test_enc, label_encoder = encode_labels(
        train_df["category"], val_df["category"], test_df["category"]
    )

    logger.info(f"Feature matrix shape: {X_train.shape}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    logger.info(f"Classes: {list(label_encoder.classes_)}")

    # Create output directory
    models_dir = Path("models/categorization")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train XGBoost
    logger.info("\n[4/4] Training XGBoost model...")
    with mlflow.start_run(run_name="xgboost-category"):
        xgb_model, xgb_train_time = train_xgboost(X_train, y_train_enc, X_val, y_val_enc)

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "xgboost",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_features": X_train.shape[1],
            }
        )

        # Evaluate
        val_metrics = evaluate_model(xgb_model, X_val, y_val_enc, label_encoder, "validation")
        test_metrics = evaluate_model(xgb_model, X_test, y_test_enc, label_encoder, "test")

        # Log metrics
        mlflow.log_metrics(
            {
                "train_time": xgb_train_time,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_throughput": val_metrics["throughput"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )

        # Log model
        mlflow.xgboost.log_model(xgb_model, "model")

        # Register model in MLflow Model Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        logger.info("Registering model in MLflow Model Registry...")
        result = mlflow.register_model(model_uri, "ticket-categorization-xgboost")

        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="ticket-categorization-xgboost",
            version=result.version,
            stage="Production",
            archive_existing_versions=True  # Auto-archive old Production versions
        )

        logger.info(f"✓ Model registered as version {result.version} in Production stage")

        # Save locally
        xgb_path = models_dir / "xgboost_category.pkl"
        joblib.dump(xgb_model, xgb_path)
        logger.info(f"✓ Saved XGBoost model to {xgb_path}")

    # Save preprocessor and label encoder
    logger.info("\nSaving artifacts...")
    preprocessor_path = models_dir / "preprocessor.pkl"
    label_encoder_path = models_dir / "label_encoder.pkl"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, label_encoder_path)

    logger.info(f"✓ Saved preprocessor to {preprocessor_path}")
    logger.info(f"✓ Saved label encoder to {label_encoder_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"\nXGBoost training time: {xgb_train_time:.2f}s")
    logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"\nModels saved to: {models_dir}")


if __name__ == "__main__":
    main()
