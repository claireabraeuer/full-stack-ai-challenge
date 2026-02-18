"""Experiment with different n_estimators for XGBoost."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import mlflow
import xgboost as xgb
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_splits
from src.features import encode_labels, preprocess_data

# Test very low numbers of estimators to find minimum viable model
N_ESTIMATORS_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]


def main():
    """Run n_estimators tuning experiment."""
    logger.info("=" * 80)
    logger.info("XGBoost n_estimators Tuning Experiment")
    logger.info("=" * 80)

    # Set MLflow experiment
    mlflow.set_experiment("xgboost-n-estimators-tuning")

    # Load and preprocess data (only once)
    logger.info("\nLoading and preprocessing data...")
    train_df, val_df, test_df = load_splits("data/splits")
    X_train, X_val, X_test, preprocessor = preprocess_data(train_df, val_df, test_df)

    y_train_enc, y_val_enc, y_test_enc, label_encoder = encode_labels(
        train_df["category"], val_df["category"], test_df["category"]
    )

    logger.info(f"Data shape: {X_train.shape}")
    logger.info(f"Training on {X_train.shape[0]:,} samples")
    logger.info(f"Validating on {X_val.shape[0]:,} samples\n")

    # Store results for plotting
    results = []

    # Train model with each n_estimators value
    for n_est in N_ESTIMATORS_VALUES:
        logger.info(f"Training with n_estimators={n_est}...")

        with mlflow.start_run(run_name=f"xgb_n{n_est}"):
            # Train model
            start_time = time.time()
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
            model.fit(X_train, y_train_enc, verbose=False)
            train_time = time.time() - start_time

            # Evaluate
            start_time = time.time()
            y_val_pred = model.predict(X_val)
            inference_time = time.time() - start_time

            val_acc = accuracy_score(y_val_enc, y_val_pred)
            val_f1 = f1_score(y_val_enc, y_val_pred, average="macro")

            # Test set
            y_test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test_enc, y_test_pred)
            test_f1 = f1_score(y_test_enc, y_test_pred, average="macro")

            # Log to MLflow
            mlflow.log_params(
                {
                    "n_estimators": n_est,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                }
            )

            mlflow.log_metrics(
                {
                    "train_time": train_time,
                    "val_accuracy": val_acc,
                    "val_f1_macro": val_f1,
                    "test_accuracy": test_acc,
                    "test_f1_macro": test_f1,
                    "inference_time": inference_time,
                    "throughput": X_val.shape[0] / inference_time,
                }
            )

            # Store for summary
            results.append(
                {
                    "n_estimators": n_est,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "train_time": train_time,
                }
            )

            logger.info(
                f"  n={n_est:3d}: Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}, "
                f"Train Time={train_time:.2f}s"
            )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(
        f"{'n_estimators':<15} {'Val Acc':<12} {'Test Acc':<12} {'Train Time':<12}"
    )
    logger.info("-" * 80)

    for r in results:
        logger.info(
            f"{r['n_estimators']:<15} {r['val_acc']:<12.4f} {r['test_acc']:<12.4f} "
            f"{r['train_time']:<12.2f}s"
        )

    # Find best
    best = max(results, key=lambda x: x["val_acc"])
    logger.info("\n" + "=" * 80)
    logger.info(
        f"Best: n_estimators={best['n_estimators']} with Val Acc={best['val_acc']:.4f}"
    )
    logger.info("=" * 80)
    logger.info("\nView results in MLflow UI:")
    logger.info("  mlflow ui --host 127.0.0.1 --port 5000")
    logger.info("  Then go to: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
