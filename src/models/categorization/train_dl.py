"""Train deep learning model (PyTorch) for ticket categorization."""

import time
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import mlflow.tracking
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import settings
from src.data import load_splits
from src.features import encode_labels, preprocess_data


class TicketClassifier(nn.Module):
    """Simple feedforward neural network for ticket categorization.

    Architecture:
        Input (514) -> Dense(256) -> ReLU -> Dropout(0.3)
                    -> Dense(128) -> ReLU -> Dropout(0.3)
                    -> Dense(64) -> ReLU -> Dropout(0.2)
                    -> Dense(5) -> Softmax
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        """Initialize the neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Reduce dropout in later layers
            drop_prob = dropout if i < len(hidden_dims) - 1 else dropout * 0.7
            layers.append(nn.Dropout(drop_prob))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


def train_pytorch_model(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    num_classes,
    hidden_dims=[256, 128, 64],
    epochs=20,
    batch_size=128,
    learning_rate=0.001,
    device="cpu",
):
    """Train PyTorch classifier.

    Args:
        X_train: Training features (sparse or dense array)
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels (encoded)
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dims: List of hidden layer sizes
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Trained model and training history
    """
    logger.info("Training PyTorch model...")
    logger.info(f"Device: {device}")
    logger.info(f"Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")

    # Convert sparse matrices to dense if needed
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TicketClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=0.3,
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(device)
            y_val_device = y_val_tensor.to(device)

            val_outputs = model(X_val_device)
            val_loss = criterion(val_outputs, y_val_device).item()
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        # Store history
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {history['train_loss'][-1]:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    training_time = time.time() - start_time
    logger.info(f"PyTorch training completed in {training_time:.2f}s")

    return model, history, training_time


def evaluate_model(model, X, y, label_encoder, device="cpu", split_name="val"):
    """Evaluate model on a dataset.

    Args:
        model: Trained PyTorch model
        X: Features (sparse or dense array)
        y: True labels (encoded)
        label_encoder: Label encoder for decoding
        device: Device for inference
        split_name: Name of the split (for logging)

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating on {split_name} set...")

    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    X_tensor = torch.FloatTensor(X).to(device)

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
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
    logger.info("Deep Learning Training Pipeline (PyTorch)")
    logger.info("=" * 80)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

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

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    logger.info(f"Feature matrix shape: {X_train.shape}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {list(label_encoder.classes_)}")

    # Create output directory
    models_dir = Path("models/categorization")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train PyTorch model
    logger.info("\n[4/4] Training PyTorch model...")

    # Hyperparameters
    hidden_dims = [256, 128, 64]
    epochs = 20
    batch_size = 128
    learning_rate = 0.001

    with mlflow.start_run(run_name="pytorch-category"):
        model, history, train_time = train_pytorch_model(
            X_train,
            y_train_enc,
            X_val,
            y_val_enc,
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "pytorch_nn",
                "hidden_dims": str(hidden_dims),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "n_features": input_dim,
                "device": device,
            }
        )

        # Evaluate
        val_metrics = evaluate_model(model, X_val, y_val_enc, label_encoder, device, "validation")
        test_metrics = evaluate_model(model, X_test, y_test_enc, label_encoder, device, "test")

        # Log metrics
        mlflow.log_metrics(
            {
                "train_time": train_time,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_throughput": val_metrics["throughput"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
            }
        )

        # Log training history
        for epoch in range(len(history["train_loss"])):
            mlflow.log_metric("epoch_train_loss", history["train_loss"][epoch], step=epoch)
            mlflow.log_metric("epoch_val_loss", history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("epoch_val_acc", history["val_acc"][epoch], step=epoch)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Register model in MLflow Model Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        logger.info("Registering model in MLflow Model Registry...")
        result = mlflow.register_model(model_uri, "ticket-categorization-pytorch")

        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="ticket-categorization-pytorch",
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info(f"✓ Model registered as version {result.version} in Production stage")

        # Save locally
        model_path = models_dir / "pytorch_category.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "num_classes": num_classes,
                "hidden_dims": hidden_dims,
            },
            model_path,
        )
        logger.info(f"✓ Saved PyTorch model to {model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"\nPyTorch training time: {train_time:.2f}s ({epochs} epochs)")
    logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"\nModels saved to: {models_dir}")


if __name__ == "__main__":
    main()
