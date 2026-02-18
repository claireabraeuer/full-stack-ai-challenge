"""Model loader with MLflow Model Registry integration.

Loads production models from MLflow Registry first, falls back to local files.
"""

import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.tracking
import mlflow.xgboost
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Load categorization models from MLflow Registry or local files."""

    def __init__(self, model_type: Optional[str] = None):
        """Initialize loader.

        Args:
            model_type: Model type ("xgboost" or "pytorch").
                       Defaults to settings.categorization_model.
        """
        self.model_type = model_type or settings.categorization_model
        self.models_dir = Path("models/categorization")

        # Load preprocessor and label encoder (shared by all models)
        self.preprocessor = self._load_preprocessor()
        self.label_encoder = self._load_label_encoder()

        logger.info(f"Initialized ModelLoader for {self.model_type}")

    def load_production_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load the current Production model from MLflow Registry.

        Returns:
            Tuple of (model, metadata dict with version info)
        """
        model_name = f"ticket-categorization-{self.model_type}"

        try:
            # Try loading from MLflow Model Registry (Production stage)
            client = mlflow.tracking.MlflowClient()

            # Get latest Production version
            versions = client.get_latest_versions(model_name, stages=["Production"])

            if not versions:
                logger.warning(
                    f"No Production version found for {model_name}, "
                    "falling back to local file"
                )
                return self._load_local_model()

            version = versions[0]
            model_uri = f"models:/{model_name}/Production"

            logger.info(
                f"Loading {model_name} version {version.version} from MLflow Registry"
            )

            # Load with appropriate flavor for each model type
            if self.model_type == "xgboost":
                model = mlflow.xgboost.load_model(model_uri)
            elif self.model_type == "pytorch":
                model = mlflow.pytorch.load_model(model_uri)
            else:
                # Fallback to PyFunc for unknown types
                model = mlflow.pyfunc.load_model(model_uri)

            # Get run metrics for baseline
            run = client.get_run(version.run_id)

            metadata = {
                "model_type": self.model_type,
                "version": version.version,
                "stage": "Production",
                "run_id": version.run_id,
                "registered_at": version.creation_timestamp,
                "baseline_metrics": {
                    "test_accuracy": run.data.metrics.get("test_accuracy"),
                    "test_f1_macro": run.data.metrics.get("test_f1_macro"),
                    "val_accuracy": run.data.metrics.get("val_accuracy"),
                    "val_f1_macro": run.data.metrics.get("val_f1_macro"),
                },
            }

            logger.info(f"✓ Loaded model version {version.version} from Registry")
            return model, metadata

        except Exception as e:
            logger.warning(
                f"Failed to load from MLflow Registry: {e}. "
                "Falling back to local file."
            )
            return self._load_local_model()

    def _load_local_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load model from local file (fallback).

        Returns:
            Tuple of (model, metadata dict)
        """
        logger.info(f"Loading {self.model_type} model from local file")

        if self.model_type == "xgboost":
            model_path = self.models_dir / "xgboost_category.pkl"
            model = joblib.load(model_path)

        elif self.model_type == "pytorch":
            model_path = self.models_dir / "pytorch_category.pth"
            checkpoint = torch.load(model_path, map_location="cpu")

            # Reconstruct model from saved state dict
            from src.models.categorization.train_dl import TicketClassifier

            model = TicketClassifier(
                input_dim=checkpoint["input_dim"],
                hidden_dims=checkpoint["hidden_dims"],
                num_classes=checkpoint["num_classes"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        metadata = {
            "model_type": self.model_type,
            "version": "local",
            "stage": "local_file",
            "source": str(model_path),
            "baseline_metrics": None,
        }

        logger.info(f"✓ Loaded model from {model_path}")
        return model, metadata

    def _load_preprocessor(self):
        """Load feature preprocessor."""
        preprocessor_path = self.models_dir / "preprocessor.pkl"
        return joblib.load(preprocessor_path)

    def _load_label_encoder(self) -> LabelEncoder:
        """Load label encoder."""
        encoder_path = self.models_dir / "label_encoder.pkl"
        return joblib.load(encoder_path)

    def predict(
        self, model: Any, ticket_df, return_confidence: bool = True
    ) -> Dict[str, Any]:
        """Run prediction on ticket data.

        Args:
            model: Loaded model (MLflow PyFunc or raw model)
            ticket_df: DataFrame with ticket data
            return_confidence: Whether to return confidence scores

        Returns:
            Dict with prediction, confidence, and label
        """
        # Create text_combined column (required by preprocessor)
        ticket_df = ticket_df.copy()
        ticket_df["text_combined"] = (
            ticket_df["subject"].fillna("") + " " + ticket_df["description"].fillna("")
        )

        # Preprocess features
        X = self.preprocessor.transform(ticket_df)

        if self.model_type == "xgboost":
            # Get prediction
            y_pred_encoded = model.predict(X)[0]

            # Get confidence (max probability)
            if return_confidence:
                y_proba = model.predict_proba(X)
                confidence = float(y_proba.max(axis=1)[0])
            else:
                confidence = None

        elif self.model_type == "pytorch":
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X.toarray())
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1)

                y_pred_encoded = torch.argmax(probs, dim=1).numpy()[0]

                if return_confidence:
                    confidence = float(probs.max(dim=1).values[0])
                else:
                    confidence = None

        # Decode label
        predicted_label = self.label_encoder.inverse_transform([y_pred_encoded])[0]

        return {
            "predicted_category": predicted_label,
            "confidence": confidence,
            "encoded_label": int(y_pred_encoded),
        }


# Convenience function
def load_production_model(model_type: Optional[str] = None):
    """Load the production model with one function call.

    Args:
        model_type: "xgboost" or "pytorch". Defaults to config setting.

    Returns:
        Tuple of (model, metadata, loader)
    """
    loader = ModelLoader(model_type)
    model, metadata = loader.load_production_model()
    return model, metadata, loader
