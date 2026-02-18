"""Configuration management for the Intelligent Support System."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    env: Literal["development", "production", "test"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # MLflow Tracking
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "support-system"

    # Vector Database
    chroma_db_path: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Graph Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Feature Store
    feast_repo_path: str = "./feature_repo"

    # Model Configuration
    categorization_model: Literal["xgboost", "catboost", "tensorflow"] = "xgboost"
    confidence_threshold: float = 0.85
    rag_top_k: int = 5

    # Data Paths
    ticket_data_path: Path = Path("./support_tickets.json")
    train_data_path: Path = Path("./data/splits/train.parquet")
    val_data_path: Path = Path("./data/splits/val.parquet")
    test_data_path: Path = Path("./data/splits/test.parquet")

    # Random seed for reproducibility
    random_seed: int = 42


# Global settings instance
settings = Settings()
