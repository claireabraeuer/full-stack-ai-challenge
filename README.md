# Intelligent Support System

An end-to-end ML system for ticket categorization and solution retrieval using XGBoost, PyTorch, and Hybrid RAG.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast package installer)
- Docker (optional)

### Setup

```bash
# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Configure environment (optional)
cp .env.example .env
```

### Prepare Data

```bash
# Split data into train/val/test (70/15/15)
uv run python scripts/prepare_data.py prepare --data support_tickets.json --output data/splits
```

### Train Models

```bash
# Train XGBoost (production model: 100% acc, 0.10s training, ~5KB)
uv run python -m src.models.categorization.train_traditional

# Train PyTorch (benchmark: 100% acc, 17.20s training, ~450KB)
uv run python -m src.models.categorization.train_dl

# Build RAG index (77K docs, semantic + graph)
uv run python scripts/build_hybrid_rag.py
```

### Run API

```bash
# Start server
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up -d
```

Access API docs at http://localhost:8000/docs

## System Overview

**Components:**
- **XGBoost Classifier**: Ticket categorization (5 categories)
- **Hybrid RAG**: Semantic search (ChromaDB) + Knowledge graph (NetworkX)
- **Drift Detection**: Confidence, accuracy, and distribution monitoring
- **Anomaly Detection**: Volume spikes, sentiment shifts, emerging issues

**Performance:**
- API latency: ~250ms
- Throughput: ~4K req/s
- Model accuracy: 100% (synthetic data)

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## API Usage

### Process Ticket

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Database sync timeout",
    "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets",
    "product": "DataSync Pro"
  }'
```

**Response:**
```json
{
  "prediction": {
    "category": "Technical Issue",
    "confidence": 0.98
  },
  "similar_tickets": [
    {
      "ticket_id": "TK-2024-004422",
      "resolution": "Increased batch size in config...",
      "satisfaction_score": 4
    }
  ]
}
```

### Other Endpoints

- `GET /health` - System status
- `POST /feedback` - Submit corrections
- `GET /metrics` - Prometheus metrics

See [API.md](API.md) for complete documentation.

## Project Structure

```
├── src/
│   ├── data/              # Data loading, validation, splitting
│   ├── features/          # Feature preprocessing (514 features)
│   ├── models/
│   │   ├── categorization/  # XGBoost + PyTorch
│   │   ├── retrieval/       # Hybrid RAG
│   │   ├── monitoring/      # Drift detection
│   │   └── anomaly/         # Anomaly detection
│   └── api/               # FastAPI (4 endpoints)
├── scripts/               # Training, analysis, demos
├── data/
│   ├── splits/            # Train/val/test (Parquet)
│   └── chroma_db/         # Vector index (77K docs)
├── models/
│   └── categorization/    # Trained models (.pkl, .pth)
└── mlruns/                # MLflow experiments
```

## Reproducibility

### Run Experiments

```bash
# Model comparison
uv run python scripts/compare_models.py

# Hyperparameter tuning
uv run python scripts/tune_n_estimators.py

# Feature importance
uv run python scripts/analyze_feature_importance.py

# Data quality analysis
uv run python scripts/analyze_data_quality.py
```

### View Results

```bash
# MLflow UI (experiments, models, metrics)
mlflow ui --host 0.0.0.0 --port 5000
```

### Run Analysis

```bash
# Anomaly detection
uv run python scripts/run_anomaly_detection.py

# Drift detection
uv run python scripts/run_drift_detection.py
```

## Docker Deployment

```bash
# Build and start all services (API, MLflow, Prometheus)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

See [DOCKER.md](DOCKER.md) for details.

## Key Design Decisions

1. **XGBoost over PyTorch**: 172x faster training, 90x smaller model, equal accuracy
2. **Hybrid RAG**: Combines semantic search + knowledge graph for 3x faster retrieval
3. **Statistical Drift Detection**: Simple z-scores vs ML-based (no training needed)
4. **Single-File API**: 4 endpoints in one file for simplicity
5. **UV + Docker**: Fast local dev + production isolation

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive design documentation.

## Documentation

- **[README.md](README.md)** - Setup and quick start (this file)
- **[MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md)** - Model benchmarks, comparisons, experiments
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design decisions
- **[API.md](API.md)** - API endpoints and usage examples
- **[DOCKER.md](DOCKER.md)** - Docker deployment guide
- **[COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)** - Project completion checklist

## License

MIT
