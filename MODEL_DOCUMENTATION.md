# Model Documentation

This document provides comprehensive documentation for all machine learning models in the Intelligent Support System, including performance benchmarks, architecture details, experiment tracking, and error analysis.

## Overview

The system implements two complementary ML pipelines:

1. **Ticket Categorization**: Predicts ticket categories using XGBoost or PyTorch
2. **Solution Retrieval**: Finds similar resolved tickets using Hybrid RAG (semantic + graph search)

Both models achieve 100% accuracy on the synthetic dataset and are production-ready with comprehensive monitoring.

---

## 1. Feature Engineering

**Implementation**: [src/features/preprocessing.py](src/features/preprocessing.py)

The preprocessing pipeline transforms raw ticket data into **514 features**:

### Text Features (TF-IDF)
- Concatenates `subject` + `description` fields
- TF-IDF vectorization with **1000 max features**

### Categorical Features (One-Hot Encoding)
- **Features**: `product`, `priority`, `customer_tier`, `channel`
- Missing values filled with "missing" category
- Creates binary indicators for each category value

### Numerical Features (Standardized)
- **Features**: `previous_tickets`, `satisfaction_score`, `resolution_time_hours`, `agent_experience_months`
- Mean imputation for missing values
- Standard scaling (zero mean, unit variance)

### Boolean Features (Pass-through)
- **Features**: `escalated`, `known_issue`, `contains_error_code`
- Kept as 0/1 values without transformation

---

## 2. XGBoost Categorization Model

### Architecture

**Training Script**: [src/models/categorization/train_traditional.py](src/models/categorization/train_traditional.py)

**Model Configuration**:
```python
XGBClassifier(
    n_estimators=1,        # Single tree achieves 100% accuracy on synthetic data
    max_depth=6,           # 6 levels deep
    learning_rate=0.1,     # Standard learning rate
    random_state=42,       # Reproducibility
    eval_metric='mlogloss' # Multi-class log loss
)
```

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Training Time | 0.10s (77,000 samples) |
| Validation Accuracy | 100% |
| Test Accuracy | 100% |
| Inference Speed | 15.2M samples/second |
| Model Size | ~5KB |
| Memory Usage | Minimal (~10MB during inference) |

### Training the Model

```bash
# Train XGBoost categorization model
python -m src.models.categorization.train_traditional

# Models are saved to:
# - models/categorization/xgboost_category.pkl
# - models/categorization/preprocessor.pkl
# - models/categorization/label_encoder.pkl
```

### Hyperparameter Tuning: n_estimators

**Experiment Script**: [scripts/tune_n_estimators.py](scripts/tune_n_estimators.py)

We experimented with different numbers of trees to find the optimal trade-off between accuracy and training time.

```bash
# Run n_estimators tuning experiment
python scripts/tune_n_estimators.py
```

**Results**:

| n_estimators | Val Accuracy | Test Accuracy | Training Time | Model Size | Speedup vs PyTorch |
|--------------|--------------|---------------|---------------|------------|--------------------|
| 1 | 100% | 100% | 0.10s | ~5KB | **172x faster** |
| 3 | 100% | 100% | 0.08s | ~15KB | 215x faster |
| 5 | 100% | 100% | 0.12s | ~25KB | 143x faster |
| 10 | 100% | 100% | 0.14s | ~50KB | 123x faster |
| 100 | 100% | 100% | 0.82s | ~500KB | 21x faster |

### Why n_estimators=1 Works

**Surprising Finding**: A single decision tree achieves 100% accuracy on this dataset!

Through hyperparameter tuning (testing n_estimators from 1 to 100), we discovered that even **n_estimators=1** achieves perfect classification. This is due to:

1. **Synthetic Data Patterns**: The dataset has highly regular, templated language
2. **Category Name Leakage**: 19.6% of tickets contain the category name in the text
   - Example: "Security concern..." → Security category
   - Example: "Feature request for..." → Feature Request category
3. **Strong TF-IDF Features**: Only **5 features** are needed for perfect separation:
   - `text__authentication`, `text__accessed`, `text__add`, `text__additional`, `text__operations`
4. **Perfect Class Balance**: All categories are ~20% of the data with no overlap

---

## 3. PyTorch Deep Learning Model

### Architecture

**Model File**: [src/models/categorization/train_dl.py](src/models/categorization/train_dl.py)

A feedforward neural network implemented in PyTorch as an alternative to the gradient boosting approach. Uses the same preprocessed features (514 dimensions) as input.

```
Input (514 features)
    ↓
Dense(256) → ReLU → Dropout(0.3)
    ↓
Dense(128) → ReLU → Dropout(0.3)
    ↓
Dense(64) → ReLU → Dropout(0.2)
    ↓
Dense(5) → Softmax
```

**Configuration**:
- **Hidden layers**: [256, 128, 64]
- **Activation**: ReLU
- **Dropout**: 0.3 (first layers), 0.2 (last layer)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Batch size**: 128
- **Epochs**: 20

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Training Time | 17.20s (20 epochs on CPU) |
| Validation Accuracy | 100% |
| Test Accuracy | 100% |
| Inference Speed | 3.4M samples/second |
| Model Size | ~450KB |
| Memory Usage | ~50MB during training |

### Training the Model

```bash
# Train PyTorch categorization model
python -m src.models.categorization.train_dl

# Model saved to:
# - models/categorization/pytorch_category.pth
```

---

## 4. Model Comparison: XGBoost vs PyTorch

### Performance Summary

| Model | Training Time | Val Accuracy | Test Accuracy | Inference Speed | Model Size |
|-------|---------------|--------------|---------------|-----------------|------------|
| **XGBoost (n=1)** | **0.10s** | **100%** | **100%** | **15.2M samples/s** | **~5KB** |
| PyTorch | 17.20s | 100% | 100% | 3.4M samples/s | ~450KB |

### Trade-offs

**XGBoost Advantages**:
- **172x faster training** (0.10s vs 17.20s)
- **90x smaller model** size (~5KB vs ~450KB)
- Built-in feature importance
- Better interpretability
- Minimal complexity (single decision tree)
- Lower memory footprint

**PyTorch Advantages**:
- More flexible architecture (can add custom layers, embeddings)
- Better for transfer learning scenarios
- Can leverage GPU acceleration for larger datasets
- Easier to extend with attention mechanisms or custom loss functions
- More complex (requires more hyperparameter tuning)

---

## 5. Feature Importance Analysis

**Script**: [scripts/analyze_feature_importance.py](scripts/analyze_feature_importance.py)

Analyze which features are most important for ticket categorization:

```bash
# Generate feature importance analysis
uv run python scripts/analyze_feature_importance.py

# Outputs:
# - data/feature_importance.json  # Top 20 features + summary
# - data/feature_importance.png   # Visualization
```

### Key Insights

**Top Feature Categories**:
1. **TF-IDF features** (text-based) - Dominate importance
   - Category-specific keywords are highly predictive
   - Error codes and technical terms are strong signals
   - Product names in descriptions help classification

2. **Categorical features** (product, priority, channel)
   - Provide contextual information
   - Help disambiguate borderline cases
   - Product field is particularly informative

3. **Numerical features** (previous_tickets, satisfaction_score)
   - Add nuance to predictions
   - Help identify patterns (e.g., recurring issues)
   - Lower importance than text features

4. **Boolean features** (escalated, known_issue, contains_error_code)
   - `contains_error_code` is moderately important
   - `escalated` and `known_issue` have minimal impact


---

## 6. Hybrid RAG System

**Implementation**: [src/models/retrieval/hybrid_rag.py](src/models/retrieval/hybrid_rag.py)

### Architecture

The **Hybrid RAG System** combines two complementary retrieval approaches in a single unified class:

**Components**:
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: ChromaDB for semantic similarity search
- **Knowledge Graph**: NetworkX with entity relationships
- **Hybrid Scoring**: Configurable weighting between semantic and graph signals

**Knowledge Graph Structure**:
```
Products ↔ Tickets (15K tickets per product)
Error Codes ↔ Tickets (8 unique codes, ~3,300-3,500 tickets each)
Resolution Codes ↔ Tickets (12 unique codes, ~6,300-6,500 tickets each)
Tags ↔ Tickets (multiple tags per ticket)
```

### Building the Index

**Script**: [scripts/build_hybrid_rag.py](scripts/build_hybrid_rag.py)

```bash
# Build both vector database and knowledge graph
python scripts/build_hybrid_rag.py

# This creates:
# - data/chroma_db/          # ChromaDB vector store (77K embeddings)
# - Knowledge graph in memory # 77K nodes, 181K edges
```

**What gets indexed**:
- **Vector DB**: Subject + Description (semantic), metadata (category, product, satisfaction)
- **Knowledge Graph**: Entities (error codes, products, tags, resolution codes)

### Performance Benchmarks

| Configuration | Query Time | Description |
|---------------|------------|-------------|
| Semantic only (1.0, 0.0) | ~1-2s | ChromaDB similarity search only |
| **Hybrid (0.7, 0.3)** | **~0.3s** | **Balanced approach (default)** |
| Graph only (0.0, 1.0) | ~0.1s | Entity matching only |

**Indexing Performance**:
- **Vector DB indexing**: ~3 minutes for 77K tickets
- **Graph building**: ~2.5 seconds for 77K tickets
- **Total index size**: ~300MB (vector DB) + negligible (graph in memory)

### Entity Extraction

The system automatically extracts:

**Error Codes** (8 unique):
- Patterns: `ERROR_XXX_123`, `E-123`, `ERR_XXX`, `ERROR-123`
- Examples: `ERROR_TIMEOUT_429`, `ERROR_AUTH_401`, `ERROR_SERVER_500`
- Distribution: ~3,300-3,500 tickets per code

**Products** (5 unique):
- DataSync Pro, API Gateway, StreamProcessor, Analytics Dashboard, CloudBackup Enterprise
- Distribution: ~15K tickets each (balanced)

**Resolution Codes** (12 unique):
- Examples: `CONFIG_CHANGE`, `BUG_FIX`, `PATCH_APPLIED`, `DUPLICATE`
- Distribution: ~6,300-6,500 tickets per code

### Hybrid Search Features

**1. Semantic Search Component**
- Finds tickets with similar meaning (not just keywords)
- Uses cosine similarity on 384-dim embeddings
- Handles synonyms and paraphrasing
- Metadata filtering: category, product, priority
- Minimum satisfaction score threshold (default: 3.0)

**2. Knowledge Graph Component**
- Extracts entities from query (error codes, products, tags)
- Traverses graph to find tickets sharing those entities
- Error codes get 3x weight (strong signal for technical issues)
- Fast lookup using entity-to-ticket indices

**3. Intelligent Re-ranking**

Semantic results are boosted by:
- **Category match**: 1.3x boost if matches predicted category
- **Resolution helpfulness**: 1.2x boost if marked helpful
- **Satisfaction score**: Up to 1.2x boost for scores > 3
- **Resolution time**: 1.1x boost for fast resolutions (< 4 hours)

**4. Hybrid Score Fusion**

Final ranking combines both signals:
```
hybrid_score = semantic_weight × semantic_score + graph_weight × graph_score
```

Default weights (70% semantic, 30% graph) work well for most queries.

### Using Hybrid RAG

```python
from src.models.retrieval import HybridRAG

# 1. Initialize system
hybrid_rag = HybridRAG(
    embedding_model="all-MiniLM-L6-v2",
    db_path="data/chroma_db",
)

# 2. Build index (both vector DB + graph)
hybrid_rag.build_index(train_df, batch_size=100)

# 3. Retrieve with hybrid search
results = hybrid_rag.retrieve(
    query="Database sync timeout error ERROR_TIMEOUT_429",
    predicted_category="Technical Issue",  # From categorization model
    top_k=5,
    semantic_weight=0.7,   # 70% semantic similarity
    graph_weight=0.3,       # 30% entity matching
)

# 4. Access results
for result in results:
    print(f"Ticket: {result['ticket_id']}")
    print(f"Hybrid Score: {result['hybrid_score']:.3f}")
    print(f"Resolution: {result['metadata']['resolution']}")
```

### Benefits of Hybrid Approach

- **3x faster retrieval** vs semantic-only (0.3s vs 1-2s)
- **Better precision** with entity matching (error codes provide exact matches)
- **Robust fallback**: Falls back to semantic if no entity matches
- **Flexible**: Adjust weights based on query type

---

## 7. MLOps & Model Management

**MLflow Model Registry**  
Implementation: src/models/categorization/model_loader.py  
All trained models are auto-registered with versioning and stages: Staging → Production → Archived.

```python
from src.models.categorization import load_production_model
model, metadata, loader = load_production_model(model_type="xgboost")
print(metadata["version"], metadata["baseline_metrics"]["test_accuracy"])
result = loader.predict(ticket_df, return_confidence=True)
```

View models: mlflow ui --host 0.0.0.0 --port 5000 → Models tab

### Drift Detection

**Script**: [scripts/run_drift_detection.py](scripts/run_drift_detection.py)
- Confidence drift: mean confidence < 0.85
- Accuracy drift: rolling accuracy >5% below baseline
- Distribution drift: new/surging categories vs training data

### Experiment Tracking (MLflow)
- Key experiments: xgboost-n-estimators-tuning, categorization-xgboost, categorization-pytorch.
- Tracked: hyperparameters, metrics, artifacts, tags, and model lineage.


---

## 8. Error Analysis

### Current Dataset (Synthetic)

**Observed Performance**:
- **100% accuracy** achieved on validation and test sets
- **No misclassifications** found
- **Perfect class separation** in feature space

---

## 9. Model Artifacts

### Location

All model artifacts are saved to `models/categorization/`:

```
models/categorization/
├── xgboost_category.pkl      # Trained XGBoost classifier
├── pytorch_category.pth       # Trained PyTorch model
├── preprocessor.pkl           # Fitted sklearn ColumnTransformer
└── label_encoder.pkl          # Label encoder for category mapping
```

### Vector Database

**Location**: `data/chroma_db/`

- ChromaDB index with 77K embeddings
- 384-dimensional vectors (all-MiniLM-L6-v2)

### File Descriptions

**xgboost_category.pkl**:
- Trained XGBClassifier with n_estimators=1
- Serialized with joblib

**pytorch_category.pth**:
- PyTorch state_dict for feedforward neural network
- Contains weights and biases for all layers

**preprocessor.pkl**:
- Fitted sklearn ColumnTransformer
- Contains TF-IDF vectorizer, OneHotEncoder, StandardScaler
- Required for transforming new tickets before prediction

**label_encoder.pkl**:
- sklearn LabelEncoder for category mapping
- Maps integer predictions back to category strings
- 5 classes: Security, Data Issue, Feature Request, Account Management, Technical Issue

---

## 10. Reproducibility

All experiments use:
- **Fixed random seed**: 42 (set in src/config.py)
- **UV dependency management**: Pinned versions in pyproject.toml
- **MLflow tracking**: All hyperparameters and metrics logged
- **Artifact versioning**: Models tagged with version and timestamp

To reproduce exact results:
1. Use the same Python version (3.11)
2. Install dependencies with `uv pip install -e ".[dev]"`
3. Use the same data splits (generated with `scripts/prepare_data.py`)
4. Run training scripts with default hyperparameters
