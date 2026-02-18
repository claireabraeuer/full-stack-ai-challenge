"""Minimal FastAPI application for intelligent support system.

Single file implementation with 4 essential endpoints:
- POST /process - Main pipeline (predict + retrieve)
- POST /feedback - Capture agent corrections
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from src.models.categorization import load_production_model
from src.models.monitoring import DriftDetector
from src.models.retrieval import HybridRAG
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Pydantic Models (Inline)
# ============================================================================


class TicketRequest(BaseModel):
    """Incoming ticket for processing."""

    subject: str = Field(..., description="Ticket subject")
    description: str = Field(..., description="Ticket description")
    product: Optional[str] = None
    priority: Optional[str] = None
    customer_tier: Optional[str] = None
    channel: Optional[str] = None
    previous_tickets: Optional[int] = 0
    satisfaction_score: Optional[float] = None
    escalated: Optional[bool] = False
    known_issue: Optional[bool] = False
    contains_error_code: Optional[bool] = False


class SimilarTicket(BaseModel):
    """Similar resolved ticket."""

    ticket_id: str
    similarity_score: float
    category: str
    product: str
    subject: str
    resolution: str
    satisfaction_score: float


class ProcessResponse(BaseModel):
    """Response from /process endpoint."""

    ticket_id: str
    predicted_category: str
    confidence: float
    model_version: str
    similar_tickets: List[SimilarTicket]


class FeedbackRequest(BaseModel):
    """Agent feedback on prediction/solution."""

    ticket_id: str
    true_category: Optional[str] = None
    resolution_helpful: Optional[bool] = None
    comments: Optional[str] = None


# ============================================================================
# Global State (Loaded at Startup)
# ============================================================================


class AppState:
    """Global application state."""

    def __init__(self):
        self.model = None
        self.metadata = None
        self.loader = None
        self.rag = None
        self.drift_detector = None

    def initialize(self):
        """Load models and indexes at startup."""
        logger.info("Initializing API...")

        # Load production model
        logger.info("Loading production model...")
        self.model, self.metadata, self.loader = load_production_model()
        logger.info(
            f"✓ Loaded {self.metadata['model_type']} v{self.metadata['version']}"
        )

        # Load RAG index
        logger.info("Loading RAG index...")
        self.rag = HybridRAG(db_path="data/chroma_db")
        logger.info("✓ Loaded RAG index")

        # Initialize drift detector
        self.drift_detector = DriftDetector(
            baseline_metrics=self.metadata.get("baseline_metrics")
        )
        logger.info("✓ Initialized drift detector")

        logger.info("API ready!")


# Global singleton
app_state = AppState()


# ============================================================================
# FastAPI App
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    app_state.initialize()
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Intelligent Support System",
    description="ML-powered ticket categorization and solution retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================


@app.post("/process", response_model=ProcessResponse)
async def process_ticket(ticket: TicketRequest):
    """Main endpoint: Predict category + retrieve solutions.

    This is the primary endpoint agents will use - provides both
    categorization and relevant solutions in one call.
    """
    try:
        # Generate ticket ID
        ticket_id = str(uuid.uuid4())

        # Step 1: Predict category
        ticket_dict = ticket.model_dump()

        # Add missing columns with default values (required by preprocessor)
        ticket_dict.setdefault("agent_experience_months", 12)
        ticket_dict.setdefault("resolution_time_hours", 0.0)
        ticket_dict.setdefault("first_response_time_hours", 0.0)
        ticket_dict.setdefault("reopened", False)
        ticket_dict.setdefault("customer_responses", 0)
        ticket_dict.setdefault("agent_responses", 0)

        ticket_df = pd.DataFrame([ticket_dict])
        prediction = app_state.loader.predict(
            app_state.model, ticket_df, return_confidence=True
        )

        # Record for drift monitoring
        app_state.drift_detector.record_prediction(
            ticket_id=ticket_id,
            predicted_category=prediction["predicted_category"],
            confidence=prediction["confidence"],
        )

        # Step 2: Retrieve similar solutions
        query = f"{ticket.subject} {ticket.description}"
        results = app_state.rag.retrieve(
            query=query,
            predicted_category=prediction["predicted_category"],
            product=ticket.product,
            top_k=5,
            semantic_weight=0.7,
            graph_weight=0.3,
        )

        # Build response
        similar_tickets = []
        for r in results:
            try:
                similar_tickets.append(
                    SimilarTicket(
                        ticket_id=r.get("ticket_id", "unknown"),
                        similarity_score=r.get("similarity", 0.0),
                        category=r.get("metadata", {}).get("category", "unknown"),
                        product=r.get("metadata", {}).get("product", "unknown"),
                        subject=r.get("metadata", {}).get("subject", "unknown"),
                        resolution=r.get("metadata", {}).get("resolution", "unknown"),
                        satisfaction_score=r.get("metadata", {}).get("satisfaction_score", 0.0),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse similar ticket: {e}, metadata: {r.get('metadata', {})}")
                continue

        return ProcessResponse(
            ticket_id=ticket_id,
            predicted_category=prediction["predicted_category"],
            confidence=prediction["confidence"],
            model_version=str(app_state.metadata["version"]),
            similar_tickets=similar_tickets,
        )

    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit agent feedback on prediction quality.

    Captures ground truth labels for future model retraining.
    """
    try:
        logger.info(f"Feedback received: {feedback.model_dump()}")
        # In production, save to database for retraining pipeline
        return None  # 204 No Content

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """System health check."""
    try:
        rag_stats = app_state.rag.get_stats()

        return {
            "status": "healthy",
            "model": {
                "type": app_state.metadata["model_type"],
                "version": str(app_state.metadata["version"]),
                "loaded": app_state.model is not None,
            },
            "rag": {
                "documents": rag_stats["vector_db"]["total_documents"],
                "graph_nodes": rag_stats["knowledge_graph"]["total_nodes"],
            },
            "drift": {
                "predictions_tracked": len(app_state.drift_detector.predictions),
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "Intelligent Support System",
        "version": "1.0.0",
        "endpoints": {
            "process": "POST /process - Main pipeline",
            "feedback": "POST /feedback - Submit corrections",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - Prometheus metrics",
            "docs": "GET /docs - Interactive API docs",
        },
    }
