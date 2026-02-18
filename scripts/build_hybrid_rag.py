"""Build hybrid RAG index (vector DB + knowledge graph)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.data import load_splits
from src.models.retrieval import HybridRAG


def main():
    """Build hybrid RAG index from training data."""
    logger.info("=" * 80)
    logger.info("Building Hybrid RAG Index")
    logger.info("=" * 80)

    # Load training data
    logger.info("\n[1/3] Loading ticket data...")
    train_df, val_df, test_df = load_splits("data/splits")

    knowledge_base = train_df.copy()
    logger.info(f"Knowledge base size: {len(knowledge_base):,} tickets")

    # Initialize hybrid RAG
    logger.info("\n[2/3] Initializing hybrid RAG system...")
    hybrid_rag = HybridRAG(
        embedding_model="all-MiniLM-L6-v2",
        db_path="data/chroma_db",
        collection_name="support_tickets",
    )

    # Build index (both vector DB and knowledge graph)
    logger.info("\n[3/3] Building index...")
    hybrid_rag.build_index(knowledge_base, batch_size=100)

    # Show statistics
    logger.info("\n" + "=" * 80)
    logger.info("System Statistics")
    logger.info("=" * 80)

    stats = hybrid_rag.get_stats()

    logger.info(f"\nVector Database:")
    logger.info(f"  Total documents: {stats['vector_db']['total_documents']:,}")
    logger.info(f"  Embedding model: {stats['vector_db']['embedding_model']}")
    logger.info(f"  Embedding dimension: {stats['vector_db']['embedding_dim']}")

    logger.info(f"\nKnowledge Graph:")
    logger.info(f"  Total nodes: {stats['knowledge_graph']['total_nodes']:,}")
    logger.info(f"  Total edges: {stats['knowledge_graph']['total_edges']:,}")

    for entity_type in ["products", "error_codes", "tags", "resolution_codes"]:
        data = stats.get(entity_type, {})
        if data.get("total", 0) > 0:
            logger.info(f"\n{entity_type.replace('_', ' ').title()}:")
            logger.info(f"  Total unique: {data['total']}")
            logger.info(f"  Top 10:")
            for entity, count in data.get("top_10", [])[:10]:
                logger.info(f"    {entity}: {count} tickets")

    # Test hybrid retrieval
    logger.info("\n" + "=" * 80)
    logger.info("Testing Hybrid Retrieval")
    logger.info("=" * 80)

    test_query = "Database sync timeout error ERROR_TIMEOUT_429 when processing large files"
    logger.info(f"\nQuery: {test_query}")

    # Extract entities
    query_entities = hybrid_rag.extract_entities_from_text(test_query)
    logger.info(f"\nExtracted entities:")
    for entity_type, values in query_entities.items():
        if values:
            logger.info(f"  {entity_type}: {values}")

    # Semantic only (graph_weight=0)
    logger.info("\n--- Semantic Only (graph_weight=0.0) ---")
    semantic_results = hybrid_rag.retrieve(
        query=test_query,
        predicted_category="Technical Issue",
        top_k=5,
        semantic_weight=1.0,
        graph_weight=0.0,
    )

    for i, result in enumerate(semantic_results, 1):
        logger.info(f"\n{i}. Ticket {result['ticket_id']}")
        logger.info(f"   Hybrid Score: {result.get('hybrid_score', 0):.3f}")
        logger.info(f"   Category: {result['metadata']['category']}")
        logger.info(f"   Product: {result['metadata']['product']}")

    # Hybrid (semantic + graph)
    logger.info("\n--- Hybrid (semantic=0.7, graph=0.3) ---")
    hybrid_results = hybrid_rag.retrieve(
        query=test_query,
        predicted_category="Technical Issue",
        top_k=5,
        semantic_weight=0.7,
        graph_weight=0.3,
    )

    for i, result in enumerate(hybrid_results, 1):
        logger.info(f"\n{i}. Ticket {result['ticket_id']}")
        logger.info(f"   Hybrid Score: {result.get('hybrid_score', 0):.3f}")
        logger.info(f"   Category: {result['metadata']['category']}")
        logger.info(f"   Product: {result['metadata']['product']}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Hybrid RAG system ready!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
