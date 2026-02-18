"""Hybrid RAG system combining semantic search and knowledge graph retrieval."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import chromadb
import networkx as nx
import pandas as pd
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer


class HybridRAG:
    """Hybrid retrieval system combining semantic search with knowledge graph.

    Features:
    - Semantic search using sentence embeddings (ChromaDB)
    - Knowledge graph with entity relationships (NetworkX)
    - Hybrid retrieval merging both signals
    - Metadata filtering and intelligent re-ranking
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        db_path: str = "data/chroma_db",
        collection_name: str = "support_tickets",
    ):
        """Initialize hybrid RAG system.

        Args:
            embedding_model: Name of sentence-transformers model
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
        """
        self.embedding_model_name = embedding_model
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Initialize embedding model for semantic search
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Initialize ChromaDB for vector storage
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize knowledge graph
        self.graph = nx.Graph()
        self.entity_to_tickets = defaultdict(set)  # Fast lookup
        self.ticket_metadata = {}

        logger.info(f"Initialized hybrid RAG system")
        logger.info(f"  Vector DB collection: {collection_name}")
        logger.info(f"  Existing documents: {self.collection.count()}")
        logger.info(f"  Graph nodes: {self.graph.number_of_nodes()}")

    # -------------------------------------------------------------------------
    # Semantic Search Methods
    # -------------------------------------------------------------------------

    def _create_document_text(self, row: pd.Series) -> str:
        """Create searchable text from ticket data."""
        parts = [
            f"Subject: {row['subject']}",
            f"Description: {row['description']}",
            f"Category: {row['category']}",
            f"Product: {row['product']}",
        ]

        if pd.notna(row.get("error_logs")) and row["error_logs"]:
            parts.append(f"Error: {row['error_logs'][:200]}")

        return " | ".join(parts)

    def _create_metadata(self, row: pd.Series) -> Dict:
        """Extract metadata for filtering."""
        metadata = {
            "ticket_id": str(row["ticket_id"]),
            "category": str(row["category"]),
            "subcategory": str(row.get("subcategory", "")),
            "product": str(row["product"]),
            "priority": str(row["priority"]),
            "resolution_helpful": bool(row.get("resolution_helpful", True)),
            "satisfaction_score": float(row.get("satisfaction_score", 0)),
            "resolution_time_hours": float(row.get("resolution_time_hours", 0)),
        }

        if "escalated" in row:
            metadata["escalated"] = bool(row["escalated"])
        if "contains_error_code" in row:
            metadata["contains_error_code"] = bool(row["contains_error_code"])

        return metadata

    def _index_vectors(self, df: pd.DataFrame, batch_size: int = 100):
        """Index tickets into vector database."""
        logger.info(f"Indexing {len(df)} tickets into vector database...")

        # Clear existing collection
        if self.collection.count() > 0:
            logger.info("Clearing existing collection...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Process in batches
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch_num = i // batch_size + 1

            # Create documents and metadata
            documents = [self._create_document_text(row) for _, row in batch_df.iterrows()]
            metadatas = [self._create_metadata(row) for _, row in batch_df.iterrows()]
            ids = [str(row["ticket_id"]) for _, row in batch_df.iterrows()]

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                documents, show_progress_bar=False, convert_to_numpy=True
            ).tolist()

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"Processed batch {batch_num}/{total_batches}")

        logger.info(f"✓ Indexed {self.collection.count()} tickets")

    # -------------------------------------------------------------------------
    # Knowledge Graph Methods
    # -------------------------------------------------------------------------

    def _extract_error_codes(self, text: str) -> Set[str]:
        """Extract error codes from text."""
        if not text or pd.isna(text):
            return set()

        patterns = [
            r"ERROR_[A-Z]+_\d+",  # ERROR_TIMEOUT_429
            r"E-\d+",  # E-429
            r"ERR_[A-Z_]+",  # ERR_TIMEOUT
            r"ERROR-\d+",  # ERROR-429
        ]

        error_codes = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            error_codes.update(matches)

        return error_codes

    def _extract_entities(self, row: pd.Series) -> Dict[str, Set[str]]:
        """Extract all entities from a ticket."""
        entities = {
            "products": set(),
            "error_codes": set(),
            "tags": set(),
            "resolution_codes": set(),
        }

        # Product
        if pd.notna(row.get("product")):
            entities["products"].add(str(row["product"]))

        # Error codes
        error_text = str(row.get("error_logs", "")) + " " + str(row.get("description", ""))
        entities["error_codes"] = self._extract_error_codes(error_text)

        # Tags
        tags = row.get("tags")
        if tags is not None and not (isinstance(tags, float) and pd.isna(tags)):
            if isinstance(tags, list):
                entities["tags"] = set(tags)
            elif isinstance(tags, str):
                tags = tags.strip("[]").replace("'", "").replace('"', "")
                entities["tags"] = {t.strip() for t in tags.split(",") if t.strip()}

        # Resolution code
        if pd.notna(row.get("resolution_code")):
            entities["resolution_codes"].add(str(row["resolution_code"]))

        return entities

    def _build_knowledge_graph(self, df: pd.DataFrame):
        """Build knowledge graph from ticket data."""
        logger.info(f"Building knowledge graph from {len(df)} tickets...")

        # Clear existing graph
        self.graph.clear()
        self.entity_to_tickets.clear()
        self.ticket_metadata.clear()

        node_counts = defaultdict(int)

        for _, row in df.iterrows():
            ticket_id = str(row["ticket_id"])

            # Add ticket node
            self.graph.add_node(
                ticket_id,
                node_type="ticket",
                category=str(row.get("category", "")),
                satisfaction=float(row.get("satisfaction_score", 0)),
                resolution_helpful=bool(row.get("resolution_helpful", True)),
            )
            node_counts["ticket"] += 1

            # Store metadata
            self.ticket_metadata[ticket_id] = {
                "category": str(row.get("category", "")),
                "product": str(row.get("product", "")),
                "subject": str(row.get("subject", "")),
                "description": str(row.get("description", "")),
                "resolution": str(row.get("resolution", "")),
                "satisfaction_score": float(row.get("satisfaction_score", 0)),
                "resolution_helpful": bool(row.get("resolution_helpful", True)),
                "resolution_time_hours": float(row.get("resolution_time_hours", 0)),
            }

            # Extract and link entities
            entities = self._extract_entities(row)

            for entity_type, entity_values in entities.items():
                for entity_value in entity_values:
                    entity_node = f"{entity_type}:{entity_value}"

                    if not self.graph.has_node(entity_node):
                        self.graph.add_node(entity_node, node_type=entity_type)
                        node_counts[entity_type] += 1

                    self.graph.add_edge(entity_node, ticket_id)
                    self.entity_to_tickets[entity_node].add(ticket_id)

        logger.info("✓ Knowledge graph built successfully")
        logger.info(f"  Total nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"  Total edges: {self.graph.number_of_edges():,}")
        for node_type, count in sorted(node_counts.items()):
            logger.info(f"  {node_type.capitalize()} nodes: {count:,}")

    def extract_entities_from_text(self, text: str) -> Dict[str, Set[str]]:
        """Extract entities from query text."""
        entities = {
            "products": set(),
            "error_codes": set(),
            "tags": set(),
            "resolution_codes": set(),
        }

        # Extract error codes
        entities["error_codes"] = self._extract_error_codes(text)

        # Extract products (match against known products)
        text_lower = text.lower()
        for node in self.graph.nodes():
            if node.startswith("products:"):
                product = node.split(":", 1)[1]
                if product.lower() in text_lower:
                    entities["products"].add(product)

        # Extract tags (match against known tags)
        for node in self.graph.nodes():
            if node.startswith("tags:"):
                tag = node.split(":", 1)[1]
                if tag.lower() in text_lower:
                    entities["tags"].add(tag)

        return entities

    def find_related_tickets_by_entities(
        self, query_entities: Dict[str, Set[str]], top_k: int = 10
    ) -> List[str]:
        """Find tickets related through shared entities."""
        ticket_scores = defaultdict(int)

        for entity_type, entity_values in query_entities.items():
            for entity_value in entity_values:
                entity_node = f"{entity_type}:{entity_value}"
                related_tickets = self.entity_to_tickets.get(entity_node, set())

                for ticket_id in related_tickets:
                    # Higher weight for error codes (strong signal)
                    weight = 3 if entity_type == "error_codes" else 1
                    ticket_scores[ticket_id] += weight

        ranked_tickets = sorted(
            ticket_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [ticket_id for ticket_id, score in ranked_tickets[:top_k]]

    # -------------------------------------------------------------------------
    # Hybrid Retrieval
    # -------------------------------------------------------------------------

    def build_index(self, df: pd.DataFrame, batch_size: int = 100):
        """Build both vector database and knowledge graph.

        Args:
            df: DataFrame with ticket data
            batch_size: Batch size for vector indexing
        """
        logger.info("=" * 80)
        logger.info("Building Hybrid RAG Index")
        logger.info("=" * 80)

        # Build semantic index
        logger.info("\n[1/2] Building semantic vector index...")
        self._index_vectors(df, batch_size)

        # Build knowledge graph
        logger.info("\n[2/2] Building knowledge graph...")
        self._build_knowledge_graph(df)

        logger.info("\n" + "=" * 80)
        logger.info("✓ Hybrid RAG index built successfully!")
        logger.info("=" * 80)

    def retrieve(
        self,
        query: str,
        predicted_category: Optional[str] = None,
        product: Optional[str] = None,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3,
        min_satisfaction: float = 3.0,
    ) -> List[Dict]:
        """Retrieve tickets using hybrid semantic + graph search.

        Args:
            query: Query text (subject + description)
            predicted_category: Predicted category for filtering
            product: Product filter
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            graph_weight: Weight for graph-based score (0-1)
            min_satisfaction: Minimum satisfaction score filter

        Returns:
            List of retrieved tickets with hybrid scores
        """
        # 1. Semantic retrieval
        semantic_results = self._semantic_retrieve(
            query, predicted_category, product, top_k * 2, min_satisfaction
        )

        # 2. Graph retrieval
        query_entities = self.extract_entities_from_text(query)
        graph_ticket_ids = self.find_related_tickets_by_entities(
            query_entities, top_k=top_k * 2
        )

        # 3. Merge results with hybrid scoring
        ticket_scores = {}

        # Score from semantic search
        for i, result in enumerate(semantic_results):
            ticket_id = result["ticket_id"]
            semantic_score = result.get("final_score", result.get("similarity", 0))
            ticket_scores[ticket_id] = {
                "semantic_score": semantic_score,
                "graph_score": 0,
                "result": result,
            }

        # Score from graph search
        for i, ticket_id in enumerate(graph_ticket_ids):
            graph_score = 1.0 - (i / len(graph_ticket_ids))

            if ticket_id in ticket_scores:
                ticket_scores[ticket_id]["graph_score"] = graph_score
            else:
                # Ticket found by graph but not semantic
                if ticket_id in self.ticket_metadata:
                    ticket_scores[ticket_id] = {
                        "semantic_score": 0,
                        "graph_score": graph_score,
                        "result": {
                            "ticket_id": ticket_id,
                            "metadata": self.ticket_metadata[ticket_id],
                            "similarity": 0,
                            "final_score": 0,
                        },
                    }

        # 4. Compute hybrid scores
        for ticket_id in ticket_scores:
            sem_score = ticket_scores[ticket_id]["semantic_score"]
            grp_score = ticket_scores[ticket_id]["graph_score"]

            hybrid_score = semantic_weight * sem_score + graph_weight * grp_score

            ticket_scores[ticket_id]["hybrid_score"] = hybrid_score
            ticket_scores[ticket_id]["result"]["hybrid_score"] = hybrid_score

        # 5. Rank by hybrid score
        ranked = sorted(
            ticket_scores.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return [item["result"] for item in ranked[:top_k]]

    def _semantic_retrieve(
        self,
        query: str,
        predicted_category: Optional[str] = None,
        product: Optional[str] = None,
        top_k: int = 5,
        min_satisfaction: float = 3.0,
    ) -> List[Dict]:
        """Perform semantic search using vector embeddings."""
        # Build metadata filter
        where_filter = {}
        where_conditions = []

        if predicted_category:
            where_conditions.append({"category": predicted_category})

        if product:
            where_conditions.append({"product": product})

        where_conditions.append({"satisfaction_score": {"$gte": min_satisfaction}})

        if len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}
        elif len(where_conditions) == 1:
            where_filter = where_conditions[0]

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        # Perform search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None,
            )
        except Exception as e:
            logger.warning(f"Query with filters failed: {e}, retrying without filters")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

        # Parse results
        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved = []
        for i, ticket_id in enumerate(results["ids"][0]):
            doc = {
                "ticket_id": ticket_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],
            }
            retrieved.append(doc)

        # Re-rank
        return self._rerank_results(retrieved, predicted_category)

    def _rerank_results(
        self, results: List[Dict], predicted_category: Optional[str] = None
    ) -> List[Dict]:
        """Re-rank results based on multiple factors."""
        for result in results:
            score = result["similarity"]

            # Boost by resolution helpfulness
            if result["metadata"].get("resolution_helpful", False):
                score *= 1.2

            # Boost by satisfaction
            satisfaction = result["metadata"].get("satisfaction_score", 0)
            score *= 1 + (satisfaction - 3) * 0.1

            # Boost by category match
            if predicted_category and result["metadata"].get("category") == predicted_category:
                score *= 1.3

            # Adjust by resolution time
            resolution_time = result["metadata"].get("resolution_time_hours", 24)
            if resolution_time < 4:
                score *= 1.1
            elif resolution_time > 48:
                score *= 0.9

            result["final_score"] = score

        return sorted(results, key=lambda x: x["final_score"], reverse=True)

    def get_stats(self) -> Dict:
        """Get statistics about the hybrid RAG system."""
        stats = {
            "vector_db": {
                "total_documents": self.collection.count(),
                "embedding_model": self.embedding_model_name,
                "embedding_dim": self.embedding_dim,
            },
            "knowledge_graph": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
            },
        }

        # Entity statistics
        for entity_type in ["products", "error_codes", "tags", "resolution_codes"]:
            entity_counts = {}
            for node in self.graph.nodes():
                if node.startswith(f"{entity_type}:"):
                    entity_value = node.split(":", 1)[1]
                    ticket_count = len(
                        [n for n in self.graph.neighbors(node) if n.startswith("TK-")]
                    )
                    entity_counts[entity_value] = ticket_count

            top_entities = sorted(
                entity_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            stats[entity_type] = {
                "total": len(entity_counts),
                "top_10": top_entities,
            }

        return stats
