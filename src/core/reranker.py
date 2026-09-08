"""
Reranker Module (Intelligent Re-ranking)
=========================================
Implements multi-stage reranking and post-processing for RAG pipelines.
Logic aligned with the financebench-rag-eval project (CRAG-style pipeline):
    retrieve (wide, k=8~15) -> rerank (strict, top=5~10) -> grade -> generate

Architecture:
    - get_reranker: Lazy singleton factory (model loaded once, reused globally).
    - Base Reranker: Abstract interface for all reranking models.
    - CrossEncoder Reranker: Local CrossEncoder (default BAAI/bge-reranker-base).
    - Reranker Pipeline: Chains multiple rerankers and post-processing steps.
    - Post-processing: Relevance thresholding & Source diversity filtering.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────
#  Lazy Singleton (mirrors clients.py in
#  financebench-rag-eval: one CrossEncoder,
#  loaded once, shared across the whole pipeline)
# ──────────────────────────────────────────────
_reranker_instance: Optional["CrossEncoderReranker"] = None
_reranker_lock = threading.Lock()


def get_reranker(
    model_name: str = "BAAI/bge-reranker-base",
    device: Optional[str] = None,
) -> "CrossEncoderReranker":
    """Return the global CrossEncoderReranker instance (lazy, thread-safe)."""
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                _reranker_instance = CrossEncoderReranker(model_name=model_name, device=device)
    return _reranker_instance


# ──────────────────────────────────────────────
#  Base Reranker Interface
# ──────────────────────────────────────────────
class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The search query.
            documents: List of document dicts (must contain 'text' key).
            top_k: Number of top documents to return.

        Returns:
            Reranked list of documents with added 'rerank_score' key.
        """
        raise NotImplementedError


# ──────────────────────────────────────────────
#  Cross-Encoder Reranker (Local & Fast)
# ──────────────────────────────────────────────
class CrossEncoderReranker(BaseReranker):
    """
    Reranker using SentenceTransformers CrossEncoder models.

    Key points aligned with financebench-rag-eval:
        - Default model 'BAAI/bge-reranker-base': lightweight, low latency,
          proven effective on FinanceBench (retrieve k=8 -> rerank top=5).
        - Batch predict, then sort descending and cut to top_k ("strict out").
        - Optional sigmoid normalization so threshold filtering (0~1) is meaningful.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_scores: bool = True,
        show_progress: bool = False,
    ):
        logging.info(f"Loading CrossEncoder Reranker: {model_name} ...")
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        self.normalize_scores = normalize_scores
        self.show_progress = show_progress
        logging.info("CrossEncoder Reranker loaded.")

    def _predict(self, pairs: List[List[str]]) -> np.ndarray:
        """Score query-document pairs; optionally squash logits to (0, 1)."""
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
        )
        scores = np.asarray(scores, dtype=float).ravel()
        if self.normalize_scores:
            # bge-reranker outputs raw logits; sigmoid maps them to (0, 1) so
            # PostProcessor.filter_by_threshold(threshold=0.x) behaves correctly.
            # Monotonic transform: does not change the ranking order.
            scores = 1.0 / (1.0 + np.exp(-scores))
        return scores

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        top_k = max(1, min(top_k, len(documents)))

        # Create query-document pairs and compute relevance scores
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self._predict(pairs)

        # Attach scores to documents
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            scored_docs.append(doc_copy)

        # Sort by score descending (same as: sorted(zip(scores, docs), reverse=True))
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_k]


# ──────────────────────────────────────────────
#  Post-Processing Strategies
# ──────────────────────────────────────────────
class PostProcessor:
    """Applies filtering and diversity constraints to reranked results."""

    @staticmethod
    def filter_by_threshold(
        docs: List[Dict[str, Any]],
        threshold: float = 0.0,
        score_key: str = "rerank_score"
    ) -> List[Dict[str, Any]]:
        """Drop documents below a certain relevance score (expect normalized 0~1)."""
        return [doc for doc in docs if doc.get(score_key, 0.0) >= threshold]

    @staticmethod
    def enforce_diversity(
        docs: List[Dict[str, Any]],
        max_per_source: int = 1,
        source_key: str = "original_filename"
    ) -> List[Dict[str, Any]]:
        """
        Ensure diversity by limiting how many chunks can come from the same source file.
        Prevents a single PDF from dominating the top_k results.
        """
        source_counts: Dict[str, int] = {}
        diverse_docs = []

        for doc in docs:
            metadata = doc.get("metadata", {})
            source = metadata.get(source_key, "unknown")

            if source_counts.get(source, 0) < max_per_source:
                diverse_docs.append(doc)
                source_counts[source] = source_counts.get(source, 0) + 1

        return diverse_docs


# ──────────────────────────────────────────────
#  Reranker Pipeline (Chaining & Composition)
# ──────────────────────────────────────────────
class RerankerPipeline(BaseReranker):
    """
    Chains multiple rerankers and post-processing steps together.
    Implements the "wide-in, strict-out" strategy from financebench-rag-eval:
    CrossEncoder (Top 15) -> Threshold Filter -> Diversity Filter -> Top 5
    """

    def __init__(
        self,
        rerankers: List[BaseReranker],
        threshold: Optional[float] = None,
        max_per_source: Optional[int] = None,
        final_top_k: int = 5
    ):
        self.rerankers = rerankers
        self.threshold = threshold
        self.max_per_source = max_per_source
        self.final_top_k = final_top_k

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.final_top_k
        if not documents:
            return []

        current_docs = documents

        # 1. Pass through all rerankers in sequence.
        #    Intermediate top_k is wider (x3) so later stages can still filter.
        for reranker in self.rerankers:
            current_docs = reranker.rerank(query, current_docs, top_k=top_k * 3)

        # 2. Apply Post-processing
        if self.threshold is not None:
            current_docs = PostProcessor.filter_by_threshold(current_docs, self.threshold)

        if self.max_per_source is not None:
            current_docs = PostProcessor.enforce_diversity(current_docs, self.max_per_source)

        # 3. Final strict cut
        return current_docs[:top_k]


# ──────────────────────────────────────────────
#  Main Demo
# ──────────────────────────────────────────────
def main():
    """Demonstrate the Reranker Pipeline (wide-in, strict-out)."""

    # Mock data simulating output from HybridVectorStore.search()
    # (wide retrieval: k=8~15, then strict rerank: top=5)
    mock_retrieved_docs = [
        {"text": "AMD net income was $1.3 billion in 2022.", "metadata": {"original_filename": "AMD_2022.pdf"}, "score": 0.85},
        {"text": "AMD net income was $3.1 billion in 2021.", "metadata": {"original_filename": "AMD_2022.pdf"}, "score": 0.82}, # Duplicate source
        {"text": "The weather in New York is sunny.", "metadata": {"original_filename": "Weather.pdf"}, "score": 0.40}, # Irrelevant
        {"text": "Intel reported a net income of $8 billion.", "metadata": {"original_filename": "Intel_2022.pdf"}, "score": 0.75},
    ]

    query = "What was AMD's net income in 2022?"

    # Initialize Pipeline with the shared singleton reranker
    pipeline = RerankerPipeline(
        rerankers=[
            get_reranker()  # BAAI/bge-reranker-base, loaded once, reused globally
        ],
        threshold=0.5,          # Meaningful now: scores are sigmoid-normalized to 0~1
        max_per_source=1,       # Max 1 chunk per PDF
        final_top_k=2
    )

    print(f"\nQuery: {query}")
    print(f"Initial retrieved: {len(mock_retrieved_docs)} docs")

    final_results = pipeline.rerank(query, mock_retrieved_docs)

    print(f"\nFinal Reranked Results ({len(final_results)} docs):")
    for i, doc in enumerate(final_results, 1):
        print(f"  [{i}] Score={doc['rerank_score']:.4f} | Source: {doc['metadata']['original_filename']}")
        print(f"      {doc['text']}")


if __name__ == "__main__":
    main()
