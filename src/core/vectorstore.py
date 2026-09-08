"""
VectorStore Module (Qdrant Native Hybrid)
=========================================
Manages dense + sparse vector indexing using Qdrant's native hybrid retrieval.

Architecture:
    - Dense vectors:  sentence-transformers (local) → Qdrant dense vector
    - Sparse vectors: FastEmbed (BM25) → Qdrant native sparse vector (NOT in-memory BM25)
    - Hybrid fusion:  Qdrant Rust engine with RRF (Reciprocal Rank Fusion)
    - Persistence:    Qdrant local path mode (no server needed)

This design eliminates the 3 common anti-patterns:
    1. ✅ No more in-memory BM25 reconstruction — Qdrant stores sparse vectors natively
    2. ✅ No more fragile text[:200] key matching — Qdrant uses point IDs + RRF
    3. ✅ Compatible with both List[Dict] (from loader.py) and List[Document] (LangChain standard)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Prefetch, FusionQuery, Fusion
)
from qdrant_client.http.models import SparseVector as QdSparseVector

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────
#  Local Embeddings (wraps sentence-transformers for LangChain compatibility)
# ──────────────────────────────────────────────
class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings using local sentence-transformers.
    Avoids the need for an external API key.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logging.info(f"Loading local embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        logging.info("Embedding model loaded.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode documents into dense vectors."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encode a query into a dense vector."""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# ──────────────────────────────────────────────
#  Persistent Hybrid Vector Store
# ──────────────────────────────────────────────
class HybridVectorStore:
    """
    Production-grade persistent vector store with Qdrant native hybrid retrieval.

    Usage:
        store = HybridVectorStore(qdrant_path="qdrant_data")

        # First time: build from loader output
        chunks = PDFLoader(...).load_all_pdfs()  # List[Dict]
        store.load_or_build(chunks)  # Auto-detects format

        # Subsequent runs: loads from Qdrant instantly (skip PDF loading)
        store.load_or_build()

        # Search
        results = store.search("AMD net income 2022", mode="hybrid", top_k=5)
    """

    COLLECTION_NAME = "finance_rag_hybrid"
    SPARSE_VECTOR_NAME = "langchain-sparse"

    def __init__(
        self,
        qdrant_path: str,
        dense_model: str = "all-MiniLM-L6-v2",
        sparse_model: str = "Qdrant/bm25",
    ):
        """
        Initialize the hybrid vector store.

        Args:
            qdrant_path: Local directory for Qdrant persistent storage.
            dense_model: SentenceTransformer model for dense embeddings.
            sparse_model: FastEmbed model for sparse embeddings.
        """
        self.qdrant_path = qdrant_path
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model

        os.makedirs(qdrant_path, exist_ok=True)

        # Qdrant client (local persistent mode — no server needed)
        self.client = QdrantClient(path=qdrant_path) #数据库连接

        # Lazy-loaded components
        self._embeddings: Optional[Embeddings] = None
        self._sparse_embeddings: Optional[FastEmbedSparse] = None
        self._vectorstore: Optional[QdrantVectorStore] = None

    # ── Lazy initialization ──

    @property
    def embeddings(self) -> Embeddings:
        """Lazy-load dense embeddings model."""
        if self._embeddings is None:
            self._embeddings = LocalSentenceTransformerEmbeddings(self.dense_model_name)
        return self._embeddings

    @property
    def sparse_embeddings(self) -> FastEmbedSparse:
        """Lazy-load sparse embeddings model (FastEmbed BM25)."""
        if self._sparse_embeddings is None:
            logging.info(f"Loading sparse embedding model: {self.sparse_model_name} ...")
            self._sparse_embeddings = FastEmbedSparse(model_name=self.sparse_model_name)
            logging.info("Sparse embedding model loaded.")
        return self._sparse_embeddings

    # ── Collection management ──

    def _collection_exists_with_data(self) -> bool:
        """Check if the collection exists and contains points."""
        try: #如果查到的数量 大于 0，返回 True；如果数量 等于 0（空表），返回 False
            collections = self.client.get_collections().collections
            col_names = [c.name for c in collections]
            if self.COLLECTION_NAME in col_names:
                count = self.client.count(collection_name=self.COLLECTION_NAME).count
                return count > 0
            return False
        except Exception as e:
            logging.warning(f"Failed to check collection: {e}")
            return False

    def _create_hybrid_collection(self) -> None:
        """Create collection with BOTH dense and sparse vector configs."""
        logging.info(f"Creating collection '{self.COLLECTION_NAME}' with hybrid vectors ...")

        # Clean up if exists (fresh start)
        try: #构建数据库
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        # Get embedding dimension from the model
        test_embedding = self.embeddings.embed_query("test")
        embedding_dim = len(test_embedding)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        logging.info(
            f"Collection created: dense={embedding_dim}-dim COSINE, "
            f"sparse=FastEmbed BM25"
        )

    # ── Data conversion ──

    @staticmethod
    def _chunks_to_documents(
        chunks: List[Union[Dict[str, Any], Document]],
    ) -> List[Document]:
        """
        Convert chunks (Dict or Document) to LangChain Documents.

        Handles both:
            - loader.py output: List[Dict] with keys: text, metadata, page
            - LangChain standard: List[Document]
        """
        #这个staticmethod表示不用实例化就能调用函数，还有一个功能是没有@staticmethod会把obj自身作为第一个参数传给self
        documents = []
        for chunk in chunks:
            if isinstance(chunk, Document):
                documents.append(chunk)
            elif isinstance(chunk, dict):
                metadata = chunk.get("metadata", {})
                # Add page to metadata for retrieval
                page = chunk.get("page", 0)
                metadata["page"] = page

                doc = Document(
                    page_content=chunk["text"],
                    metadata=metadata,
                )
                documents.append(doc)
            else:
                raise TypeError(f"Unsupported chunk type: {type(chunk)}")
        #就是把chunk转成document格式然后然后一个document的列表
        return documents

    # ── Build / Load ──

    def build_index(
        self,
        chunks: List[Union[Dict[str, Any], Document]],
        batch_size: int = 100,
    ) -> None:
        """
        Build a fresh hybrid index from chunks.

        Args:
            chunks: Data from loader.py (List[Dict]) or LangChain (List[Document]).
            batch_size: Batch size for vector upsert.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        logging.info(f"Converting {len(chunks)} chunks to LangChain Documents ...")
        documents = self._chunks_to_documents(chunks)

        # Create collection with hybrid config
        self._create_hybrid_collection()

        # Initialize QdrantVectorStore with HYBRID mode (stores both dense + sparse)
        self._vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name=self.SPARSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        # Add documents in batches
        total = len(documents)
        logging.info(f"Adding {total} documents to Qdrant in batches of {batch_size} ...")

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            self._vectorstore.add_documents(batch)
            progress = min(i + batch_size, total)
            logging.info(f"  Upserted {progress}/{total} points")

        logging.info(f"✅ Index build complete. Total points: {total}")

    def load_index(self) -> bool:
        """
        Load existing index from Qdrant (skip PDF loading).

        Returns:
            True if loaded successfully, False if no index exists.
        """
        if not self._collection_exists_with_data():
            logging.warning(f"Collection '{self.COLLECTION_NAME}' does not exist or is empty.")
            return False

        logging.info(f"Loading existing collection '{self.COLLECTION_NAME}' ...")

        # Build QdrantVectorStore directly (avoid from_existing_collection client issue)
        self._vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name=self.SPARSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        count = self.client.count(collection_name=self.COLLECTION_NAME).count
        logging.info(f"✅ Index loaded. {count} points ready for hybrid retrieval.")
        return True

    def load_or_build(
        self,
        chunks: Optional[List[Union[Dict[str, Any], Document]]] = None,
    ) -> None:
        """
        Core pattern: load from Qdrant if exists, otherwise build from chunks.

        Args:
            chunks: Required only when building a new index (first time).
        """
        if self.load_index():
            return

        if chunks is None:
            raise ValueError(
                "No existing index found and no chunks provided. "
                "Run PDFLoader first or provide chunks."
            )

        self.build_index(chunks)

    def delete_index(self) -> None:
        """Delete the collection (reset)."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._vectorstore = None
            logging.info(f"✅ Collection '{self.COLLECTION_NAME}' deleted.")
        except Exception as e:
            logging.error(f"Failed to delete collection: {e}")

    # ── Search ──

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        query_filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with specified mode using Qdrant's query_points API.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            mode: 'dense', 'sparse', or 'hybrid'.
            query_filter: Optional qdrant_client.models.Filter，用于 metadata 过滤
                （对齐 financebench-rag-eval retrieve.py 的 2-step relax 机制）。

        Returns:
            List of result dicts with keys: text, metadata, page, score, search_type.
        """
        if self._vectorstore is None:
            raise RuntimeError("VectorStore not initialized. Call load_or_build() first.")

        # Encode query vectors
        query_vector = self.embeddings.embed_query(query)
        sparse_raw = self.sparse_embeddings.embed_query(query)
        # Convert langchain_qdrant SparseVector → qdrant_client SparseVector
        sparse_vector = QdSparseVector(indices=sparse_raw.indices, values=sparse_raw.values)

        if mode == "dense":
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            points = response.points
        elif mode == "sparse":
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=sparse_vector,
                using=self.SPARSE_VECTOR_NAME,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            points = response.points
        elif mode == "hybrid":
            # Hybrid: dense / sparse 两路对称 prefetch（各 top_k*3 候选池）
            # → Qdrant 引擎内原生 RRF 融合（Reciprocal Rank Fusion）。
            # 注意：两路候选池必须对称，若 dense 只参与最终 limit 条融合，
            # 弱检索路（BM25）的噪声会把 dense 相关结果挤出 top_k，导致召回反降。
            prefetch_limit = top_k * 3
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=query_vector, limit=prefetch_limit, filter=query_filter),
                    Prefetch(
                        query=sparse_vector,
                        using=self.SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            points = response.points
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'dense', 'sparse', or 'hybrid'.")

        # Convert results
        converted = []
        for point in points:
            payload = point.payload or {}
            # LangChain QdrantVectorStore stores as 'page_content' (not 'text')
            text = payload.get("page_content", payload.get("text", ""))
            metadata = payload.get("metadata", {})
            page = metadata.get("page", 0) if isinstance(metadata, dict) else 0
            result = {
                "text": text,
                "metadata": metadata,
                "page": page,
                "score": float(getattr(point, "score", 0.0)),
                "search_type": mode,
            }
            converted.append(result)

        return converted

    def hybrid_search_with_rrf(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF) with Qdrant point IDs.

        Use this when you need fine-grained control over RRF parameters.
        For simpler use cases, prefer search(mode="hybrid") which uses
        Qdrant's native RRF in Rust.

        RRF formula: score = Σ 1/(k + rank_i)  for each document across rankings

        Args:
            query: Search query.
            top_k: Number of results to return.
            rrf_k: RRF smoothing parameter (default 60).

        Returns:
            List of result dicts with RRF scores.
        """
        if self._vectorstore is None:
            raise RuntimeError("VectorStore not initialized.")

        query_vector = self.embeddings.embed_query(query)
        sparse_raw = self.sparse_embeddings.embed_query(query)
        sparse_vector = QdSparseVector(indices=sparse_raw.indices, values=sparse_raw.values)

        # Dense search
        dense_response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k * 3,
            with_payload=True,
        )
        dense_points = dense_response.points

        # Sparse search
        sparse_response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=sparse_vector,
            using=self.SPARSE_VECTOR_NAME,
            limit=top_k * 3,
            with_payload=True,
        )
        sparse_points = sparse_response.points

        # RRF fusion using point IDs (zero collision)
        rrf_scores: Dict[int, float] = {}
        doc_cache: Dict[int, Dict[str, Any]] = {}

        for rank, point in enumerate(dense_points):
            pid = point.id
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (rrf_k + rank + 1)
            if pid not in doc_cache:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                doc_cache[pid] = {
                    "text": payload.get("page_content", payload.get("text", "")),
                    "metadata": metadata,
                    "page": metadata.get("page", 0) if isinstance(metadata, dict) else 0,
                }

        for rank, point in enumerate(sparse_points):
            pid = point.id
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (rrf_k + rank + 1)
            if pid not in doc_cache:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                doc_cache[pid] = {
                    "text": payload.get("page_content", payload.get("text", "")),
                    "metadata": metadata,
                    "page": metadata.get("page", 0) if isinstance(metadata, dict) else 0,
                }

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for pid in sorted_ids[:top_k]:
            result = {
                **doc_cache[pid],
                "score": rrf_scores[pid],
                "search_type": "hybrid_rrf",
            }
            results.append(result)

        return results

    def get_retriever(self, k: int = 4):
        """
        Get a configured LangChain retriever with hybrid search.

        Args:
            k: Number of documents to retrieve.

        Returns:
            LangChain BaseRetriever configured for hybrid search.
        """
        if self._vectorstore is None:
            raise RuntimeError("VectorStore not initialized. Call load_or_build() first.")

        # retrieval_mode is already set on the vectorstore during load/build
        # Do NOT pass it in search_kwargs — it causes AssertionError in MMR
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored index."""
        count = 0
        try:
            count = self.client.count(collection_name=self.COLLECTION_NAME).count
        except Exception:
            pass

        return {
            "point_count": count,
            "dense_model": self.dense_model_name,
            "sparse_model": self.sparse_model_name,
            "qdrant_path": self.qdrant_path,
            "collection_name": self.COLLECTION_NAME,
        }


# ──────────────────────────────────────────────
#  Main demo
# ──────────────────────────────────────────────
def main():
    """Demonstrate the hybrid vector store with load_or_build pattern."""

    data_dir = r"D:\finance_rag\finance_rag_evaluation\data"
    qdrant_path = r"D:\finance_rag\finance_rag_evaluation\qdrant_data"

    store = HybridVectorStore(
        qdrant_path=qdrant_path,
        dense_model="all-MiniLM-L6-v2",
        sparse_model="Qdrant/bm25",
    )

    # ── Key pattern: load from Qdrant if exists, otherwise build ──
    loaded = store.load_index()

    if not loaded:
        print("\n" + "=" * 60)
        print("FIRST RUN: Loading PDFs and building Qdrant hybrid index ...")
        print("=" * 60)

        from finance_rag_evaluation.src.core.loader import PDFLoader

        loader = PDFLoader(data_dir=data_dir, chunk_size=512, chunk_overlap=50)
        chunks = loader.load_all_pdfs()

        if not chunks:
            print("❌ No chunks to index. Exiting.")
            return

        store.build_index(chunks)
    else:
        print("\n" + "=" * 60)
        print("✅ Qdrant hybrid index loaded (skipped PDF loading).")
        print("=" * 60)

    # Show stats
    stats = store.get_stats()
    print(f"\nStore stats: {json.dumps(stats, indent=2)}")

    # ── Test retrieval ──
    print("\n" + "=" * 80)
    print("HYBRID SEARCH EXAMPLES")
    print("=" * 80)

    test_queries = [
        "What is Amazon's revenue for 2019?",
        "AMD net income 2022",
        "10-K annual report 3M",
        "quarterly earnings AMCOR",
    ]

    for query in test_queries:
        print(f"\n{'=' * 40}")
        print(f"Query: {query}")
        print(f"{'=' * 40}")

        for mode in ["dense", "sparse", "hybrid"]:
            print(f"\n--- {mode.upper()} SEARCH (top 3) ---")
            results = store.search(query, top_k=3, mode=mode)
            for i, r in enumerate(results, 1):
                metadata = r["metadata"]
                filename = metadata.get("original_filename", metadata.get("source", "N/A"))
                print(f"  [{i}] Score={r['score']:.4f} | File: {filename} | Page: {r['page']}")
                preview = r["text"][:150].replace("\n", " ")
                print(f"      {preview}...")


if __name__ == "__main__":
    main()