from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # type: ignore # noqa: F401

            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _next_id(self) -> str:
        idx = self._next_index
        self._next_index += 1
        return str(idx)

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        return {
            "id": self._next_id(),
            "document": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": {
                "doc_id": doc.id,
                "source": doc.metadata.get("source", ""),
            },
        }

    def _search_records(
        self, query: str, records: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        query_embedding = self._embedding_fn(query)
        scored = [
            {**record, "score": compute_similarity(query_embedding, record["embedding"]), "content": record["document"]}
            for record in records
        ]
        return sorted(scored, key=lambda r: r["score"], reverse=True)[:top_k]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...], metadatas=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma:
            self._collection.add(
                ids=[r["id"] for r in records],
                documents=[r["document"] for r in records],
                embeddings=[r["embedding"] for r in records],
                metadatas=[r["metadata"] for r in records],
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        if self._use_chroma:
            return self._chroma_search(query, top_k=top_k)
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(
        self, query: str, top_k: int = 3, metadata_filter: dict | None = None
    ) -> list[dict[str, Any]]:
        """
        Search with optional metadata pre-filtering.

        For ChromaDB: pass where= clause directly.
        For in-memory: filter self._store then run similarity search.
        """
        if metadata_filter is None:
            metadata_filter = {}

        if self._use_chroma:
            return self._chroma_search(query, top_k=top_k, where=metadata_filter or None)

        filtered = [
            record
            for record in self._store
            if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            results = self._collection.get(where={"doc_id": doc_id})
            ids_to_delete = results.get("ids", [])
            if not ids_to_delete:
                return False
            self._collection.delete(ids=ids_to_delete)
            return True

        initial_size = len(self._store)
        self._store = [r for r in self._store if r["metadata"]["doc_id"] != doc_id]
        return len(self._store) < initial_size

    # ------------------------------------------------------------------ #
    # ChromaDB helpers                                                     #
    # ------------------------------------------------------------------ #

    def _chroma_search(
        self, query: str, top_k: int, where: dict | None = None
    ) -> list[dict[str, Any]]:
        """Query ChromaDB and normalize results to the common dict format."""
        query_embedding = self._embedding_fn(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content": doc,
                "document": doc,
                "metadata": meta,
                # ChromaDB returns L2 distance; convert to a similarity-like score
                "score": 1.0 / (1.0 + dist),
            })
        return output
