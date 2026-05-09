"""
ChromaDB-backed vector store with hybrid semantic and BM25 search.

Embeddings live in Chroma (docs collection, one row per child chunk).
All other persistent state. Stored model name, file hashes, and parent
section text. Lives in SQLite via MetadataDB. The Chroma index is
therefore a rebuildable cache over the canonical data in SQLite.

Hybrid retrieval combines BGE cosine similarity with BM25 keyword
ranking via Reciprocal Rank Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.errors import ChromaError, NotFoundError

from rag.errors import StoreError
from rag.metadata_db import MetadataDB

if TYPE_CHECKING:
    from pathlib import Path

    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_QUERY_PREFIX = ""
DEFAULT_FETCH_N_MULTIPLIER = 4
DEFAULT_RRF_K = 60
DEFAULT_RERANKER_ENABLED = False
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# BGE alternative. Change all three together. See config.py and README for details.
DEFAULT_RERANKER_POOL_SIZE = 20


@dataclass
class SearchResult:
    """A single result returned by VectorStore.search."""

    filename: str
    excerpt: str
    section: str
    score: float


class VectorStore:
    """Persistent ChromaDB store with hybrid search and parent-child retrieval."""

    def __init__(
        self,
        data_dir: Path,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        fetch_n_multiplier: int = DEFAULT_FETCH_N_MULTIPLIER,
        rrf_k: int = DEFAULT_RRF_K,
        reranker_enabled: bool = DEFAULT_RERANKER_ENABLED,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_pool_size: int = DEFAULT_RERANKER_POOL_SIZE,
    ) -> None:
        """
        Initialise the vector store, creating the index and metadata DB if needed.

        Args:
            data_dir: Directory where ChromaDB and the SQLite metadata DB persist.
            model_name: SentenceTransformer model identifier. Changing this
                        between runs triggers a full reindex to avoid dimension
                        mismatches.
            query_prefix: String prepended to every search query before
                          embedding. BGE v1.5 requires a specific phrase. E5
                          uses "query: ". Must match the conventions of
                          model_name.
            fetch_n_multiplier: Candidate pool multiplier: search fetches
                                top_k * fetch_n_multiplier from each of
                                semantic and BM25 before fusing with RRF.
            rrf_k: Reciprocal Rank Fusion K parameter. Higher values flatten
                   the ranking curve. The literature commonly uses 60.
            reranker_enabled: When True, a cross-encoder reranks the top
                              reranker_pool_size parent sections after RRF
                              fusion and the result is truncated to top_k.
            reranker_model: CrossEncoder identifier to load when reranking.
                            Ignored when reranker_enabled is False.
            reranker_pool_size: How many deduplicated parent sections to
                                rerank. Must be >= top_k when reranking is
                                enabled.
        """

        # Yeah, I know, imports inside functions are usually a code smell.
        # Deferred so that importing this module does not open a PersistentClient.
        # Code paths that mock or bypass VectorStore (e.g. indexer-only tests) pay
        # no ChromaDB startup cost.
        import chromadb  # noqa: PLC0415
        from chromadb.config import Settings  # noqa: PLC0415

        self._model_name = model_name
        self._query_prefix = query_prefix
        self._fetch_n_multiplier = fetch_n_multiplier
        self._rrf_k = rrf_k
        self._reranker_enabled = reranker_enabled
        self._reranker_model_name = reranker_model
        self._reranker_pool_size = reranker_pool_size
        self._reranker: CrossEncoder | None = None

        data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(data_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._db = MetadataDB(data_dir / "metadata.db")

        self._model: SentenceTransformer | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []

        self._check_model_compatibility()

        self._docs = self._client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

        self.build_bm25()
        logger.info("VectorStore ready at %s", data_dir)

    def close(self) -> None:
        """
        Release resources held by the store.

        Closes the SQLite metadata DB and the Chroma client. Idempotent.
        Sentence-transformer models and the BM25 index are in memory and
        will be garbage-collected with the instance.
        """

        self._db.close()
        if hasattr(self, "_client"):
            self._client.close()  # type: ignore[attr-defined]
            # Chroma's shared system registry keeps SQLite connections alive until
            # explicitly cleared. Drop all our references to Chroma objects before
            # gc.collect() so the connections are actually freed.
            SharedSystemClient.clear_system_cache()
            if hasattr(self, "_docs"):
                del self._docs
            del self._client
            gc.collect()

    def __enter__(self) -> VectorStore:
        """Return self so with VectorStore(...) as store: yields the store."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Close the store on exit, even if the with block raised."""

        self.close()

    def _check_model_compatibility(self) -> None:
        """
        Wipe the index if the stored embedding model name has changed.

        Also handles first-run after the Tier 2 refactor: if the SQLite DB
        has no model recorded but Chroma already has a docs collection,
        that collection is orphaned stale data and gets dropped.
        """

        stored = self._db.get_model_name()
        if stored is None:
            self._drop_docs_collection()
        elif (
            stored != self._model_name
        ):  # pragma: no cover (exercised on manual upgrade)
            logger.info(
                "Embedding model changed from %s to %s. Clearing index.",
                stored,
                self._model_name,
            )
            self._drop_docs_collection()
            self._db.clear_indexed_state()

        self._db.set_model_name(self._model_name)

    def _drop_docs_collection(self) -> None:
        """Delete the Chroma docs collection if it exists."""

        try:
            self._client.delete_collection("docs")
        except NotFoundError:
            pass
        except ChromaError as exc:
            logger.warning("Could not delete docs collection: %s", exc)

    @property
    def _embed_model(self) -> SentenceTransformer:
        """Lazily load and cache the sentence-transformer embedding model."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def _reranker_model(self) -> CrossEncoder:
        """Lazily load and cache the cross-encoder reranker model."""

        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker

    def _delete_file_chunks(self, file_path: str) -> None:
        """
        Remove all docs (Chroma), parents, and hash (SQLite) for file_path.

        Raises:
            StoreError: If Chroma lookup or delete fails. The delete path is
                part of the upsert/delete contract. Silently continuing would
                produce duplicate or stale data in the index, which is worse
                than a visible failure.
        """

        try:
            results = self._docs.get(where={"source": file_path})
        except ChromaError as exc:
            raise StoreError(f"Lookup failed on docs for {file_path}") from exc

        if results["ids"]:
            try:
                self._docs.delete(ids=results["ids"])
            except ChromaError as exc:
                raise StoreError(f"Delete failed on docs for {file_path}") from exc

        self._db.delete_parents_by_source(file_path)
        self._db.delete_file_hash(file_path)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split text into word tokens for BM25."""

        return re.findall(r"\b\w+\b", text.lower())

    def build_bm25(self) -> None:
        """
        Rebuild the in-memory BM25 index from all child chunks.

        Called automatically after each upsert or delete. Also exposed
        publicly so the indexer can defer it until after a batch of upserts.
        """

        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        result = self._docs.get(include=["documents"])
        documents = result["documents"]
        if not result["ids"] or not documents:
            self._bm25 = None
            self._bm25_ids = []
            return
        self._bm25_ids = result["ids"]
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def upsert_file(
        self,
        file_path: str,
        chunks: list[dict[str, Any]],
        file_hash: str,
        rebuild_bm25: bool = True,
    ) -> None:
        """
        Index all chunks for a file, replacing any previously indexed content.

        Chunks are grouped by consecutive section into parent documents. Child
        chunks are embedded (BGE) and stored in Chroma. Parent sections are
        stored in SQLite for retrieval.

        Args:
            file_path: Source file path, used as a namespace for chunk IDs.
            chunks: List of dicts with text, section, chunk_index keys.
            file_hash: SHA-256 hash of the file, stored for change detection.
            rebuild_bm25: Rebuild the BM25 index after upsert. Set False when
                          batching files. Call build_bm25() manually at the
                          end.
        """

        self._delete_file_chunks(file_path)
        if not chunks:
            self._db.set_file_hash(file_path, file_hash)
            return

        groups: list[tuple[str, list[dict[str, Any]]]] = []
        for chunk in chunks:
            if not groups or groups[-1][0] != chunk["section"]:
                groups.append((chunk["section"], [chunk]))
            else:
                groups[-1][1].append(chunk)

        parent_rows: list[tuple[str, str, str, str]] = []
        child_ids: list[str] = []
        child_texts: list[str] = []
        child_metas: list[dict[str, Any]] = []

        for g_idx, (section, group) in enumerate(groups):
            pid = hashlib.md5(
                f"{file_path}::{section}::{g_idx}".encode(),
                usedforsecurity=False,
            ).hexdigest()

            parent_text = "\n\n".join(c["text"] for c in group)
            parent_rows.append((pid, file_path, section, parent_text))

            for c in group:
                child_ids.append(f"{file_path}::chunk::{c['chunk_index']}")
                child_texts.append(c["text"])
                child_metas.append(
                    {
                        "source": file_path,
                        "section": section,
                        "chunk_index": c["chunk_index"],
                        "parent_id": pid,
                    }
                )

        self._db.upsert_parents(parent_rows)

        embeddings = self._embed_model.encode(
            child_texts, show_progress_bar=False
        ).tolist()

        self._docs.upsert(
            ids=child_ids,
            embeddings=embeddings,
            documents=child_texts,
            metadatas=child_metas,  # type: ignore[arg-type]
        )

        self._db.set_file_hash(file_path, file_hash)

        logger.debug("Upserted %d chunks for %s", len(child_ids), file_path)

        if rebuild_bm25:
            self.build_bm25()

    def delete_file(self, file_path: str) -> None:
        """
        Remove all indexed content for a source file.

        Args:
            file_path: Source file path used when the file was indexed.
        """

        self._delete_file_chunks(file_path)
        self.build_bm25()

    def delete_directory(self, directory: Path) -> None:
        """
        Remove all indexed content for files whose path falls under directory.

        Rebuilds the BM25 index once after all deletions rather than once
        per file.

        Args:
            directory: Resolved absolute path to the directory being removed.
        """

        prefix = str(directory.resolve())
        indexed = self.get_indexed_sources()

        to_delete = [
            fpath
            for fpath in indexed
            if fpath == prefix or fpath.startswith(prefix + "/")
        ]

        for fpath in to_delete:
            self._delete_file_chunks(fpath)

        if to_delete:
            self.build_bm25()

    def get_indexed_sources(self) -> dict[str, str]:
        """
        Return a mapping of source file path to SHA-256 hash for all indexed files.

        Returns:
            Dict mapping file path strings to their stored SHA-256 hashes.
        """

        return self._db.all_file_hashes()

    def find_path_by_hash(self, file_hash: str, exclude_path: str) -> str | None:
        """
        Return another indexed path that has this hash, or None.

        Lets the indexer skip embedding the same content twice when the
        same file appears under multiple source directories.
        """

        return self._db.find_path_by_hash(file_hash, exclude_path)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Hybrid semantic + BM25 search with parent-child retrieval.

        Runs BGE semantic search (cosine) and BM25 keyword search in
        parallel, then fuses their ranked lists with Reciprocal Rank Fusion.
        Deduplicates results by parent section and returns full section
        text rather than individual chunk excerpts. When the reranker is
        enabled, the top reranker_pool_size deduplicated parents are
        re-scored by a cross-encoder before truncating to top_k.

        Args:
            query: The search query string.
            top_k: Number of distinct sections to return.

        Returns:
            List of SearchResult. When the reranker is enabled, the score
            field holds the cross-encoder logit. Otherwise it is the RRF
            score. In both cases results are sorted descending.
        """

        if self._docs.count() == 0:
            return []

        import numpy as np  # noqa: PLC0415

        dedup_target = (
            max(self._reranker_pool_size, top_k) if self._reranker_enabled else top_k
        )
        fetch_n = min(dedup_target * self._fetch_n_multiplier, self._docs.count())

        # Semantic search with BGE query prefix
        query_emb = self._embed_model.encode(
            [self._query_prefix + query], show_progress_bar=False
        ).tolist()

        sem = self._docs.query(
            query_embeddings=query_emb,
            n_results=fetch_n,
            include=["metadatas", "distances"],
        )

        sem_ids: list[str] = sem["ids"][0]
        sem_metas: list[dict[str, Any]] = sem["metadatas"][0]  # type: ignore[index, assignment]

        # BM25 keyword search
        bm25_rank: dict[str, int] = {}
        if self._bm25 is not None:
            query_tokens = self._tokenize(query)
            if query_tokens:
                scores = self._bm25.get_scores(query_tokens)
                n = len(scores)

                if fetch_n >= n:
                    top_indices = np.argsort(scores)[::-1]
                else:
                    # Partition for top fetch_n, then sort only those.
                    part = np.argpartition(scores, n - fetch_n)[n - fetch_n :]
                    top_indices = part[np.argsort(scores[part])[::-1]]

                for rank, idx in enumerate(top_indices):
                    bm25_rank[self._bm25_ids[idx]] = rank

        # Reciprocal Rank Fusion
        K = self._rrf_k
        rrf: dict[str, float] = {}
        id_to_meta: dict[str, dict[str, Any]] = {}

        for rank, (cid, meta) in enumerate(zip(sem_ids, sem_metas, strict=False)):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K + rank)
            id_to_meta[cid] = meta

        missing_ids: list[str] = []
        for cid, rank in bm25_rank.items():
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K + rank)
            if cid not in id_to_meta:
                missing_ids.append(cid)

        if missing_ids:
            rows = self._docs.get(ids=missing_ids, include=["metadatas"])
            id_to_meta.update(
                dict(zip(rows["ids"], rows["metadatas"] or [], strict=False))  # type: ignore[arg-type]
            )

        # Deduplicate by parent
        seen_parents: set[str] = set()
        selected: list[tuple[str, dict[str, Any], float]] = []

        for cid in sorted(rrf, key=rrf.__getitem__, reverse=True):
            if len(selected) >= dedup_target:
                break

            meta = id_to_meta.get(cid, {})
            pid = meta.get("parent_id", cid)
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            selected.append((pid, meta, round(rrf[cid], 4)))

        # Batch-fetch parent section text from SQLite
        parent_texts = self._db.get_parent_texts([pid for pid, _, _ in selected])

        if self._reranker_enabled and selected:
            pairs = [(query, parent_texts.get(pid, "")) for pid, _, _ in selected]

            scores = self._reranker_model.predict(pairs)  # type: ignore[arg-type]
            ranked = sorted(
                zip(scores, selected, strict=True),
                key=lambda pair: float(pair[0]),
                reverse=True,
            )

            selected = [
                (pid, meta, round(float(score), 4))
                for score, (pid, meta, _rrf) in ranked[:top_k]
            ]
        else:
            selected = selected[:top_k]

        return [
            SearchResult(
                filename=meta.get("source", ""),
                excerpt=parent_texts.get(pid, ""),
                section=meta.get("section", ""),
                score=score,
            )
            for pid, meta, score in selected
        ]
