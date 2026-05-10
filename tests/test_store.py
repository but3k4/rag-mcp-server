"""Tests for rag.store.VectorStore using a real ephemeral ChromaDB instance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from chromadb.errors import InternalError
import pytest

from rag.errors import StoreError
from rag.store import SearchResult, VectorStore

TOP_K = 3

_CHUNKS = [
    {"text": "chunk one", "section": "", "chunk_index": 0},
    {"text": "chunk two", "section": "", "chunk_index": 1},
]


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> VectorStore:
    """Create a VectorStore backed by a temp directory, shared across the module."""

    data_dir = tmp_path_factory.mktemp("rag_data")
    return VectorStore(data_dir)


class TestVectorStoreInit:
    """Tests for VectorStore initialisation."""

    def test_creates_data_dir(self, tmp_path: Path) -> None:
        """VectorStore creates its data directory if it does not exist."""

        data_dir = tmp_path / "new_subdir"
        assert not data_dir.exists()
        VectorStore(data_dir)
        assert data_dir.is_dir()


class TestUpsertAndRetrieve:
    """Tests for upsert_file and get_indexed_sources."""

    def test_upsert_and_retrieve_source(self, store: VectorStore) -> None:
        """An upserted file appears in get_indexed_sources with its hash."""

        store.upsert_file("/tmp/a.txt", _CHUNKS, "abc123")
        sources = store.get_indexed_sources()
        assert "/tmp/a.txt" in sources
        assert sources["/tmp/a.txt"] == "abc123"

    def test_upsert_replaces_old_chunks(self, store: VectorStore) -> None:
        """Re-upserting a file stores the new hash."""

        store.upsert_file("/tmp/b.txt", _CHUNKS, "hash_old")
        store.upsert_file("/tmp/b.txt", _CHUNKS, "hash_new")
        sources = store.get_indexed_sources()
        assert sources["/tmp/b.txt"] == "hash_new"

    def test_empty_chunks_still_records_hash(self, store: VectorStore) -> None:
        """
        Empty chunks clear docs/parents but still record the hash.

        Without this, files that yield no extractable text (e.g. scanned-only
        PDFs) would be re-processed on every startup.
        """

        store.upsert_file("/tmp/c.txt", _CHUNKS, "initial")
        store.upsert_file("/tmp/c.txt", [], "empty")
        sources = store.get_indexed_sources()
        assert sources.get("/tmp/c.txt") == "empty"

    def test_multiple_sources_all_returned(self, store: VectorStore) -> None:
        """get_indexed_sources returns all distinct source files."""

        store.upsert_file("/tmp/x.txt", _CHUNKS, "hx")
        store.upsert_file("/tmp/y.txt", _CHUNKS, "hy")
        sources = store.get_indexed_sources()
        assert "/tmp/x.txt" in sources
        assert "/tmp/y.txt" in sources


class TestDeleteFile:
    """Tests for delete_file."""

    def test_delete_removes_source(self, tmp_path: Path) -> None:
        """delete_file removes all chunks for the specified file."""

        vs = VectorStore(tmp_path / "del_store")
        vs.upsert_file("/tmp/del.txt", _CHUNKS, "h1")
        assert "/tmp/del.txt" in vs.get_indexed_sources()
        vs.delete_file("/tmp/del.txt")
        assert "/tmp/del.txt" not in vs.get_indexed_sources()

    def test_delete_nonexistent_does_not_raise(self, tmp_path: Path) -> None:
        """Deleting a file that was never indexed does not raise."""

        vs = VectorStore(tmp_path / "del_store2")
        vs.delete_file("/tmp/never_indexed.txt")  # must not raise


class TestDeleteDirectory:
    """Tests for delete_directory."""

    def test_delete_directory_removes_all_files_under_it(self, tmp_path: Path) -> None:
        """delete_directory removes every indexed file whose path starts with the directory."""

        vs = VectorStore(tmp_path / "dir_store")
        vs.upsert_file("/data/project/a.txt", _CHUNKS, "ha")
        vs.upsert_file("/data/project/b.txt", _CHUNKS, "hb")
        vs.upsert_file("/data/other/c.txt", _CHUNKS, "hc")
        vs.delete_directory(Path("/data/project"))
        sources = vs.get_indexed_sources()
        assert "/data/project/a.txt" not in sources
        assert "/data/project/b.txt" not in sources
        assert "/data/other/c.txt" in sources

    def test_delete_directory_empty_is_noop(self, tmp_path: Path) -> None:
        """delete_directory on a directory with no indexed files does not raise."""

        vs = VectorStore(tmp_path / "dir_store2")
        vs.delete_directory(Path("/no/files/here"))  # must not raise


class TestSearch:
    """Tests for hybrid search."""

    def test_search_returns_list(self, store: VectorStore) -> None:
        """search() returns a list."""

        store.upsert_file(
            "/tmp/search.txt",
            [{"text": "The quick brown fox", "section": "", "chunk_index": 0}],
            "hfox",
        )
        results = store.search("fox", top_k=1)
        assert isinstance(results, list)

    def test_search_result_fields(self, store: VectorStore) -> None:
        """Each search result has filename, excerpt, section, and score."""

        store.upsert_file(
            "/tmp/fields.txt",
            [
                {
                    "text": "Python is a programming language",
                    "section": "intro",
                    "chunk_index": 0,
                }
            ],
            "hpy",
        )
        results = store.search("programming", top_k=1)
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert isinstance(r.filename, str)
        assert isinstance(r.excerpt, str)
        assert isinstance(r.section, str)
        assert isinstance(r.score, float)

    def test_search_empty_store_returns_empty(self, tmp_path: Path) -> None:
        """Searching an empty store returns an empty list."""

        vs = VectorStore(tmp_path / "empty_store")
        results = vs.search("anything", top_k=5)
        assert results == []

    def test_search_top_k_limits_results(self, store: VectorStore) -> None:
        """search() returns at most top_k results."""

        for i in range(10):
            store.upsert_file(
                f"/tmp/topk_{i}.txt",
                [{"text": f"document number {i}", "section": "", "chunk_index": 0}],
                f"h{i}",
            )
        results = store.search("document", top_k=TOP_K)
        assert len(results) <= TOP_K

    def test_search_dedupes_by_parent_section(self, tmp_path: Path) -> None:
        """Multiple child chunks sharing a parent section return a single result."""

        vs = VectorStore(tmp_path / "dedup_store")
        vs.upsert_file(
            "/tmp/dedup.txt",
            [
                {
                    "text": "alpha chunk about rabbits",
                    "section": "Intro",
                    "chunk_index": 0,
                },
                {
                    "text": "beta chunk about rabbits",
                    "section": "Intro",
                    "chunk_index": 1,
                },
            ],
            "hdup",
        )
        results = vs.search("rabbits", top_k=5)
        sources = [r.filename for r in results]
        assert sources.count("/tmp/dedup.txt") == 1


class TestReranker:
    """Tests for the optional cross-encoder reranker stage."""

    @staticmethod
    def _build_store_with_fake_reranker(
        tmp_path: Path, scores: list[float], pool_size: int = 20
    ) -> VectorStore:
        """Build a VectorStore with the reranker enabled and a mock CrossEncoder."""

        vs = VectorStore(
            tmp_path / "rerank_store",
            reranker_enabled=True,
            reranker_pool_size=pool_size,
        )
        fake = MagicMock(name="cross_encoder")
        fake.predict.return_value = scores
        vs._reranker = fake  # skip lazy-load. Inject fake directly
        return vs

    def test_disabled_preserves_rrf_ordering(self, tmp_path: Path) -> None:
        """With reranker disabled, results match the prior RRF-based path."""

        vs = VectorStore(tmp_path / "no_rerank")
        vs.upsert_file(
            "/tmp/a.txt",
            [{"text": "alpha", "section": "S1", "chunk_index": 0}],
            "ha",
        )
        vs.upsert_file(
            "/tmp/b.txt",
            [{"text": "beta", "section": "S2", "chunk_index": 0}],
            "hb",
        )
        results = vs.search("alpha", top_k=2)
        assert [r.filename for r in results] == ["/tmp/a.txt", "/tmp/b.txt"]

    def test_enabled_reorders_by_cross_encoder_scores(self, tmp_path: Path) -> None:
        """The reranker's scores determine the returned order."""

        docs = [
            (
                "/tmp/a.txt",
                [{"text": "alpha document", "section": "S1", "chunk_index": 0}],
                "ha",
            ),
            (
                "/tmp/b.txt",
                [{"text": "alpha content", "section": "S2", "chunk_index": 0}],
                "hb",
            ),
        ]

        # Establish the RRF-only baseline order for this corpus.
        baseline = VectorStore(tmp_path / "baseline")
        for path, chunks, h in docs:
            baseline.upsert_file(path, chunks, h)
        baseline_order = [r.filename for r in baseline.search("alpha", top_k=2)]

        # Now run with the reranker, giving the last RRF candidate the top score.
        # The reranker should invert the baseline order.
        vs = self._build_store_with_fake_reranker(tmp_path, scores=[])
        for path, chunks, h in docs:
            vs.upsert_file(path, chunks, h)
        vs._reranker.predict.return_value = [0.1, 0.9]  # type: ignore[union-attr]
        rerank_order = [r.filename for r in vs.search("alpha", top_k=2)]

        assert rerank_order == list(reversed(baseline_order))

    def test_enabled_truncates_to_top_k(self, tmp_path: Path) -> None:
        """Results returned to caller are capped at top_k even with a larger pool."""

        vs = self._build_store_with_fake_reranker(
            tmp_path, scores=[0.5, 0.4, 0.3], pool_size=5
        )
        for i in range(3):
            vs.upsert_file(
                f"/tmp/file_{i}.txt",
                [{"text": "alpha", "section": f"S{i}", "chunk_index": 0}],
                f"h{i}",
            )
        vs._reranker.predict.return_value = [0.5, 0.4, 0.3]  # type: ignore[union-attr]
        requested_top_k = 2
        results = vs.search("alpha", top_k=requested_top_k)
        assert len(results) == requested_top_k

    def test_enabled_calls_predict_with_query_and_parent_text(
        self, tmp_path: Path
    ) -> None:
        """Cross-encoder receives (query, parent_text) pairs."""

        vs = self._build_store_with_fake_reranker(tmp_path, scores=[0.5])
        vs.upsert_file(
            "/tmp/only.txt",
            [
                {
                    "text": "the actual chunk text",
                    "section": "S",
                    "chunk_index": 0,
                }
            ],
            "h",
        )
        vs._reranker.predict.return_value = [0.5]  # type: ignore[union-attr]
        vs.search("my query", top_k=1)
        vs._reranker.predict.assert_called_once()  # type: ignore[union-attr]
        pairs = vs._reranker.predict.call_args.args[0]  # type: ignore[union-attr]
        assert len(pairs) == 1
        assert pairs[0][0] == "my query"
        assert "the actual chunk text" in pairs[0][1]


class TestClose:
    """Tests for VectorStore.close()."""

    def test_close_closes_metadata_db(self, tmp_path: Path) -> None:
        """VectorStore.close delegates to the metadata DB."""

        vs = VectorStore(tmp_path / "close_store")
        vs.close()
        # A second close on the DB is a no-op. A fresh instance on the
        # same path still works, proving the files were flushed cleanly.
        vs2 = VectorStore(tmp_path / "close_store")
        vs2.close()

    def test_double_close_is_safe(self, tmp_path: Path) -> None:
        """Calling close twice does not raise."""

        vs = VectorStore(tmp_path / "double_close")
        vs.close()
        vs.close()  # must not raise

    def test_context_manager_closes_on_exit(self, tmp_path: Path) -> None:
        """With VectorStore(...) closes the metadata DB on block exit."""

        data_dir = tmp_path / "cm_store"
        with VectorStore(data_dir) as vs:
            vs.upsert_file("/tmp/cm.txt", _CHUNKS, "hcm")
        # After exit, opening a new store on the same path must succeed,
        # which would fail if the previous connection held a WAL lock.
        with VectorStore(data_dir) as vs2:
            assert "/tmp/cm.txt" in vs2.get_indexed_sources()

    def test_context_manager_closes_on_exception(self, tmp_path: Path) -> None:
        """Store closes even when the with block raises."""

        data_dir = tmp_path / "cm_raise"
        with pytest.raises(RuntimeError), VectorStore(data_dir):
            raise RuntimeError("simulated failure")
        # Re-opening proves the first connection released its lock.
        with VectorStore(data_dir):
            pass


class TestChromaErrorHandling:
    """
    Tests for typed propagation of ChromaDB errors.

    A silent-continue on store failure would produce duplicate or stale
    data in the index. These tests pin the contract: any ChromaError
    surfaces as StoreError, chained to the underlying cause, so the
    indexer's batch loop can count it as failed and move on.
    """

    def test_delete_file_raises_store_error_on_get(self, tmp_path: Path) -> None:
        """A ChromaError during lookup surfaces as StoreError."""

        vs = VectorStore(tmp_path / "err_get")
        vs.upsert_file("/tmp/e.txt", _CHUNKS, "h1")
        vs.build_bm25 = MagicMock()  # type: ignore[method-assign]
        docs_mock = MagicMock(name="docs")
        docs_mock.get.side_effect = InternalError("boom")
        vs._docs = docs_mock

        with pytest.raises(StoreError) as exc_info:
            vs.delete_file("/tmp/e.txt")
        assert isinstance(exc_info.value.__cause__, InternalError)

    def test_delete_file_raises_store_error_on_delete(self, tmp_path: Path) -> None:
        """A ChromaError during delete surfaces as StoreError."""

        vs = VectorStore(tmp_path / "err_del")
        vs.upsert_file("/tmp/e.txt", _CHUNKS, "h1")
        vs.build_bm25 = MagicMock()  # type: ignore[method-assign]
        docs_mock = MagicMock(name="docs")
        docs_mock.get.return_value = {"ids": ["id1"]}
        docs_mock.delete.side_effect = InternalError("delete failed")
        vs._docs = docs_mock

        with pytest.raises(StoreError) as exc_info:
            vs.delete_file("/tmp/e.txt")
        assert isinstance(exc_info.value.__cause__, InternalError)
