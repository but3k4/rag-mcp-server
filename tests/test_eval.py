"""Tests for the retrieval evaluation harness in rag.eval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from rag.eval import EvalQuery, evaluate, load_dataset
from rag.store import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

_TWO_QUERIES = 2
_HALF = 0.5


def _write_dataset(path: Path, entries: list[dict[str, object]]) -> Path:
    """Write a JSON dataset and return the file path."""

    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _result(filename: str) -> SearchResult:
    """Build a SearchResult with the given filename and placeholder fields."""

    return SearchResult(filename=filename, excerpt="", section="", score=0.0)


class TestLoadDataset:
    """Tests for load_dataset."""

    def test_parses_valid_array(self, tmp_path: Path) -> None:
        """A well-formed JSON array becomes a list of EvalQuery records."""

        path = _write_dataset(
            tmp_path / "ds.json",
            [
                {"query": "q1", "relevant": ["/abs/a.md"]},
                {"query": "q2", "relevant": ["/abs/b.md", "/abs/c.md"]},
            ],
        )
        queries = load_dataset(path)
        assert len(queries) == _TWO_QUERIES
        assert queries[0] == EvalQuery(query="q1", relevant=frozenset({"/abs/a.md"}))
        assert queries[1].relevant == frozenset({"/abs/b.md", "/abs/c.md"})

    def test_resolves_relative_paths_against_dataset_dir(self, tmp_path: Path) -> None:
        """Relative paths in 'relevant' resolve against the dataset's parent."""

        sub = tmp_path / "ds_dir"
        sub.mkdir()
        path = _write_dataset(
            sub / "ds.json",
            [{"query": "q", "relevant": ["docs/a.md"]}],
        )
        expected = str((sub / "docs/a.md").resolve())
        queries = load_dataset(path)
        assert queries[0].relevant == frozenset({expected})

    def test_rejects_non_array_root(self, tmp_path: Path) -> None:
        """A JSON object at the root is rejected with a clear message."""

        path = tmp_path / "ds.json"
        path.write_text(json.dumps({"queries": []}), encoding="utf-8")
        with pytest.raises(TypeError, match="must be a JSON array"):
            load_dataset(path)

    def test_rejects_missing_keys(self, tmp_path: Path) -> None:
        """An entry missing 'query' or 'relevant' is rejected."""

        path = _write_dataset(tmp_path / "ds.json", [{"query": "q"}])
        with pytest.raises(ValueError, match="missing 'query' or 'relevant'"):
            load_dataset(path)

    def test_rejects_relevant_not_a_list(self, tmp_path: Path) -> None:
        """A non-list 'relevant' value is rejected."""

        path = _write_dataset(
            tmp_path / "ds.json",
            [{"query": "q", "relevant": "/abs/a.md"}],
        )
        with pytest.raises(TypeError, match="'relevant' must be a list"):
            load_dataset(path)

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        """A file that is not valid JSON raises JSONDecodeError."""

        path = tmp_path / "ds.json"
        path.write_text("not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_dataset(path)

    def test_empty_array_returns_no_queries(self, tmp_path: Path) -> None:
        """An empty JSON array produces an empty list."""

        path = _write_dataset(tmp_path / "ds.json", [])
        assert load_dataset(path) == []


class TestEvaluate:
    """Tests for evaluate."""

    def test_perfect_recall_and_mrr(self) -> None:
        """A relevant doc returned at rank 1 yields recall 1.0 and rr 1.0."""

        store = MagicMock()
        store.search.return_value = [_result("/a.md"), _result("/b.md")]
        queries = [EvalQuery(query="q", relevant=frozenset({"/a.md"}))]
        results = evaluate(store, queries, top_k=2)
        assert results.queries_evaluated == 1
        assert results.recall_at_k == 1.0
        assert results.mrr == 1.0

    def test_zero_recall_when_no_match(self) -> None:
        """If none of the retrieved docs are relevant, recall and mrr are 0."""

        store = MagicMock()
        store.search.return_value = [_result("/x.md")]
        queries = [EvalQuery(query="q", relevant=frozenset({"/a.md"}))]
        results = evaluate(store, queries, top_k=1)
        assert results.recall_at_k == 0.0
        assert results.mrr == 0.0

    def test_partial_recall_when_one_of_many_relevant(self) -> None:
        """One of two relevant docs retrieved yields recall 0.5."""

        store = MagicMock()
        store.search.return_value = [_result("/a.md"), _result("/x.md")]
        queries = [
            EvalQuery(query="q", relevant=frozenset({"/a.md", "/b.md"})),
        ]
        results = evaluate(store, queries, top_k=2)
        assert results.recall_at_k == _HALF

    def test_mrr_uses_first_relevant_rank(self) -> None:
        """The first relevant doc at rank 3 produces rr = 1/3."""

        store = MagicMock()
        store.search.return_value = [
            _result("/x.md"),
            _result("/y.md"),
            _result("/a.md"),
        ]
        queries = [EvalQuery(query="q", relevant=frozenset({"/a.md"}))]
        results = evaluate(store, queries, top_k=3)
        assert results.mrr == pytest.approx(1 / 3, abs=1e-4)

    def test_dedupes_repeated_filenames_by_rank(self) -> None:
        """A file appearing twice in raw results contributes one rank slot."""

        store = MagicMock()
        store.search.return_value = [
            _result("/x.md"),
            _result("/a.md"),
            _result("/a.md"),
        ]
        queries = [EvalQuery(query="q", relevant=frozenset({"/a.md"}))]
        results = evaluate(store, queries, top_k=3)
        # /a.md is at rank 2 in the deduped list, not rank 3.
        assert results.mrr == pytest.approx(0.5, abs=1e-4)

    def test_skips_queries_with_no_relevant(self) -> None:
        """Queries with empty 'relevant' are excluded from aggregates."""

        store = MagicMock()
        store.search.return_value = [_result("/a.md")]
        queries = [
            EvalQuery(query="q1", relevant=frozenset()),
            EvalQuery(query="q2", relevant=frozenset({"/a.md"})),
        ]
        results = evaluate(store, queries, top_k=1)
        assert results.queries_evaluated == 1
        assert results.recall_at_k == 1.0
        assert any(p.get("skipped") for p in results.per_query)

    def test_empty_query_list_returns_zero_metrics(self) -> None:
        """An empty queries iterable produces a zero-everything result."""

        store = MagicMock()
        results = evaluate(store, [], top_k=5)
        assert results.queries_evaluated == 0
        assert results.recall_at_k == 0.0
        assert results.mrr == 0.0
        assert results.per_query == []

    def test_per_query_breakdown_present(self) -> None:
        """per_query records the retrieved files and metrics for each query."""

        store = MagicMock()
        store.search.return_value = [_result("/a.md"), _result("/b.md")]
        queries = [EvalQuery(query="q", relevant=frozenset({"/b.md"}))]
        results = evaluate(store, queries, top_k=2)
        assert len(results.per_query) == 1
        entry = results.per_query[0]
        assert entry["query"] == "q"
        assert entry["retrieved"] == ["/a.md", "/b.md"]
        assert entry["recall"] == 1.0
        assert entry["rr"] == _HALF

    def test_passes_top_k_to_store_search(self) -> None:
        """Forwards top_k as a keyword arg to store.search."""

        store = MagicMock()
        store.search.return_value = []
        queries = [EvalQuery(query="q", relevant=frozenset({"/a.md"}))]
        evaluate(store, queries, top_k=7)
        store.search.assert_called_once_with("q", top_k=7)
