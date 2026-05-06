"""Retrieval evaluation harness.

Loads a labeled dataset of queries with their relevant document paths,
runs each query against an indexed VectorStore, and reports recall@k
plus mean reciprocal rank. Use it to compare config changes (chunk size,
embedder, reranker on/off) on a representative corpus.

Dataset format (JSON):

    [
      {"query": "...", "relevant": ["path/to/doc.md", ...]},
      ...
    ]

Relevant paths may be absolute or relative to the dataset file's parent.

For an A/B comparison, run the harness twice with different configs (most
commonly different RAG_DATA_DIR values pointing at separately indexed
corpora) and diff the JSON outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from rag.config import ConfigError, load_config
from rag.logging_config import configure_logging
from rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EvalQuery:
    """One labeled evaluation query.

    Attributes:
        query: The natural-language query string.
        relevant: Set of file paths considered correct hits. A retrieved
                  result counts as a hit if its filename is in this set.
    """

    query: str
    relevant: frozenset[str]


@dataclass(frozen=True)
class EvalResults:
    """Aggregate evaluation metrics plus per-query breakdown.

    Attributes:
        queries_evaluated: Count of queries that contributed to the
                           aggregate metrics. Queries with no labeled
                           relevant docs are excluded.
        top_k: The k passed to store.search.
        recall_at_k: Mean fraction of relevant docs retrieved in the top k.
        mrr: Mean reciprocal rank of the first relevant doc.
        per_query: One dict per query for inspection. Includes the query,
                   relevant set, retrieved files, recall, and rr.
    """

    queries_evaluated: int
    top_k: int
    recall_at_k: float
    mrr: float
    per_query: list[dict[str, Any]] = field(default_factory=list)


def load_dataset(path: Path) -> list[EvalQuery]:
    """Load eval queries from a JSON file.

    Format: a JSON array where each item is a dict with 'query' (str) and
    'relevant' (list[str]) keys. Relative paths in 'relevant' are resolved
    against the dataset file's parent directory.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of EvalQuery records.

    Raises:
        TypeError: If a structural element has the wrong type.
        ValueError: If a required key is missing.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"Dataset must be a JSON array, got {type(raw).__name__}")

    base = path.parent
    queries: list[EvalQuery] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise TypeError(f"Entry {i} must be an object, got {type(entry).__name__}")
        if "query" not in entry or "relevant" not in entry:
            raise ValueError(f"Entry {i} missing 'query' or 'relevant' key")

        relevant = entry["relevant"]
        if not isinstance(relevant, list):
            raise TypeError(
                f"Entry {i} 'relevant' must be a list, got {type(relevant).__name__}"
            )
        queries.append(
            EvalQuery(
                query=str(entry["query"]),
                relevant=frozenset(_resolve(p, base) for p in relevant),
            )
        )
    return queries


def _resolve(p: str, base: Path) -> str:
    """Resolve a relative path against base. Absolute paths pass through."""

    path = Path(p)
    if not path.is_absolute():
        path = (base / p).resolve()
    return str(path)


def _unique_filenames(results: list[Any]) -> list[str]:
    """Return result filenames in rank order, with duplicates removed."""

    seen: set[str] = set()
    unique: list[str] = []
    for r in results:
        if r.filename not in seen:
            seen.add(r.filename)
            unique.append(r.filename)
    return unique


def evaluate(
    store: VectorStore, queries: Iterable[EvalQuery], top_k: int
) -> EvalResults:
    """Run each query against store and compute recall@k plus MRR.

    Queries with no labeled relevant docs are excluded from the aggregate
    metrics and logged at warning. They appear in per_query with a
    "skipped" marker for transparency.

    Args:
        store: An indexed VectorStore.
        queries: Labeled evaluation queries.
        top_k: How many results to fetch per query.

    Returns:
        EvalResults with aggregate metrics and per-query breakdown.
    """

    per_query: list[dict[str, Any]] = []
    recalls: list[float] = []
    rrs: list[float] = []

    for q in queries:
        if not q.relevant:
            logger.warning(
                "Query has no labeled relevant docs, excluding from metrics: %r",
                q.query,
            )
            per_query.append(
                {
                    "query": q.query,
                    "skipped": True,
                    "reason": "no labeled relevant docs",
                }
            )
            continue

        results = store.search(q.query, top_k=top_k)
        retrieved = _unique_filenames(results)
        retrieved_set = set(retrieved)

        hits = sum(1 for f in q.relevant if f in retrieved_set)
        recall = hits / len(q.relevant)

        rr = 0.0
        for rank, f in enumerate(retrieved, start=1):
            if f in q.relevant:
                rr = 1.0 / rank
                break

        recalls.append(recall)
        rrs.append(rr)
        per_query.append(
            {
                "query": q.query,
                "relevant": sorted(q.relevant),
                "retrieved": retrieved,
                "recall": round(recall, 4),
                "rr": round(rr, 4),
            }
        )

    n = len(recalls)
    return EvalResults(
        queries_evaluated=n,
        top_k=top_k,
        recall_at_k=round(sum(recalls) / n, 4) if n else 0.0,
        mrr=round(sum(rrs) / n, 4) if n else 0.0,
        per_query=per_query,
    )


def main() -> None:
    """Entry point for the rag-eval CLI.

    Reads cfg.data_dir / cfg.embedder_model from the environment and
    config.toml the same way the server does, opens the existing
    VectorStore (no indexing), runs every query in the dataset, and
    writes a JSON metrics summary to stdout. Exits non-zero on
    configuration or dataset errors.
    """

    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Run retrieval evaluation against an indexed VectorStore.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the JSON eval dataset file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override cfg.top_k for evaluation.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Include per-query results in the JSON output.",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(PROJECT_ROOT)
    except ConfigError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.exit(1)

    configure_logging(level=cfg.log_level, log_format=cfg.log_format)

    try:
        queries = load_dataset(args.dataset)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"Dataset error: {exc}\n")
        sys.exit(1)

    top_k = args.top_k if args.top_k is not None else cfg.top_k

    with VectorStore(
        cfg.data_dir,
        model_name=cfg.embedder_model,
        query_prefix=cfg.query_prefix,
        fetch_n_multiplier=cfg.fetch_n_multiplier,
        rrf_k=cfg.rrf_k,
        reranker_enabled=cfg.reranker_enabled,
        reranker_model=cfg.reranker_model,
        reranker_pool_size=cfg.reranker_pool_size,
    ) as store:
        results = evaluate(store, queries, top_k=top_k)

    output: dict[str, Any] = {
        "queries_evaluated": results.queries_evaluated,
        "top_k": results.top_k,
        "recall_at_k": results.recall_at_k,
        "mrr": results.mrr,
    }
    if args.per_query:
        output["per_query"] = results.per_query

    sys.stdout.write(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
