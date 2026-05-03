"""
Typed configuration for the RAG server.

Settings are loaded from three sources in decreasing precedence:
    1. Environment variables (e.g. TRANSPORT, PORT, RAG_CHUNK_SIZE).
    2. The TOML file at RAG_CONFIG (default: <project>/config.toml).
    3. Hard-coded defaults in this module.

All validation happens at load time. Invalid values raise ConfigError
before the rest of the server initialises, so misconfiguration surfaces
with a readable message instead of a traceback from deep inside a
downstream component.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING, Literal

from rag.errors import RagError

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Transport = Literal["stdio", "sse", "streamable-http"]
LogFormat = Literal["json", "text"]

_VALID_TRANSPORTS: tuple[Transport, ...] = ("stdio", "sse", "streamable-http")
_VALID_WATCHERS = ("native", "poll")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_VALID_LOG_FORMATS: tuple[LogFormat, ...] = ("json", "text")

DEFAULT_PORT = 8765
DEFAULT_CHUNK_SIZE = 1400
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
DEFAULT_FETCH_N_MULTIPLIER = 4
DEFAULT_RRF_K = 60
DEFAULT_EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_QUERY_PREFIX = ""
DEFAULT_RERANKER_ENABLED = False
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# If you prefer BAAI/BGE models, override all three together via env vars or config.toml.
# The query prefix is part of the BGE embedding convention and must change with the model:
#   RAG_EMBEDDER_MODEL  = "BAAI/bge-base-en-v1.5"
#   RAG_QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "
#   RAG_RERANKER_MODEL  = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_POOL_SIZE = 20
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MiB
DEFAULT_MAX_QUERY_LENGTH = 1000
DEFAULT_WATCHER_DEBOUNCE_SECONDS = 0.5
DEFAULT_LOG_FORMAT: LogFormat = "json"
DEFAULT_HEALTH_PORT = 0  # disabled


class ConfigError(RagError, ValueError):
    """Raised when the environment + config file cannot produce a valid RagConfig."""


@dataclass(frozen=True)
class RagConfig:
    """Validated configuration for the RAG server."""

    transport: Transport
    port: int
    data_dir: Path
    config_path: Path
    source_dirs: tuple[Path, ...]
    chunk_size: int
    chunk_overlap: int
    use_polling: bool
    log_level: str
    top_k: int
    fetch_n_multiplier: int
    rrf_k: int
    embedder_model: str
    query_prefix: str
    reranker_enabled: bool
    reranker_model: str
    reranker_pool_size: int
    allowed_base_dirs: tuple[Path, ...]
    max_file_size_bytes: int
    max_query_length: int
    watcher_debounce_seconds: float
    log_format: LogFormat
    health_port: int

    @classmethod
    def from_sources(
        cls,
        env: Mapping[str, str],
        toml_data: Mapping[str, object],
        project_root: Path,
    ) -> RagConfig:
        """
        Build a RagConfig from parsed environment + TOML inputs.

        Keeping this pure (no file or os.environ access) makes the loader
        fully testable.

        Args:
            env: Mapping to read environment variables from.
            toml_data: Parsed config.toml contents (may be empty).
            project_root: Used to resolve default data_dir and config_path.

        Raises:
            ConfigError: If any value fails validation.
        """

        indexing = _indexing_section(toml_data)
        retrieval = _retrieval_section(toml_data)
        security = _security_section(toml_data)

        transport = _parse_transport(env.get("TRANSPORT", "stdio"))
        port = _parse_int(
            env.get("PORT"),
            default=DEFAULT_PORT,
            name="PORT",
            minimum=1,
            maximum=65535,
        )

        data_dir = _resolve_path(
            env.get("RAG_DATA_DIR", str(project_root / ".rag-data"))
        )

        config_path = _resolve_path(
            env.get("RAG_CONFIG", str(project_root / "config.toml"))
        )

        source_dirs = _parse_source_dirs(
            env.get("RAG_SOURCE_DIRS"),
            indexing.get("source_dirs"),
        )

        chunk_size = _parse_int(
            env.get("RAG_CHUNK_SIZE"),
            default=_as_int(indexing.get("chunk_size"), DEFAULT_CHUNK_SIZE),
            name="chunk_size",
            minimum=1,
        )

        chunk_overlap = _parse_int(
            env.get("RAG_CHUNK_OVERLAP"),
            default=_as_int(indexing.get("chunk_overlap"), DEFAULT_CHUNK_OVERLAP),
            name="chunk_overlap",
            minimum=0,
        )

        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )

        use_polling = _parse_watcher(env.get("RAG_WATCHER", "native"))
        log_level = _parse_log_level(env.get("LOG_LEVEL", "INFO"))

        top_k = _parse_int(
            env.get("RAG_TOP_K"),
            default=_as_int(retrieval.get("top_k"), DEFAULT_TOP_K),
            name="top_k",
            minimum=1,
        )

        fetch_n_multiplier = _parse_int(
            env.get("RAG_FETCH_N_MULTIPLIER"),
            default=_as_int(
                retrieval.get("fetch_n_multiplier"), DEFAULT_FETCH_N_MULTIPLIER
            ),
            name="fetch_n_multiplier",
            minimum=1,
        )

        rrf_k = _parse_int(
            env.get("RAG_RRF_K"),
            default=_as_int(retrieval.get("rrf_k"), DEFAULT_RRF_K),
            name="rrf_k",
            minimum=1,
        )

        embedder_model = _parse_str(
            env.get("RAG_EMBEDDER_MODEL"),
            default=_as_str(retrieval.get("embedder_model"), DEFAULT_EMBEDDER_MODEL),
            name="embedder_model",
        )

        query_prefix = env.get("RAG_QUERY_PREFIX") or _as_str(
            retrieval.get("query_prefix"), DEFAULT_QUERY_PREFIX
        )

        reranker_enabled = _parse_bool(
            env.get("RAG_RERANKER_ENABLED"),
            default=_as_bool(
                retrieval.get("reranker_enabled"), DEFAULT_RERANKER_ENABLED
            ),
            name="reranker_enabled",
        )

        reranker_model = _parse_str(
            env.get("RAG_RERANKER_MODEL"),
            default=_as_str(retrieval.get("reranker_model"), DEFAULT_RERANKER_MODEL),
            name="reranker_model",
        )

        reranker_pool_size = _parse_int(
            env.get("RAG_RERANKER_POOL_SIZE"),
            default=_as_int(
                retrieval.get("reranker_pool_size"), DEFAULT_RERANKER_POOL_SIZE
            ),
            name="reranker_pool_size",
            minimum=1,
        )

        if reranker_enabled and reranker_pool_size < top_k:
            raise ConfigError(
                f"reranker_pool_size ({reranker_pool_size}) must be "
                f">= top_k ({top_k}) when reranker is enabled"
            )

        allowed_base_dirs = _parse_path_list(
            env.get("RAG_ALLOWED_BASE_DIRS"),
            security.get("allowed_base_dirs"),
            field_name="allowed_base_dirs",
        )

        max_file_size_bytes = _parse_int(
            env.get("RAG_MAX_FILE_SIZE_BYTES"),
            default=_as_int(
                indexing.get("max_file_size_bytes"), DEFAULT_MAX_FILE_SIZE_BYTES
            ),
            name="max_file_size_bytes",
            minimum=1,
        )

        max_query_length = _parse_int(
            env.get("RAG_MAX_QUERY_LENGTH"),
            default=_as_int(
                retrieval.get("max_query_length"), DEFAULT_MAX_QUERY_LENGTH
            ),
            name="max_query_length",
            minimum=1,
        )

        watcher_debounce_seconds = _parse_float(
            env.get("RAG_WATCHER_DEBOUNCE_SECONDS"),
            default=_as_float(
                indexing.get("watcher_debounce_seconds"),
                DEFAULT_WATCHER_DEBOUNCE_SECONDS,
            ),
            name="watcher_debounce_seconds",
            minimum=0.0,
        )

        log_format = _parse_log_format(env.get("RAG_LOG_FORMAT", DEFAULT_LOG_FORMAT))
        health_port = _parse_int(
            env.get("RAG_HEALTH_PORT"),
            default=DEFAULT_HEALTH_PORT,
            name="health_port",
            minimum=0,
            maximum=65535,
        )

        return cls(
            transport=transport,
            port=port,
            data_dir=data_dir,
            config_path=config_path,
            source_dirs=source_dirs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_polling=use_polling,
            log_level=log_level,
            top_k=top_k,
            fetch_n_multiplier=fetch_n_multiplier,
            rrf_k=rrf_k,
            embedder_model=embedder_model,
            query_prefix=query_prefix,
            reranker_enabled=reranker_enabled,
            reranker_model=reranker_model,
            reranker_pool_size=reranker_pool_size,
            allowed_base_dirs=allowed_base_dirs,
            max_file_size_bytes=max_file_size_bytes,
            max_query_length=max_query_length,
            watcher_debounce_seconds=watcher_debounce_seconds,
            log_format=log_format,
            health_port=health_port,
        )


def load_config(
    project_root: Path,
    env: Mapping[str, str] | None = None,
) -> RagConfig:
    """
    Read environment + config.toml from disk and return a validated RagConfig.

    Args:
        project_root: Used to resolve defaults for RAG_DATA_DIR and RAG_CONFIG.
        env: Override the environment for testing. Defaults to os.environ.

    Raises:
        ConfigError: If the file is unreadable or any value fails validation.
    """

    env = os.environ if env is None else env
    config_path = _resolve_path(
        env.get("RAG_CONFIG", str(project_root / "config.toml"))
    )
    toml_data = _load_toml(config_path)
    return RagConfig.from_sources(env, toml_data, project_root)


def _load_toml(path: Path) -> dict[str, object]:
    """Load a TOML file. Missing file is treated as an empty config."""

    if not path.is_file():
        logger.info("No config.toml at %s. Relying on environment variables", path)
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def _indexing_section(toml_data: Mapping[str, object]) -> Mapping[str, object]:
    """Return the [indexing] table, or an empty mapping if absent or malformed."""

    indexing = toml_data.get("indexing", {})
    if not isinstance(indexing, dict):
        raise ConfigError(
            f"config.toml [indexing] must be a table. Got {type(indexing).__name__}"
        )
    return indexing


def _retrieval_section(toml_data: Mapping[str, object]) -> Mapping[str, object]:
    """Return the [retrieval] table, or an empty mapping if absent or malformed."""

    retrieval = toml_data.get("retrieval", {})
    if not isinstance(retrieval, dict):
        raise ConfigError(
            f"config.toml [retrieval] must be a table. Got {type(retrieval).__name__}"
        )
    return retrieval


def _security_section(toml_data: Mapping[str, object]) -> Mapping[str, object]:
    """Return the [security] table, or an empty mapping if absent or malformed."""

    security = toml_data.get("security", {})
    if not isinstance(security, dict):
        raise ConfigError(
            f"config.toml [security] must be a table. Got {type(security).__name__}"
        )
    return security


def _resolve_path(value: str) -> Path:
    """Expand ~ and resolve symlinks to return an absolute Path."""

    return Path(value).expanduser().resolve()


def _parse_transport(raw: str) -> Transport:
    if raw not in _VALID_TRANSPORTS:
        raise ConfigError(
            f"Invalid TRANSPORT {raw!r}. "
            f"Must be one of: {', '.join(_VALID_TRANSPORTS)}."
        )
    return raw


def _parse_int(
    raw: str | None,
    default: int,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce an optional string value to int, with range validation."""

    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer. Got {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}. Got {value}")

    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}. Got {value}")

    return value


def _as_int(value: object, default: int) -> int:
    """Coerce a TOML value to int, falling back to default if absent."""

    if value is None:
        return default

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"Expected integer in config.toml. Got {value!r} ({type(value).__name__})"
        )
    return value


def _parse_float(
    raw: str | None,
    default: float,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce an optional string value to float, with range validation."""

    if raw is None or raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number. Got {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}. Got {value}")

    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}. Got {value}")

    return value


def _as_float(value: object, default: float) -> float:
    """Coerce a TOML value to float, falling back to default if absent."""

    if value is None:
        return default

    if isinstance(value, bool):
        raise ConfigError(f"Expected number in config.toml. Got {value!r} (bool)")

    if isinstance(value, (int, float)):
        return float(value)

    raise ConfigError(
        f"Expected number in config.toml. Got {value!r} ({type(value).__name__})"
    )


def _parse_bool(raw: str | None, default: bool, name: str) -> bool:
    """Parse an env-var string as bool. Accepts true/false/1/0/yes/no/on/off."""

    if raw is None:
        return default

    lower = raw.strip().lower()
    if lower in ("true", "1", "yes", "on"):
        return True

    if lower in ("false", "0", "no", "off"):
        return False

    raise ConfigError(f"{name} must be a boolean (true/false). Got {raw!r}")


def _as_bool(value: object, default: bool) -> bool:
    """Coerce a TOML value to bool, falling back to default if absent."""

    if value is None:
        return default

    if not isinstance(value, bool):
        raise ConfigError(
            f"Expected boolean in config.toml. Got {value!r} ({type(value).__name__})"
        )

    return value


def _as_str(value: object, default: str) -> str:
    """Coerce a TOML value to str, falling back to default if absent."""

    if value is None:
        return default

    if not isinstance(value, str):
        raise ConfigError(
            f"Expected string in config.toml. Got {value!r} ({type(value).__name__})"
        )

    return value


def _parse_str(raw: str | None, default: str, name: str) -> str:
    """Return raw if present and non-empty, else default. Reject empty-after-strip."""

    if raw is None:
        return default

    if not raw.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return raw


def _parse_source_dirs(env_raw: str | None, toml_value: object) -> tuple[Path, ...]:
    """Parse source_dirs from env (colon-separated) or the TOML list."""

    return _parse_path_list(env_raw, toml_value, field_name="source_dirs")


def _parse_path_list(
    env_raw: str | None, toml_value: object, field_name: str
) -> tuple[Path, ...]:
    """
    Parse a list of paths from env (colon-separated) or a TOML list of strings.

    Env, when present, wins over the TOML value.
    """

    if env_raw is not None:
        parts = [p for p in env_raw.split(os.pathsep) if p]
        return tuple(_resolve_path(p) for p in parts)

    if toml_value is None:
        return ()

    if not isinstance(toml_value, list) or not all(
        isinstance(p, str) for p in toml_value
    ):
        raise ConfigError(f"config.toml {field_name} must be a list of strings")
    return tuple(_resolve_path(p) for p in toml_value)


def _parse_watcher(raw: str) -> bool:
    """Return True for 'poll', False for 'native'. Raise on anything else."""

    normalised = raw.lower()
    if normalised not in _VALID_WATCHERS:
        raise ConfigError(f"RAG_WATCHER must be one of {_VALID_WATCHERS}. Got {raw!r}")
    return normalised == "poll"


def _parse_log_level(raw: str) -> str:
    """Validate a logging level name and return it uppercased."""

    normalised = raw.upper()
    if normalised not in _VALID_LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}. Got {raw!r}")
    return normalised


def _parse_log_format(raw: str) -> LogFormat:
    """Validate a log format name."""

    normalised = raw.strip().lower()
    if normalised not in _VALID_LOG_FORMATS:
        raise ConfigError(
            f"RAG_LOG_FORMAT must be one of {_VALID_LOG_FORMATS}. Got {raw!r}"
        )
    return normalised
