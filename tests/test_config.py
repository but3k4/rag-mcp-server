"""Tests for rag.config.RagConfig loading, precedence, and validation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from rag.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_FETCH_N_MULTIPLIER,
    DEFAULT_HEALTH_PORT,
    DEFAULT_LOG_FORMAT,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_MAX_QUERY_LENGTH,
    DEFAULT_PORT,
    DEFAULT_QUERY_PREFIX,
    DEFAULT_RERANKER_ENABLED,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DEFAULT_WATCHER_DEBOUNCE_SECONDS,
    ConfigError,
    RagConfig,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build(
    env: dict[str, str] | None = None,
    toml: dict[str, object] | None = None,
    project_root: str = "/tmp/rag-root",
) -> RagConfig:
    """Build a RagConfig from explicit env + toml inputs for testing."""

    from pathlib import Path  # noqa: PLC0415

    return RagConfig.from_sources(
        env=env or {},
        toml_data=toml or {},
        project_root=Path(project_root),
    )


class TestDefaults:
    """Values produced when env and TOML provide nothing."""

    def test_defaults(self) -> None:
        """Empty sources produce the documented defaults."""

        cfg = _build()
        assert cfg.transport == "stdio"
        assert cfg.port == DEFAULT_PORT
        assert cfg.chunk_size == DEFAULT_CHUNK_SIZE
        assert cfg.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert cfg.use_polling is False
        assert cfg.log_level == "INFO"
        assert cfg.source_dirs == ()
        assert cfg.data_dir.name == ".rag-data"
        assert cfg.config_path.name == "config.toml"
        assert cfg.top_k == DEFAULT_TOP_K
        assert cfg.fetch_n_multiplier == DEFAULT_FETCH_N_MULTIPLIER
        assert cfg.rrf_k == DEFAULT_RRF_K
        assert cfg.embedder_model == DEFAULT_EMBEDDER_MODEL
        assert cfg.query_prefix == DEFAULT_QUERY_PREFIX
        assert cfg.reranker_enabled is DEFAULT_RERANKER_ENABLED
        assert cfg.reranker_model == DEFAULT_RERANKER_MODEL
        assert cfg.reranker_pool_size == DEFAULT_RERANKER_POOL_SIZE
        assert cfg.allowed_base_dirs == ()
        assert cfg.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_BYTES
        assert cfg.max_query_length == DEFAULT_MAX_QUERY_LENGTH
        assert cfg.watcher_debounce_seconds == DEFAULT_WATCHER_DEBOUNCE_SECONDS
        assert cfg.log_format == DEFAULT_LOG_FORMAT
        assert cfg.health_port == DEFAULT_HEALTH_PORT


class TestTomlDrivenValues:
    """Values read from the [indexing] TOML table when env is empty."""

    def test_toml_populates_indexing_fields(self, tmp_path: Path) -> None:
        """chunk_size, chunk_overlap, and source_dirs come from TOML."""

        from pathlib import Path as _Path  # noqa: PLC0415

        chunk_size_from_toml = 500
        chunk_overlap_from_toml = 50
        a = tmp_path / "a"
        b = tmp_path / "b"
        cfg = _build(
            toml={
                "indexing": {
                    "chunk_size": chunk_size_from_toml,
                    "chunk_overlap": chunk_overlap_from_toml,
                    "source_dirs": [str(a), str(b)],
                }
            },
        )
        assert cfg.chunk_size == chunk_size_from_toml
        assert cfg.chunk_overlap == chunk_overlap_from_toml
        assert list(cfg.source_dirs) == [
            _Path(str(a)).expanduser().resolve(),
            _Path(str(b)).expanduser().resolve(),
        ]


class TestEnvOverrides:
    """Env vars take precedence over TOML and defaults."""

    def test_env_beats_toml(self) -> None:
        """Env values win when both env and TOML are set."""

        env_chunk_size = 200
        env_chunk_overlap = 10
        cfg = _build(
            env={
                "RAG_CHUNK_SIZE": str(env_chunk_size),
                "RAG_CHUNK_OVERLAP": str(env_chunk_overlap),
            },
            toml={"indexing": {"chunk_size": 999, "chunk_overlap": 99}},
        )
        assert cfg.chunk_size == env_chunk_size
        assert cfg.chunk_overlap == env_chunk_overlap

    def test_source_dirs_env_overrides_toml(self, tmp_path: Path) -> None:
        """RAG_SOURCE_DIRS wins over [indexing].source_dirs."""

        from pathlib import Path as _Path  # noqa: PLC0415

        x = tmp_path / "x"
        y = tmp_path / "y"
        cfg = _build(
            env={"RAG_SOURCE_DIRS": f"{x}{os.pathsep}{y}"},
            toml={"indexing": {"source_dirs": [str(tmp_path / "from_toml")]}},
        )
        assert list(cfg.source_dirs) == [
            _Path(str(x)).expanduser().resolve(),
            _Path(str(y)).expanduser().resolve(),
        ]

    def test_transport_watcher_and_log_level_from_env(self) -> None:
        """Transport, watcher, and log level are controlled by env only."""

        custom_port = 9000
        cfg = _build(
            env={
                "TRANSPORT": "sse",
                "RAG_WATCHER": "poll",
                "LOG_LEVEL": "debug",
                "PORT": str(custom_port),
            }
        )
        assert cfg.transport == "sse"
        assert cfg.use_polling is True
        assert cfg.log_level == "DEBUG"
        assert cfg.port == custom_port


class TestRetrievalSection:
    """Tests for the [retrieval] TOML section and its env overrides."""

    def test_toml_populates_retrieval_fields(self) -> None:
        """[retrieval] table drives the retrieval knobs when env is empty."""

        top_k_from_toml = 8
        mult_from_toml = 6
        rrf_k_from_toml = 40
        cfg = _build(
            toml={
                "retrieval": {
                    "top_k": top_k_from_toml,
                    "fetch_n_multiplier": mult_from_toml,
                    "rrf_k": rrf_k_from_toml,
                    "embedder_model": "intfloat/e5-base-v2",
                    "query_prefix": "query: ",
                }
            },
        )
        assert cfg.top_k == top_k_from_toml
        assert cfg.fetch_n_multiplier == mult_from_toml
        assert cfg.rrf_k == rrf_k_from_toml
        assert cfg.embedder_model == "intfloat/e5-base-v2"
        assert cfg.query_prefix == "query: "

    def test_env_overrides_retrieval_toml(self) -> None:
        """Env vars beat [retrieval] values."""

        cfg = _build(
            env={
                "RAG_TOP_K": "3",
                "RAG_FETCH_N_MULTIPLIER": "2",
                "RAG_RRF_K": "10",
                "RAG_EMBEDDER_MODEL": "custom/model",
                "RAG_QUERY_PREFIX": "search: ",
            },
            toml={
                "retrieval": {
                    "top_k": 99,
                    "fetch_n_multiplier": 99,
                    "rrf_k": 99,
                    "embedder_model": "ignored",
                    "query_prefix": "ignored",
                }
            },
        )
        expected_top_k = 3
        expected_multiplier = 2
        expected_rrf_k = 10
        assert cfg.top_k == expected_top_k
        assert cfg.fetch_n_multiplier == expected_multiplier
        assert cfg.rrf_k == expected_rrf_k
        assert cfg.embedder_model == "custom/model"
        assert cfg.query_prefix == "search: "


class TestValidation:
    """Bad values raise ConfigError with a readable message."""

    def test_invalid_transport(self) -> None:
        """An unknown TRANSPORT value is rejected."""

        with pytest.raises(ConfigError, match="Invalid TRANSPORT"):
            _build(env={"TRANSPORT": "grpc"})

    def test_port_not_integer(self) -> None:
        """A non-numeric PORT value is rejected."""

        with pytest.raises(ConfigError, match="PORT must be an integer"):
            _build(env={"PORT": "eighty"})

    def test_port_out_of_range(self) -> None:
        """A PORT above 65535 is rejected."""

        with pytest.raises(ConfigError, match="PORT must be <= 65535"):
            _build(env={"PORT": "70000"})

    def test_port_zero_rejected(self) -> None:
        """A PORT below 1 is rejected."""

        with pytest.raises(ConfigError, match="PORT must be >= 1"):
            _build(env={"PORT": "0"})

    def test_chunk_size_non_positive(self) -> None:
        """chunk_size must be >= 1."""

        with pytest.raises(ConfigError, match="chunk_size must be >= 1"):
            _build(env={"RAG_CHUNK_SIZE": "0"})

    def test_chunk_overlap_negative(self) -> None:
        """chunk_overlap must be >= 0."""

        with pytest.raises(ConfigError, match="chunk_overlap must be >= 0"):
            _build(env={"RAG_CHUNK_OVERLAP": "-1"})

    def test_overlap_ge_chunk_size(self) -> None:
        """chunk_overlap equal to or larger than chunk_size is rejected."""

        with pytest.raises(
            ConfigError,
            match=r"chunk_overlap \(100\) must be < chunk_size \(100\)",
        ):
            _build(env={"RAG_CHUNK_SIZE": "100", "RAG_CHUNK_OVERLAP": "100"})

    def test_invalid_watcher(self) -> None:
        """An unknown RAG_WATCHER value is rejected."""

        with pytest.raises(ConfigError, match="RAG_WATCHER must be"):
            _build(env={"RAG_WATCHER": "inotify"})

    def test_invalid_log_level(self) -> None:
        """An unknown LOG_LEVEL value is rejected."""

        with pytest.raises(ConfigError, match="LOG_LEVEL must be"):
            _build(env={"LOG_LEVEL": "verbose"})

    def test_toml_indexing_wrong_type(self) -> None:
        """[indexing] must be a table, not a string."""

        with pytest.raises(ConfigError, match=r"\[indexing\] must be a table"):
            _build(toml={"indexing": "not-a-table"})

    def test_toml_source_dirs_wrong_type(self) -> None:
        """source_dirs must be a list of strings."""

        with pytest.raises(ConfigError, match="source_dirs must be a list"):
            _build(toml={"indexing": {"source_dirs": "not-a-list"}})

    def test_toml_chunk_size_wrong_type(self) -> None:
        """A string chunk_size in TOML is rejected. Int is required."""

        with pytest.raises(ConfigError, match="Expected integer"):
            _build(toml={"indexing": {"chunk_size": "1000"}})

    def test_toml_retrieval_wrong_type(self) -> None:
        """[retrieval] must be a table."""

        with pytest.raises(ConfigError, match=r"\[retrieval\] must be a table"):
            _build(toml={"retrieval": "not-a-table"})

    def test_top_k_non_positive(self) -> None:
        """top_k must be >= 1."""

        with pytest.raises(ConfigError, match="top_k must be >= 1"):
            _build(env={"RAG_TOP_K": "0"})

    def test_fetch_n_multiplier_non_positive(self) -> None:
        """fetch_n_multiplier must be >= 1."""

        with pytest.raises(ConfigError, match="fetch_n_multiplier must be >= 1"):
            _build(env={"RAG_FETCH_N_MULTIPLIER": "0"})

    def test_rrf_k_non_positive(self) -> None:
        """rrf_k must be >= 1."""

        with pytest.raises(ConfigError, match="rrf_k must be >= 1"):
            _build(env={"RAG_RRF_K": "0"})

    def test_embedder_model_empty_string(self) -> None:
        """An empty or whitespace embedder_model is rejected."""

        with pytest.raises(ConfigError, match="embedder_model must be"):
            _build(env={"RAG_EMBEDDER_MODEL": "   "})

    def test_toml_embedder_model_wrong_type(self) -> None:
        """A non-string embedder_model in TOML is rejected."""

        with pytest.raises(ConfigError, match="Expected string"):
            _build(toml={"retrieval": {"embedder_model": 42}})

    def test_reranker_pool_size_below_top_k_when_enabled(self) -> None:
        """reranker_pool_size must be >= top_k when reranking is enabled."""

        with pytest.raises(
            ConfigError,
            match=r"reranker_pool_size \(3\) must be >= top_k \(5\)",
        ):
            _build(
                env={"RAG_RERANKER_ENABLED": "true", "RAG_RERANKER_POOL_SIZE": "3"},
            )

    def test_reranker_pool_size_below_top_k_allowed_when_disabled(self) -> None:
        """A small reranker_pool_size is fine when reranker is disabled."""

        cfg = _build(env={"RAG_RERANKER_POOL_SIZE": "1"})
        assert cfg.reranker_enabled is False
        assert cfg.reranker_pool_size == 1

    def test_reranker_enabled_non_boolean(self) -> None:
        """A non-boolean RAG_RERANKER_ENABLED is rejected."""

        with pytest.raises(ConfigError, match="reranker_enabled must be a boolean"):
            _build(env={"RAG_RERANKER_ENABLED": "maybe"})

    def test_toml_reranker_enabled_wrong_type(self) -> None:
        """A non-boolean reranker_enabled in TOML is rejected."""

        with pytest.raises(ConfigError, match="Expected boolean"):
            _build(toml={"retrieval": {"reranker_enabled": "yes"}})

    def test_toml_security_wrong_type(self) -> None:
        """[security] must be a table."""

        with pytest.raises(ConfigError, match=r"\[security\] must be a table"):
            _build(toml={"security": "nope"})

    def test_toml_allowed_base_dirs_wrong_type(self) -> None:
        """allowed_base_dirs must be a list of strings."""

        with pytest.raises(
            ConfigError, match="allowed_base_dirs must be a list of strings"
        ):
            _build(toml={"security": {"allowed_base_dirs": "one-string"}})

    def test_max_file_size_zero_rejected(self) -> None:
        """max_file_size_bytes must be >= 1."""

        with pytest.raises(ConfigError, match="max_file_size_bytes must be >= 1"):
            _build(env={"RAG_MAX_FILE_SIZE_BYTES": "0"})

    def test_max_query_length_zero_rejected(self) -> None:
        """max_query_length must be >= 1."""

        with pytest.raises(ConfigError, match="max_query_length must be >= 1"):
            _build(env={"RAG_MAX_QUERY_LENGTH": "0"})

    def test_invalid_log_format(self) -> None:
        """RAG_LOG_FORMAT must be json or text."""

        with pytest.raises(ConfigError, match="RAG_LOG_FORMAT must be"):
            _build(env={"RAG_LOG_FORMAT": "xml"})

    def test_watcher_debounce_negative(self) -> None:
        """watcher_debounce_seconds must be >= 0."""

        with pytest.raises(ConfigError, match="watcher_debounce_seconds must be >= 0"):
            _build(env={"RAG_WATCHER_DEBOUNCE_SECONDS": "-1"})

    def test_watcher_debounce_not_a_number(self) -> None:
        """A non-numeric RAG_WATCHER_DEBOUNCE_SECONDS is rejected."""

        with pytest.raises(
            ConfigError, match="watcher_debounce_seconds must be a number"
        ):
            _build(env={"RAG_WATCHER_DEBOUNCE_SECONDS": "soon"})

    def test_watcher_debounce_zero_disables(self) -> None:
        """watcher_debounce_seconds of 0 is valid and means debouncing is off."""

        cfg = _build(env={"RAG_WATCHER_DEBOUNCE_SECONDS": "0"})
        assert cfg.watcher_debounce_seconds == 0.0

    def test_watcher_debounce_toml_float(self) -> None:
        """A TOML float value populates watcher_debounce_seconds."""

        expected = 1.5
        cfg = _build(toml={"indexing": {"watcher_debounce_seconds": expected}})
        assert cfg.watcher_debounce_seconds == expected

    def test_toml_watcher_debounce_wrong_type(self) -> None:
        """A non-numeric TOML value is rejected."""

        with pytest.raises(ConfigError, match="Expected number"):
            _build(toml={"indexing": {"watcher_debounce_seconds": "half-second"}})

    def test_health_port_negative(self) -> None:
        """health_port must be >= 0."""

        with pytest.raises(ConfigError, match="health_port must be >= 0"):
            _build(env={"RAG_HEALTH_PORT": "-1"})

    def test_health_port_out_of_range(self) -> None:
        """health_port must be <= 65535."""

        with pytest.raises(ConfigError, match="health_port must be <= 65535"):
            _build(env={"RAG_HEALTH_PORT": "70000"})

    def test_health_port_zero_disables(self) -> None:
        """health_port of 0 is valid and means the sidecar is off."""

        cfg = _build(env={"RAG_HEALTH_PORT": "0"})
        assert cfg.health_port == 0

    def test_health_port_env_enables(self) -> None:
        """Setting RAG_HEALTH_PORT to a valid TCP port enables the sidecar."""

        cfg = _build(env={"RAG_HEALTH_PORT": "8766"})
        expected_port = 8766
        assert cfg.health_port == expected_port

    def test_log_format_text_accepted(self) -> None:
        """'text' is a valid log_format value."""

        cfg = _build(env={"RAG_LOG_FORMAT": "text"})
        assert cfg.log_format == "text"


class TestSecuritySection:
    """Tests for the [security] TOML section and its env override."""

    def test_allowed_base_dirs_from_toml(self, tmp_path: Path) -> None:
        """[security].allowed_base_dirs resolves to a tuple of Paths."""

        from pathlib import Path as _Path  # noqa: PLC0415

        a = tmp_path / "a"
        b = tmp_path / "b"
        cfg = _build(
            toml={"security": {"allowed_base_dirs": [str(a), str(b)]}},
        )
        assert list(cfg.allowed_base_dirs) == [
            _Path(str(a)).expanduser().resolve(),
            _Path(str(b)).expanduser().resolve(),
        ]

    def test_allowed_base_dirs_env_overrides_toml(self, tmp_path: Path) -> None:
        """RAG_ALLOWED_BASE_DIRS wins over [security].allowed_base_dirs."""

        import os  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415

        x = tmp_path / "x"
        cfg = _build(
            env={"RAG_ALLOWED_BASE_DIRS": f"{x}{os.pathsep}"},
            toml={"security": {"allowed_base_dirs": [str(tmp_path / "from_toml")]}},
        )
        assert list(cfg.allowed_base_dirs) == [
            _Path(str(x)).expanduser().resolve(),
        ]


class TestLoadConfig:
    """load_config() reads the TOML file from disk and applies env overrides."""

    def test_missing_file_is_treated_as_empty(self, tmp_path: Path) -> None:
        """A missing config.toml falls back to defaults, not an error."""

        cfg = load_config(project_root=tmp_path, env={})
        assert cfg.chunk_size == DEFAULT_CHUNK_SIZE
        assert cfg.source_dirs == ()

    def test_reads_toml_file(self, tmp_path: Path) -> None:
        """A present config.toml populates the returned RagConfig."""

        expected_chunk_size = 800
        expected_chunk_overlap = 80
        (tmp_path / "config.toml").write_text(
            f"[indexing]\n"
            f"chunk_size = {expected_chunk_size}\n"
            f"chunk_overlap = {expected_chunk_overlap}\n",
            encoding="utf-8",
        )

        cfg = load_config(project_root=tmp_path, env={})
        assert cfg.chunk_size == expected_chunk_size
        assert cfg.chunk_overlap == expected_chunk_overlap

    def test_invalid_toml_raises_config_error(self, tmp_path: Path) -> None:
        """A syntax error in config.toml surfaces as ConfigError."""

        (tmp_path / "config.toml").write_text("this is not = [valid toml\n")
        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_config(project_root=tmp_path, env={})
