"""Early, supported configuration for JAX's persistent compilation cache."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

_CACHE_ENV = "PUBLIC_TRANSPORTATION_JAX_COMPILATION_CACHE_DIR"
_MIN_TIME_ENV = "PUBLIC_TRANSPORTATION_JAX_CACHE_MIN_COMPILE_SECONDS"
_MIN_SIZE_ENV = "PUBLIC_TRANSPORTATION_JAX_CACHE_MIN_ENTRY_BYTES"


@dataclass(frozen=True, slots=True)
class JaxCompilationCacheConfiguration:
    enabled: bool
    directory: str | None
    minimum_compile_seconds: float
    minimum_entry_bytes: int


def configure_jax_compilation_cache(
    directory: str | os.PathLike[str] | None = None,
    *,
    minimum_compile_seconds: float | None = None,
    minimum_entry_bytes: int | None = None,
    logger: logging.Logger | None = None,
) -> JaxCompilationCacheConfiguration:
    """Configure JAX's cache before the first computation or compilation.

    The directory may be passed explicitly or through
    ``PUBLIC_TRANSPORTATION_JAX_COMPILATION_CACHE_DIR``.  Threshold defaults of
    zero ensure that expensive and small serialized entries alike are eligible.
    Only public JAX configuration and compilation-cache APIs are used.
    """
    selected = directory or os.environ.get(_CACHE_ENV)
    minimum_compile_seconds = (
        float(os.environ.get(_MIN_TIME_ENV, "0"))
        if minimum_compile_seconds is None
        else float(minimum_compile_seconds)
    )
    minimum_entry_bytes = (
        int(os.environ.get(_MIN_SIZE_ENV, "0"))
        if minimum_entry_bytes is None
        else int(minimum_entry_bytes)
    )
    if minimum_compile_seconds < 0 or minimum_entry_bytes < 0:
        raise ValueError("JAX compilation-cache thresholds must be non-negative.")
    if selected is None:
        configuration = JaxCompilationCacheConfiguration(
            False, None, minimum_compile_seconds, minimum_entry_bytes
        )
    else:
        import jax
        from jax.experimental.compilation_cache import compilation_cache

        cache_path = str(Path(selected).expanduser().resolve())
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_enable_compilation_cache", True)
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", minimum_compile_seconds
        )
        jax.config.update(
            "jax_persistent_cache_min_entry_size_bytes", minimum_entry_bytes
        )
        compilation_cache.set_cache_dir(cache_path)
        configuration = JaxCompilationCacheConfiguration(
            True, cache_path, minimum_compile_seconds, minimum_entry_bytes
        )
    (logger or logging.getLogger(__name__)).info(
        "JAX persistent compilation cache enabled=%s directory=%s "
        "minimum_compile_seconds=%s minimum_entry_bytes=%s",
        configuration.enabled,
        configuration.directory,
        configuration.minimum_compile_seconds,
        configuration.minimum_entry_bytes,
    )
    return configuration


def configure_jax_compilation_cache_from_environment() -> (
    JaxCompilationCacheConfiguration
):
    """Apply environment configuration during package import."""
    return configure_jax_compilation_cache()
