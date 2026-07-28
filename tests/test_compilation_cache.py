from __future__ import annotations

import logging

import pytest

from public_transportation.compilation_cache import configure_jax_compilation_cache


def test_explicit_compilation_cache_configuration(tmp_path, caplog):
    cache = tmp_path / "jax-cache"
    with caplog.at_level(logging.INFO):
        configuration = configure_jax_compilation_cache(
            cache,
            minimum_compile_seconds=0.25,
            minimum_entry_bytes=128,
        )

    assert configuration.enabled
    assert configuration.directory == str(cache.resolve())
    assert configuration.minimum_compile_seconds == 0.25
    assert configuration.minimum_entry_bytes == 128
    assert cache.is_dir()
    assert "JAX persistent compilation cache enabled=True" in caplog.text


@pytest.mark.parametrize(
    ("minimum_compile_seconds", "minimum_entry_bytes"), [(-1.0, 0), (0.0, -1)]
)
def test_compilation_cache_rejects_negative_thresholds(
    tmp_path, minimum_compile_seconds, minimum_entry_bytes
):
    with pytest.raises(ValueError, match="must be non-negative"):
        configure_jax_compilation_cache(
            tmp_path,
            minimum_compile_seconds=minimum_compile_seconds,
            minimum_entry_bytes=minimum_entry_bytes,
        )
