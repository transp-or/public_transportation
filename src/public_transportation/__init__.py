from .version import __version__
from .compilation_cache import (
    JaxCompilationCacheConfiguration,
    configure_jax_compilation_cache,
    configure_jax_compilation_cache_from_environment,
)

_jax_compilation_cache = configure_jax_compilation_cache_from_environment()

__all__ = [
    "JaxCompilationCacheConfiguration",
    "__version__",
    "configure_jax_compilation_cache",
]
