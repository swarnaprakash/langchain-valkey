"""LangChain integration for Valkey vector store."""

from .config import DistanceMetric, FlatConfig, HNSWConfig
from .exceptions import (
    ValkeyConnectionError,
    ValkeyQueryError,
    ValkeySchemaError,
    ValkeySearchModuleError,
    ValkeyVectorStoreError,
)
from .maximal_marginal_relevance import MaximalMarginalRelevanceSelector
from .vectorstores import ValkeyVectorStore

__all__ = [
    "ValkeyVectorStore",
    "HNSWConfig",
    "FlatConfig",
    "DistanceMetric",
    "MaximalMarginalRelevanceSelector",
    "ValkeyVectorStoreError",
    "ValkeyConnectionError",
    "ValkeySearchModuleError",
    "ValkeySchemaError",
    "ValkeyQueryError",
]
__version__ = "0.1.0"
