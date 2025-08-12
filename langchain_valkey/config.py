"""Configuration classes for Valkey vector store."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"


# Default common metadata fields for indexing
DEFAULT_INDEXED_METADATA_FIELDS = {
    "source": "TAG",
    "category": "TAG",
    "author": "TAG",
    "date": "NUMERIC",
    "type": "TAG",
}


@dataclass
class HNSWConfig:
    """Configuration for HNSW (Hierarchical Navigable Small World) index.

    HNSW provides fast approximate nearest neighbor search with good recall.
    Optimized for speed over accuracy.
    """

    index_name: str
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    vector_dimension: Optional[int] = None  # Auto-detect if None
    m: int = 16  # Number of connections per node
    ef_construction: int = 200  # Size of dynamic candidate list during construction
    ef_runtime: int = 10  # Size of dynamic candidate list during search
    indexed_metadata_fields: Optional[Dict[str, str]] = None  # Custom metadata fields
    dynamic_schema: bool = False  # Future-proof parameter for dynamic indexing


@dataclass
class FlatConfig:
    """Configuration for FLAT (brute force) index.

    FLAT provides exact nearest neighbor search with perfect recall.
    Optimized for accuracy over speed.
    """

    index_name: str
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    vector_dimension: Optional[int] = None  # Auto-detect if None
    indexed_metadata_fields: Optional[Dict[str, str]] = None  # Custom metadata fields
    dynamic_schema: bool = False  # Future-proof parameter for dynamic indexing
