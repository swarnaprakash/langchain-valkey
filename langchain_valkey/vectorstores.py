"""Valkey vector store implementation for LangChain."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import valkey
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from .config import DEFAULT_INDEXED_METADATA_FIELDS, FlatConfig, HNSWConfig
from .exceptions import (
    ValkeyConnectionError,
    ValkeyQueryError,
    ValkeySchemaError,
)
from .maximal_marginal_relevance import MaximalMarginalRelevanceSelector
from .search import ValkeySearchClient

logger = logging.getLogger(__name__)


class ValkeyVectorStore(VectorStore):
    """Valkey vector store implementation.

    This class provides a LangChain-compatible interface for storing and
    retrieving documents using Valkey's vector search capabilities.

    Features:
    - HNSW and FLAT index support
    - Rich metadata filtering
    - Efficient pipeline operations
    - Comprehensive error handling
    - Auto-detect or explicit vector dimensions
    """

    def __init__(
        self,
        index_name: str,
        embedding: Embeddings,
        valkey_url: Optional[str] = None,
        client: Optional[valkey.Valkey] = None,
        config: Optional[Union[HNSWConfig, FlatConfig]] = None,
        key_prefix: Optional[str] = None,
        auto_create_index: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Valkey vector store.

        Args:
            valkey_url: Connection URL for Valkey instance (if client not provided)
            client: Pre-configured Valkey client (takes precedence over valkey_url)
            index_name: Name of the vector index
            embedding: Embedding model to use
            config: Index configuration (HNSWConfig or FlatConfig)
            key_prefix: Prefix for document keys in Valkey (defaults to index_name)
            auto_create_index: Whether to automatically create index if it doesn't exist
            **kwargs: Additional Valkey client options (only used when valkey_url is provided).
                     Ignored if client is provided. Common options include decode_responses,
                     socket_timeout, socket_connect_timeout, retry_on_timeout, etc.
        """
        if client is not None:
            self.client = client
            logger.debug("Using provided Valkey client")
        elif valkey_url is not None:
            try:
                self.client = valkey.from_url(valkey_url, **kwargs)
                logger.debug(f"Created new Valkey client from URL: {valkey_url}")
            except Exception as e:
                raise ValkeyConnectionError(
                    f"Failed to connect to Valkey at {valkey_url}: {e}"
                )
        else:
            raise ValueError("Either valkey_url or client must be provided")

        self.index_name = index_name
        self._embedding = embedding
        self.key_prefix = key_prefix if key_prefix is not None else index_name
        self.auto_create_index = auto_create_index

        if config is None:
            self.config = HNSWConfig(index_name=index_name)
            logger.debug("Using default HNSW configuration")
        else:
            self.config = config
            logger.debug(f"Using provided {type(config).__name__} configuration")

        self.indexed_metadata_fields = (
            config.indexed_metadata_fields
            if config and config.indexed_metadata_fields
            else DEFAULT_INDEXED_METADATA_FIELDS.copy()
        )

        self._vector_dimension = config.vector_dimension if config else None
        self._search_client = ValkeySearchClient(self.client)

        self._verify_connection()

        if self.auto_create_index:
            self._ensure_index_exists()

        logger.info(
            f"Initialized ValkeyVectorStore: index={index_name}, "
            f"config={type(self.config).__name__}, "
            f"dimension={self._vector_dimension or 'auto-detect'}, "
            f"indexed_fields={len(self.indexed_metadata_fields)}"
        )

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding model."""
        return self._embedding

    def _verify_connection(self) -> None:
        """Verify Valkey connection."""
        try:
            self.client.ping()
        except Exception as e:
            raise ValkeyConnectionError(f"Cannot connect to Valkey: {e}")

    def _index_exists(self) -> bool:
        """Check if the index exists."""
        try:
            self.client.execute_command("FT.INFO", self.index_name)
            return True
        except valkey.exceptions.ResponseError as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "unknown index" in error_msg:
                return False
            raise  # Re-raise unexpected errors

    def _ensure_index_exists(self) -> None:
        """Create index if it doesn't exist."""
        if self._index_exists():
            logger.debug(f"Index {self.index_name} already exists")
        else:
            logger.info(f"Creating index {self.index_name}")
            self._create_index()

    def _create_index(self) -> None:
        """Create the vector search index."""
        if self._vector_dimension is None:
            logger.warning(
                "Vector dimension not specified. Index will be created when first document is added."
            )
            return

        if self._index_exists():
            logger.info(f"Index {self.index_name} already exists, skipping creation")
            return

        try:
            cmd = [
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                f"{self.key_prefix}:",
            ]

            schema_fields = []

            if isinstance(self.config, HNSWConfig):
                vector_params = [
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    str(self._vector_dimension),
                    "DISTANCE_METRIC",
                    self.config.distance_metric.value,
                    "M",
                    str(self.config.m),
                    "EF_CONSTRUCTION",
                    str(self.config.ef_construction),
                    "EF_RUNTIME",
                    str(self.config.ef_runtime),
                ]
                schema_fields.extend(
                    ["embedding", "VECTOR", "HNSW", "12"] + vector_params
                )
            elif isinstance(self.config, FlatConfig):
                vector_params = [
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    str(self._vector_dimension),
                    "DISTANCE_METRIC",
                    self.config.distance_metric.value,
                ]
                schema_fields.extend(
                    ["embedding", "VECTOR", "FLAT", "6"] + vector_params
                )
            else:
                raise ValueError(f"Unsupported config type: {type(self.config)}")

            # TAG field type for string content (Valkey Search only supports TAG, NUMERIC, VECTOR)
            schema_fields.extend(["page_content", "TAG"])

            schema_fields.extend(["metadata", "TAG"])

            for field_name, field_type in self.indexed_metadata_fields.items():
                schema_fields.extend([field_name, field_type])

            cmd.extend(["SCHEMA"] + schema_fields)

            logger.info(
                f"Creating index with command: {' '.join(cmd[:10])}... (truncated)"
            )
            logger.debug(f"Full index creation command: {cmd}")
            logger.info(
                f"Index parameters: dimension={self._vector_dimension}, fields={len(schema_fields)//2}"
            )

            self.client.execute_command(*cmd)

            logger.info(f"Successfully created index {self.index_name}")

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise ValkeySchemaError(f"Failed to create index {self.index_name}: {e}")

    def _generate_doc_id(self) -> str:
        """Generate a unique document ID using UUID4."""
        return str(uuid.uuid4())

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Convert embedding to bytes for storage using NumPy binary format."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _deserialize_embedding(self, embedding_bytes: bytes) -> List[float]:
        """Convert bytes back to embedding list."""
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata to JSON string with datetime handling."""
        return json.dumps(metadata, default=str, ensure_ascii=False)

    def _deserialize_metadata(self, metadata_json: str) -> Dict[str, Any]:
        """Convert JSON string back to metadata dict."""
        return json.loads(metadata_json)

    def _build_documents_from_search_results(
        self,
        parsed_results: List[Tuple[str, Dict[str, Any], float, Optional[List[float]]]],
    ) -> List[Tuple[Document, float, Optional[List[float]]]]:
        """Build Document objects from parsed search results."""
        documents_with_scores = []

        for doc_id, fields, score, vector in parsed_results:
            # Clean document ID by removing key prefix
            clean_doc_id = doc_id.split(":", 1)[1] if ":" in doc_id else doc_id

            # Extract fields
            page_content = fields.get("page_content", "")
            metadata_str = fields.get("metadata", "{}")

            # Parse metadata using existing method
            metadata = self._deserialize_metadata(metadata_str)

            document = Document(
                page_content=page_content, metadata=metadata, id=clean_doc_id
            )
            documents_with_scores.append((document, score, vector))

        return documents_with_scores

    def _validate_filter_value(self, value: Any) -> str:
        """Validate and sanitize filter values for Valkey search syntax."""
        str_value = str(value)
        # These characters break Valkey search query syntax
        invalid_chars = {"{", "}", ":", "|", "(", ")"}
        if any(char in str_value for char in invalid_chars):
            raise ValueError(
                f"Filter value '{str_value}' contains invalid characters. "
                f"Characters {invalid_chars} are not allowed in filter values."
            )
        return str_value

    def _validate_filter(self, filter: Dict[str, Any]) -> None:
        """Validate filter fields and values before building query."""
        for key, value in filter.items():
            if key not in self.indexed_metadata_fields:
                raise ValueError(
                    f"Filter field '{key}' is not indexed. "
                    f"Available indexed fields: {list(self.indexed_metadata_fields.keys())}"
                )
            if value is not None:
                if isinstance(value, (list, tuple)):
                    for v in value:
                        self._validate_filter_value(v)
                else:
                    self._validate_filter_value(value)

    def _is_indexable_field(self, field_name: str, field_value: Any) -> bool:
        """Check if a metadata field should be indexed separately."""
        return (
            field_name in self.indexed_metadata_fields
            and field_value is not None
            and str(field_value).strip()  # Non-empty string representation
        )

    def _prepare_document_hash(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        doc_id: str,
    ) -> Dict[str, Any]:
        """Prepare document data for storage using dual metadata approach."""
        if len(embedding) != self._vector_dimension:
            raise ValkeySchemaError(
                f"Embedding dimension {len(embedding)} does not match "
                f"expected dimension {self._vector_dimension}"
            )

        # Dual metadata storage: JSON preserves all fields exactly, individual fields enable fast search filtering
        hash_data = {
            "page_content": content,
            "embedding": self._serialize_embedding(embedding),
            "metadata": self._serialize_metadata(metadata),  # JSON backup
        }

        for field_name, field_value in metadata.items():
            if self._is_indexable_field(field_name, field_value):
                hash_data[field_name] = str(field_value)

        return hash_data

    @override
    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add
            **kwargs: Additional arguments (may include 'ids')

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Generate embeddings for all documents at once (LangChain standard pattern)
        texts = [doc.page_content for doc in documents]
        embeddings = self._embedding.embed_documents(texts)

        # Auto-detect vector dimension and create index if needed
        # This happens only when dimension wasn't provided at initialization
        # and only on the first add_documents call - subsequent calls skip this
        if self._vector_dimension is None:
            self._vector_dimension = len(embeddings[0])
            if self.auto_create_index:
                self._create_index()

        return self._store_documents(documents, embeddings, **kwargs)

    def _store_documents(
        self, documents: List[Document], embeddings: List[List[float]], **kwargs: Any
    ) -> List[str]:
        """Store documents in Valkey using pipeline operations."""
        pipeline = self.client.pipeline()
        doc_ids = []

        provided_ids = kwargs.get("ids", [])

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            if provided_ids and i < len(provided_ids):
                doc_id = provided_ids[i]
            elif doc.id:  # Check if document has an ID
                doc_id = doc.id
            else:
                doc_id = self._generate_doc_id()

            hash_data = self._prepare_document_hash(
                doc.page_content, embedding, doc.metadata, doc_id
            )

            key = f"{self.key_prefix}:{doc_id}"
            pipeline.hset(key, mapping=hash_data)
            doc_ids.append(doc_id)

        try:
            pipeline.execute()
            logger.info(f"Successfully stored {len(doc_ids)} documents")
        except Exception as e:
            raise ValkeyQueryError(f"Failed to store documents: {e}")

        return doc_ids

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, Optional[List[float]]]]:
        """Search for similar documents with scores using a vector embedding.

        Args:
            embedding: Query embedding vector
            k: Number of documents to return
            filter: Optional metadata filters
            include_vectors: If True, return embedding vectors as well
            **kwargs: Additional search parameters

        Returns:
            List of (document, score, vector) tuples. Vector is None if include_vectors=False.
        """
        query_vector = self._serialize_embedding(embedding)

        knn_part = f"[KNN {k} @embedding $vec_param AS vector_score]"

        if filter:
            self._validate_filter(filter)
            filter_parts = []
            for key, value in filter.items():
                if value is not None:
                    filter_parts.append(f"@{key}:{{{value}}}")
            filter_expr = " ".join(filter_parts)
            query_str = f"{filter_expr} =>{knn_part}"
        else:
            query_str = f"*=>{knn_part}"

        params = {"vec_param": query_vector}

        try:
            parsed_results = self._search_client.search_with_vectors(
                self.index_name, query_str, params, include_vectors
            )
            return self._build_documents_from_search_results(parsed_results)
        except Exception as e:
            raise ValkeyQueryError(f"Vector search failed: {e}")

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance to embedding vector."""
        docs_with_scores_and_vectors = self._similarity_search_with_score_by_vector(
            embedding, k=fetch_k, include_vectors=True, **kwargs
        )

        if len(docs_with_scores_and_vectors) <= k:
            return [doc for doc, score, vector in docs_with_scores_and_vectors]

        # Filter out docs without embeddings
        valid_candidates = [
            (doc, score, vector)
            for doc, score, vector in docs_with_scores_and_vectors
            if vector is not None
        ]

        if not valid_candidates:
            return [doc for doc, score, vector in docs_with_scores_and_vectors[:k]]

        # Use MaximalMarginalRelevanceSelector for document selection
        mmr_selector = MaximalMarginalRelevanceSelector()
        candidate_embeddings = [vector for doc, score, vector in valid_candidates]

        selected_indices = mmr_selector.select(
            query_embedding=np.array(embedding),
            candidate_embeddings=candidate_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [valid_candidates[i][0] for i in selected_indices]

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance."""
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    @override
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents using a vector embedding.

        Args:
            embedding: Query embedding vector
            k: Number of documents to return
            filter: Optional metadata filters
            **kwargs: Additional search parameters

        Returns:
            List of similar documents
        """
        docs_with_scores = self._similarity_search_with_score_by_vector(
            embedding, k, filter, include_vectors=False, **kwargs
        )
        return [doc for doc, score, vector in docs_with_scores]

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filters
            **kwargs: Additional search parameters

        Returns:
            List of similar documents
        """
        query_embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, filter, **kwargs)

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filters
            **kwargs: Additional search parameters

        Returns:
            List of (document, score) tuples
        """
        query_embedding = self._embedding.embed_query(query)
        docs_with_scores = self._similarity_search_with_score_by_vector(
            query_embedding, k, filter, include_vectors=False, **kwargs
        )
        return [(doc, score) for doc, score, vector in docs_with_scores]

    @override
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments

        Returns:
            Success status
        """
        if not ids:
            return True

        try:
            pipeline = self.client.pipeline()
            for doc_id in ids:
                pipeline.delete(f"{self.key_prefix}:{doc_id}")
            results = pipeline.execute()

            return all(result > 0 for result in results)

        except Exception as e:
            raise ValkeyQueryError(f"Failed to delete documents: {e}")

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by IDs.

        Args:
            ids: List of document IDs
            **kwargs: Additional arguments

        Returns:
            List of documents
        """
        if not ids:
            return []

        try:
            pipeline = self.client.pipeline()
            for doc_id in ids:
                pipeline.hgetall(f"{self.key_prefix}:{doc_id}")
            results = pipeline.execute()

            documents = []
            for doc_id, result in zip(ids, results):
                if result:
                    page_content = result.get(b"page_content", b"").decode()
                    metadata_json = result.get(b"metadata", b"{}").decode()
                    metadata = self._deserialize_metadata(metadata_json)
                    documents.append(
                        Document(
                            id=doc_id, page_content=page_content, metadata=metadata
                        )
                    )

            return documents

        except Exception as e:
            raise ValkeyQueryError(f"Failed to retrieve documents: {e}")

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ValkeyVectorStore:
        """Create vector store from texts.

        Args:
            texts: List of texts
            embedding: Embedding model
            metadatas: Optional metadata for each text
            **kwargs: Additional arguments including valkey_url, index_name, etc.

        Returns:
            ValkeyVectorStore instance
        """
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata))

        vector_store = cls(embedding=embedding, **kwargs)

        vector_store.add_documents(documents)

        return vector_store
