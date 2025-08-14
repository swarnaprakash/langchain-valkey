"""Unit tests for Valkey vector store."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import FakeEmbeddings

from langchain_valkey import ValkeyVectorStore
from langchain_valkey.config import HNSWConfig, DistanceMetric
from langchain_valkey.exceptions import ValkeyConnectionError


@pytest.fixture
def mock_embeddings():
    """Fixture for fake embeddings."""
    return FakeEmbeddings(size=3)


def setup_mock_client():
    """Setup a mock Valkey client with common responses."""
    client = MagicMock()
    client.ping.return_value = True
    client.execute_command.return_value = [{"name": b"search", "ver": 20804}]
    return client


@pytest.fixture
def mock_valkey_client():
    """Fixture for mock Valkey client."""
    return setup_mock_client()


def test_import():
    """Test that ValkeyVectorStore can be imported."""
    from langchain_valkey import ValkeyVectorStore

    assert ValkeyVectorStore is not None


def test_vectorstore_class():
    """Test ValkeyVectorStore class exists and has expected attributes."""
    assert hasattr(ValkeyVectorStore, "__init__")
    assert hasattr(ValkeyVectorStore, "add_documents")
    assert hasattr(ValkeyVectorStore, "similarity_search")


def test_vectorstore_inheritance():
    """Test ValkeyVectorStore inherits from VectorStore."""
    from langchain_core.vectorstores import VectorStore

    assert issubclass(ValkeyVectorStore, VectorStore)


@patch("valkey.from_url")
def test_init_success(mock_from_url, mock_embeddings):
    """Test successful ValkeyVectorStore initialization."""
    mock_client = setup_mock_client()
    mock_from_url.return_value = mock_client

    vectorstore = ValkeyVectorStore(
        index_name="test_index",
        embedding=mock_embeddings,
        valkey_url="valkey://localhost:6379",
        auto_create_index=False,
    )

    assert vectorstore.index_name == "test_index"
    assert vectorstore._embedding == mock_embeddings
    assert vectorstore.client == mock_client
    mock_from_url.assert_called_once_with("valkey://localhost:6379")


@patch("valkey.from_url")
def test_init_connection_error(mock_from_url, mock_embeddings):
    """Test ValkeyVectorStore initialization with connection error."""
    mock_from_url.side_effect = Exception("Connection failed")

    with pytest.raises(ValkeyConnectionError):
        ValkeyVectorStore(
            index_name="test_index",
            embedding=mock_embeddings,
            valkey_url="valkey://localhost:6379",
        )


def test_config_validation():
    """Test configuration validation."""
    config = HNSWConfig(
        index_name="test_index",
        distance_metric=DistanceMetric.COSINE,
        m=16,
        ef_construction=200,
        ef_runtime=10,
    )

    assert config.index_name == "test_index"
    assert config.distance_metric == DistanceMetric.COSINE
    assert config.m == 16


def test_similarity_search_by_vector(mock_valkey_client, mock_embeddings):
    """Test similarity search by vector."""
    # Setup
    vectorstore = ValkeyVectorStore(
        index_name="test_index",
        embedding=mock_embeddings,
        client=mock_valkey_client,
        auto_create_index=False,
    )
    vectorstore._vector_dimension = 3

    # Mock raw FT.SEARCH response: [count, doc_id, [field_name, field_value, ...]]
    mock_raw_result = [
        1,  # count
        "test_index:doc1",  # doc_id
        [  # field data
            b"vector_score",
            0.95,
            b"page_content",
            b"Test content",
            b"metadata",
            b'{"source": "test"}',
        ],
    ]
    mock_valkey_client.execute_command.return_value = mock_raw_result

    # Test vector search
    query_vector = [0.1, 0.2, 0.3]
    results = vectorstore.similarity_search_by_vector(query_vector, k=1)

    # Verify results
    assert len(results) == 1
    assert results[0].page_content == "Test content"
    assert results[0].metadata == {"source": "test"}

    # Verify execute_command was called with correct parameters
    execute_call = mock_valkey_client.execute_command.call_args
    assert execute_call[0][0] == "FT.SEARCH"
    assert execute_call[0][1] == "test_index"
    assert "*=>[KNN 1 @embedding $vec_param AS vector_score]" in execute_call[0][2]


def test_similarity_search_by_vector_with_filter(mock_valkey_client, mock_embeddings):
    """Test similarity search by vector with metadata filter."""
    # Setup
    vectorstore = ValkeyVectorStore(
        index_name="test_index",
        embedding=mock_embeddings,
        client=mock_valkey_client,
        auto_create_index=False,
    )
    vectorstore._vector_dimension = 3

    # Mock response
    mock_raw_result = [
        1,
        "test_index:doc1",
        [
            b"vector_score",
            0.95,
            b"page_content",
            b"Filtered content",
            b"metadata",
            b'{"category": "tech"}',
        ],
    ]
    mock_valkey_client.execute_command.return_value = mock_raw_result

    # Test with filter
    query_vector = [0.1, 0.2, 0.3]
    results = vectorstore.similarity_search_by_vector(
        query_vector, k=1, filter={"category": "tech"}
    )

    # Verify results
    assert len(results) == 1
    assert results[0].page_content == "Filtered content"

    # Verify filter was applied in query
    execute_call = mock_valkey_client.execute_command.call_args
    query_string = execute_call[0][2]
    assert "@category:{tech}" in query_string


@pytest.fixture
def vectorstore(mock_valkey_client, mock_embeddings):
    """Create ValkeyVectorStore instance for testing utilities."""
    return ValkeyVectorStore(
        index_name="test_index",
        embedding=mock_embeddings,
        client=mock_valkey_client,
        auto_create_index=False,
    )


def test_embedding_serialization_preserves_precision(vectorstore):
    """Test that embedding serialization preserves floating point precision."""
    # Test with various edge cases
    test_cases = [
        [0.1, 0.2, 0.3],  # Simple case
        [0.0, -1.0, 1.0],  # Edge values
        [1e-6, 1e6, -1e-6],  # Very small and large numbers
        [],  # Empty embedding
    ]

    for original in test_cases:
        serialized = vectorstore._serialize_embedding(original)
        deserialized = vectorstore._deserialize_embedding(serialized)

        assert len(deserialized) == len(original)
        for orig, deser in zip(original, deserialized):
            assert abs(orig - deser) < 1e-6


def test_metadata_serialization_handles_complex_types(vectorstore):
    """Test metadata serialization with various Python types."""
    test_metadata = {
        "string": "test",
        "integer": 42,
        "float": 3.14,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "null": None,
    }

    serialized = vectorstore._serialize_metadata(test_metadata)
    deserialized = vectorstore._deserialize_metadata(serialized)

    assert deserialized == test_metadata


def test_filter_value_type_conversions(vectorstore):
    """Test that filter values are converted to strings correctly."""
    test_cases = [
        ("string_value", "string_value"),
        (42, "42"),
        (3.14159, "3.14159"),
        (True, "True"),
        (False, "False"),
    ]

    for input_val, expected in test_cases:
        result = vectorstore._validate_filter_value(input_val)
        assert result == expected
        assert isinstance(result, str)


def test_document_ids_are_unique_and_valid(vectorstore):
    """Test document ID generation produces unique, valid identifiers."""
    # Generate many IDs to test uniqueness
    ids = [vectorstore._generate_doc_id() for _ in range(100)]

    # All should be strings
    assert all(isinstance(doc_id, str) for doc_id in ids)

    # All should be non-empty
    assert all(len(doc_id) > 0 for doc_id in ids)

    # All should be unique
    assert len(set(ids)) == len(ids)

    # Should not contain problematic characters for Valkey keys
    for doc_id in ids[:10]:  # Check first 10
        assert " " not in doc_id  # No spaces
        assert "\n" not in doc_id  # No newlines
        assert "\t" not in doc_id  # No tabs


def test_validate_filter_success(vectorstore):
    """Test filter validation with valid indexed fields."""
    # Setup indexed fields
    vectorstore.indexed_metadata_fields = {"category": "TAG", "score": "NUMERIC"}

    # Valid filters should not raise exceptions
    vectorstore._validate_filter({"category": "tech"})
    vectorstore._validate_filter({"score": "95"})
    vectorstore._validate_filter({"category": "tech", "score": "95"})
    vectorstore._validate_filter({"category": ["tech", "science"]})


def test_validate_filter_invalid_field(vectorstore):
    """Test filter validation with non-indexed fields."""
    vectorstore.indexed_metadata_fields = {"category": "TAG"}

    # Non-indexed field should raise ValueError
    with pytest.raises(ValueError, match="Filter field 'invalid_field' is not indexed"):
        vectorstore._validate_filter({"invalid_field": "value"})


def test_validate_filter_invalid_characters(vectorstore):
    """Test filter validation with invalid characters in values."""
    vectorstore.indexed_metadata_fields = {"category": "TAG"}

    # Invalid characters should raise ValueError
    invalid_values = ["{tech}", "tech:value", "tech|other", "(tech)", "tech}"]

    for invalid_value in invalid_values:
        with pytest.raises(ValueError, match="contains invalid characters"):
            vectorstore._validate_filter({"category": invalid_value})
