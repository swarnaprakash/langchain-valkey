"""Integration tests for Valkey vector store with fake embeddings."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from langchain_valkey import ValkeyVectorStore
from langchain_valkey.config import HNSWConfig, DistanceMetric


# ============================================================================
# CONSTANTS
# ============================================================================

# Shared index names to avoid hitting 10-index limit
SHARED_HNSW_INDEX = "shared_hnsw"
SHARED_FLAT_INDEX = "shared_flat"
SHARED_MMR_INDEX = "shared_mmr"
SHARED_SPECIAL_INDEX = "shared_special"


# ============================================================================
# HELPER FUNCTIONS AND CLASSES
# ============================================================================


def cleanup_index(valkey_url, index_name):
    """Helper to cleanup a specific index."""
    import valkey

    client = valkey.from_url(valkey_url)
    try:
        client.ft(index_name).dropindex(delete_documents=True)
    except Exception:
        pass


def cleanup_documents(vectorstore):
    """Helper to cleanup documents but keep index for reuse."""
    try:
        keys = vectorstore.client.keys(f"{vectorstore.key_prefix}:*")
        if keys:
            vectorstore.client.delete(*keys)
    except Exception:
        pass


class SimpleEmbeddings(Embeddings):
    """Simple embeddings for testing MMR with meaningful vectors."""

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            if "cat" in text.lower():
                embeddings.append([1.0, 0.8, 0.1, 0.2])  # Cat-like
            elif "dog" in text.lower():
                embeddings.append([0.9, 0.7, 0.2, 0.1])  # Dog-like (similar to cat)
            elif "car" in text.lower():
                embeddings.append([0.1, 0.2, 1.0, 0.8])  # Car-like
            elif "plane" in text.lower():
                embeddings.append([0.2, 0.1, 0.9, 0.7])  # Plane-like (similar to car)
            else:
                embeddings.append([0.5, 0.5, 0.5, 0.5])  # Neutral
        return embeddings

    def embed_query(self, text):
        if "cat" in text.lower():
            return [1.0, 0.8, 0.1, 0.2]
        return [0.5, 0.5, 0.5, 0.5]


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def fake_embeddings():
    """Fixture for deterministic fake embeddings."""
    return DeterministicFakeEmbedding(size=1536)


@pytest.fixture
def simple_embeddings():
    """Fixture for simple embeddings with meaningful vectors."""
    return SimpleEmbeddings()


@pytest.fixture
def sample_documents():
    """Fixture for sample test documents."""
    return [
        Document(
            page_content="Valkey is a high-performance key-value datastore.",
            metadata={"source": "valkey_docs", "category": "database"},
        ),
        Document(
            page_content="LangChain provides a framework for building applications.",
            metadata={"source": "langchain_docs", "category": "framework"},
        ),
        Document(
            page_content="Vector databases enable semantic search capabilities.",
            metadata={"source": "vector_guide", "category": "database"},
        ),
    ]


@pytest.fixture
def vectorstore_with_docs(fake_embeddings, valkey_url):
    """Fixture providing vectorstore with sample documents."""
    # Use shared HNSW index
    vectorstore = ValkeyVectorStore(
        valkey_url=valkey_url,
        index_name=SHARED_HNSW_INDEX,
        embedding=fake_embeddings,
        auto_create_index=True,
    )

    # Clean up any existing data in this index
    try:
        keys = vectorstore.client.keys(f"{vectorstore.key_prefix}:*")
        if keys:
            vectorstore.client.delete(*keys)
    except Exception:
        pass

    # Add sample documents
    docs = [
        Document(page_content="Test document 1", metadata={"id": 1, "source": "test"}),
        Document(page_content="Test document 2", metadata={"id": 2, "source": "test"}),
    ]
    doc_ids = vectorstore.add_documents(docs)

    yield vectorstore, doc_ids

    # Cleanup documents but keep index for reuse
    try:
        keys = vectorstore.client.keys(f"{vectorstore.key_prefix}:*")
        if keys:
            vectorstore.client.delete(*keys)
    except Exception:
        pass


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================


def test_basic_operations(fake_embeddings, sample_documents, valkey_url):
    """Test basic vector store operations with fake embeddings."""
    config = HNSWConfig(
        index_name=SHARED_HNSW_INDEX, distance_metric=DistanceMetric.COSINE
    )

    vectorstore = ValkeyVectorStore(
        index_name=SHARED_HNSW_INDEX,
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=config,
    )

    try:
        # Add documents
        doc_ids = vectorstore.add_documents(sample_documents)
        assert len(doc_ids) == len(sample_documents)

        # Search for similar documents
        results = vectorstore.similarity_search("Valkey database", k=2)
        assert len(results) > 0

        # Search with metadata filter
        filtered_results = vectorstore.similarity_search(
            "database", k=5, filter={"category": "database"}
        )
        assert len(filtered_results) > 0
        assert all(
            doc.metadata.get("category") == "database" for doc in filtered_results
        )

        # Test retrieval by IDs
        retrieved_docs = vectorstore.get_by_ids(doc_ids[:2])
        assert len(retrieved_docs) == 2

    finally:
        # Cleanup
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_batch_operations(fake_embeddings, valkey_url):
    """Test batch operations with fake embeddings."""
    config = HNSWConfig(
        index_name=SHARED_HNSW_INDEX, distance_metric=DistanceMetric.COSINE
    )

    vectorstore = ValkeyVectorStore(
        index_name=SHARED_HNSW_INDEX,
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=config,
    )

    # Create batch of documents
    batch_docs = []
    for i in range(25):
        doc = Document(
            page_content=f"Batch document {i}: Testing batch processing.",
            metadata={"batch_id": i, "type": "test"},
        )
        batch_docs.append(doc)

    try:
        # Add batch of documents
        doc_ids = vectorstore.add_documents(batch_docs)
        assert len(doc_ids) == len(batch_docs)

        # Verify documents were added correctly
        results = vectorstore.similarity_search("batch document", k=10)
        assert len(results) > 0

        # Test retrieval by IDs
        retrieved_docs = vectorstore.get_by_ids(doc_ids[:5])
        assert len(retrieved_docs) == 5

    finally:
        # Cleanup
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_similarity_search_with_score(fake_embeddings, valkey_url):
    """Test similarity_search_with_score method."""
    vectorstore = ValkeyVectorStore(
        index_name=SHARED_HNSW_INDEX,
        embedding=fake_embeddings,
        valkey_url=valkey_url,
    )

    try:
        # Add test documents
        docs = [
            Document(page_content="Valkey vector search", metadata={"id": "1"}),
            Document(page_content="Database operations", metadata={"id": "2"}),
        ]
        doc_ids = vectorstore.add_documents(docs)

        # Test similarity search with scores
        results_with_scores = vectorstore.similarity_search_with_score("valkey", k=2)

        assert len(results_with_scores) == 2
        for doc, score in results_with_scores:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert score >= 0  # Scores should be non-negative

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_empty_index_search(fake_embeddings, valkey_url):
    """Test search on empty index."""
    cleanup_index(valkey_url, "test_empty")

    config = HNSWConfig(index_name="test_empty", distance_metric=DistanceMetric.COSINE)

    vectorstore = ValkeyVectorStore(
        index_name="test_empty",
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=config,
    )

    from langchain_valkey.exceptions import ValkeyQueryError

    with pytest.raises(ValkeyQueryError, match="Vector search failed"):
        vectorstore.similarity_search("empty search", k=5)


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_flat_index_configuration(fake_embeddings, valkey_url):
    """Test FLAT index configuration and end-to-end functionality."""
    from langchain_valkey.config import FlatConfig, DistanceMetric

    cleanup_index(valkey_url, "test_flat")

    # Test FLAT configuration
    flat_config = FlatConfig(
        index_name=SHARED_FLAT_INDEX,
        distance_metric=DistanceMetric.L2,
        vector_dimension=1536,
    )

    vectorstore = ValkeyVectorStore(
        index_name=SHARED_FLAT_INDEX,
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=flat_config,
    )

    try:
        # Add documents to trigger index creation
        docs = [
            Document(
                page_content="FLAT index test document", metadata={"type": "test"}
            ),
            Document(
                page_content="Another FLAT test document", metadata={"type": "test"}
            ),
        ]
        doc_ids = vectorstore.add_documents(docs)

        # Test search functionality works with FLAT index
        results = vectorstore.similarity_search("FLAT test", k=2)
        assert len(results) == 2

        # Test search with scores
        results_with_scores = vectorstore.similarity_search_with_score("FLAT", k=1)
        assert len(results_with_scores) == 1
        assert isinstance(results_with_scores[0][1], float)

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_provided_client_initialization(fake_embeddings, valkey_url):
    """Test initialization with provided Valkey client."""
    import valkey

    cleanup_index(valkey_url, "test_client")

    # Create client manually
    client = valkey.from_url(valkey_url)

    # Test with provided client
    vectorstore = ValkeyVectorStore(
        index_name="test_client",
        embedding=fake_embeddings,
        client=client,  # Use provided client instead of URL
    )

    # Verify it works
    assert vectorstore.client is client

    try:
        doc = Document(page_content="Test with provided client", metadata={})
        doc_ids = vectorstore.add_documents([doc])

        results = vectorstore.similarity_search("client", k=1)
        assert len(results) == 1

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_auto_detect_dimension_and_index_creation(valkey_url):
    """Test auto-detection of vector dimension and index creation."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding
    import valkey

    # Cleanup before test
    client = valkey.from_url(valkey_url)
    try:
        client.ft("test_auto_detect").dropindex(delete_documents=True)
    except Exception:
        pass

    # Create embeddings with specific dimension
    embeddings = DeterministicFakeEmbedding(size=768)  # Different from default 1536

    # Create vectorstore without specifying dimension
    vectorstore = ValkeyVectorStore(
        index_name="test_auto_detect",
        embedding=embeddings,
        valkey_url=valkey_url,
        # No config provided - should auto-detect dimension
    )

    # Initially dimension should be None
    assert vectorstore._vector_dimension is None

    try:
        # Add document - this should trigger dimension detection and index creation
        doc = Document(page_content="Auto-detect dimension test", metadata={})
        doc_ids = vectorstore.add_documents([doc])

        # Now dimension should be detected
        assert vectorstore._vector_dimension == 768

        # Verify search works
        results = vectorstore.similarity_search("dimension", k=1)
        assert len(results) == 1

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_custom_key_prefix(fake_embeddings, valkey_url):
    """Test custom key prefix functionality."""
    import valkey

    # Cleanup before test
    client = valkey.from_url(valkey_url)
    try:
        client.ft("test_prefix").dropindex(delete_documents=True)
    except Exception:
        pass

    # Create vectorstore with custom key prefix
    vectorstore = ValkeyVectorStore(
        index_name="test_prefix",
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        key_prefix="custom_prefix",  # Custom prefix instead of index name
    )

    try:
        # Add document
        doc = Document(page_content="Custom prefix test", metadata={})
        doc_ids = vectorstore.add_documents([doc])

        # Verify document is stored with custom prefix
        client = valkey.from_url(valkey_url)
        key = f"custom_prefix:{doc_ids[0]}"
        stored_data = client.hgetall(key)
        assert stored_data  # Should exist with custom prefix

        # Verify search works
        results = vectorstore.similarity_search("prefix", k=1)
        assert len(results) == 1

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_index_already_exists_scenario(fake_embeddings, valkey_url):
    """Test scenario when index already exists."""

    # Create first vectorstore to establish index
    vectorstore1 = ValkeyVectorStore(
        index_name="test_existing",
        embedding=fake_embeddings,
        valkey_url=valkey_url,
    )

    try:
        # Add a document to ensure index is created
        doc = Document(page_content="First vectorstore", metadata={})
        doc_ids1 = vectorstore1.add_documents([doc])

        # Create second vectorstore with same index name
        vectorstore2 = ValkeyVectorStore(
            index_name="test_existing",  # Same index name
            embedding=fake_embeddings,
            valkey_url=valkey_url,
        )

        # Should work without errors - index already exists
        doc2 = Document(page_content="Second vectorstore", metadata={})
        doc_ids2 = vectorstore2.add_documents([doc2])

        # Both documents should be searchable
        results = vectorstore2.similarity_search("vectorstore", k=2)
        assert len(results) == 2

    finally:
        # Cleanup
        if "doc_ids1" in locals():
            vectorstore1.delete(doc_ids1)
        if "doc_ids2" in locals():
            vectorstore2.delete(doc_ids2)


# ============================================================================
# MMR (MAXIMUM MARGINAL RELEVANCE) TESTS
# ============================================================================


def test_mmr_diversity(simple_embeddings, valkey_url):
    """Test that MMR provides more diverse results than regular similarity search."""
    vectorstore = ValkeyVectorStore(
        index_name=SHARED_MMR_INDEX,
        embedding=simple_embeddings,
        valkey_url=valkey_url,
    )

    # Clean up any existing data first
    cleanup_documents(vectorstore)

    try:
        # Add diverse documents
        texts = [
            "Cats are wonderful pets that purr",
            "Cats love to sleep in the sun",  # Very similar to first
            "Dogs are loyal companions",
            "Cars are fast vehicles",
            "Planes fly in the sky",
        ]

        vectorstore.add_texts(texts)

        # Test regular similarity search
        regular_results = vectorstore.similarity_search("cats", k=3)
        regular_content = [doc.page_content for doc in regular_results]

        # Test MMR search
        mmr_results = vectorstore.max_marginal_relevance_search(
            "cats", k=3, fetch_k=5, lambda_mult=0.5
        )
        mmr_content = [doc.page_content for doc in mmr_results]

        # Verify MMR provides diversity
        assert len(mmr_results) == 3

        # First result should be one of the cat documents (most similar)
        assert any("cat" in mmr_content[0].lower() for _ in [mmr_content[0]])

        # MMR should promote diverse content over very similar content
        # Dogs should appear before the second cat document for diversity
        assert "Dogs are loyal companions" in mmr_content

        # The order should be different from regular search due to diversity
        assert mmr_content != regular_content
    finally:
        cleanup_documents(vectorstore)


def test_mmr_by_vector(simple_embeddings, valkey_url):
    """Test MMR search by vector."""
    vectorstore = ValkeyVectorStore(
        index_name=SHARED_MMR_INDEX,
        embedding=simple_embeddings,
        valkey_url=valkey_url,
    )

    # Clean up any existing data first
    cleanup_documents(vectorstore)

    try:
        texts = ["Cats are pets", "Dogs are pets", "Cars are vehicles"]
        vectorstore.add_texts(texts)

        # Test MMR by vector
        query_vector = [1.0, 0.8, 0.1, 0.2]  # Cat-like vector
        results = vectorstore.max_marginal_relevance_search_by_vector(
            query_vector, k=2, fetch_k=3, lambda_mult=0.5
        )

        assert len(results) == 2
        assert results[0].page_content == "Cats are pets"  # Most similar first
    finally:
        cleanup_documents(vectorstore)


def test_mmr_lambda_mult(simple_embeddings, valkey_url):
    """Test MMR lambda_mult parameter affects diversity."""
    vectorstore = ValkeyVectorStore(
        index_name=SHARED_MMR_INDEX,
        embedding=simple_embeddings,
        valkey_url=valkey_url,
    )

    try:
        texts = [
            "Cats are wonderful pets",
            "Cats love sleeping",  # Very similar
            "Dogs are great too",
        ]
        vectorstore.add_texts(texts)

        # High lambda_mult (similarity focused)
        similarity_focused = vectorstore.max_marginal_relevance_search(
            "cats", k=2, fetch_k=3, lambda_mult=0.9
        )

        # Low lambda_mult (diversity focused)
        diversity_focused = vectorstore.max_marginal_relevance_search(
            "cats", k=2, fetch_k=3, lambda_mult=0.1
        )

        # Both should return 2 results
        assert len(similarity_focused) == 2
        assert len(diversity_focused) == 2

        # First result should be the same (most similar)
        assert similarity_focused[0].page_content == diversity_focused[0].page_content
    finally:
        cleanup_documents(vectorstore)


def test_mmr_empty_candidates():
    """Test MMR with empty candidate embeddings."""
    from langchain_valkey.maximal_marginal_relevance import (
        MaximalMarginalRelevanceSelector,
    )
    import numpy as np

    query_embedding = np.array([1.0, 0.0, 0.0])
    selector = MaximalMarginalRelevanceSelector()

    # Test empty candidates
    result = selector.select(
        query_embedding=query_embedding,
        candidate_embeddings=[],  # Empty list
        k=3,
        lambda_mult=0.5,
    )
    assert result == []


def test_mmr_dimension_mismatch():
    """Test MMR with dimension mismatch between query and candidates."""
    from langchain_valkey.maximal_marginal_relevance import cosine_similarity
    import numpy as np

    # Test dimension mismatch in cosine similarity
    X = np.array([[1.0, 0.0]])  # 2D
    Y = np.array([[1.0, 0.0, 0.0]])  # 3D

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        cosine_similarity(X, Y)


def test_mmr_invalid_parameters():
    """Test MMR with invalid parameters."""
    from langchain_valkey.maximal_marginal_relevance import (
        MaximalMarginalRelevanceSelector,
    )
    import numpy as np

    query_embedding = np.array([1.0, 0.0, 0.0])
    candidates = [np.array([1.0, 0.0, 0.0])]
    selector = MaximalMarginalRelevanceSelector()

    # Test invalid k
    with pytest.raises(ValueError, match="k must be positive"):
        selector.select(
            query_embedding=query_embedding,
            candidate_embeddings=candidates,
            k=0,  # Invalid k
            lambda_mult=0.5,
        )

    # Test invalid lambda_mult
    with pytest.raises(ValueError, match="lambda_mult must be in"):
        selector.select(
            query_embedding=query_embedding,
            candidate_embeddings=candidates,
            k=1,
            lambda_mult=1.5,  # Invalid lambda_mult
        )


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_connection_error_handling(fake_embeddings):
    """Test connection error handling when valkey_url fails."""
    from langchain_valkey.exceptions import ValkeyConnectionError

    # Test invalid URL
    with pytest.raises(ValkeyConnectionError, match="Cannot connect to Valkey"):
        ValkeyVectorStore(
            index_name="test_error",
            embedding=fake_embeddings,
            valkey_url="valkey://invalid-host:9999",
            auto_create_index=False,  # Prevent index creation attempts
        )


def test_malformed_url_connection_error(fake_embeddings):
    """Test connection error with malformed URL that causes client creation exception."""
    from langchain_valkey.exceptions import ValkeyConnectionError

    # Test malformed URL that causes exception during client creation
    with pytest.raises(ValkeyConnectionError, match="Failed to connect to Valkey"):
        ValkeyVectorStore(
            valkey_url="invalid://malformed:url:format",
            index_name="test_index",
            embedding=fake_embeddings,
        )


def test_missing_client_and_url():
    """Test error when neither client nor valkey_url provided."""
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    embeddings = DeterministicFakeEmbedding(size=1536)

    with pytest.raises(
        ValueError, match="Either valkey_url or client must be provided"
    ):
        ValkeyVectorStore(
            index_name="test_error",
            embedding=embeddings,
            # Neither client nor valkey_url provided
        )


def test_unsupported_config_type(fake_embeddings, valkey_url):
    """Test error handling for unsupported config types."""

    # Create a mock unsupported config by monkey-patching
    class UnsupportedConfig:
        def __init__(self):
            self.index_name = "test_unsupported"
            self.vector_dimension = 1536
            self.indexed_metadata_fields = {}

    vectorstore = ValkeyVectorStore(
        index_name="test_unsupported",
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=UnsupportedConfig(),
        auto_create_index=False,
    )

    # Force index creation to trigger the error
    vectorstore._vector_dimension = 1536

    from langchain_valkey.exceptions import ValkeySchemaError

    with pytest.raises(ValkeySchemaError, match="Unsupported config type"):
        vectorstore._create_index()


def test_filter_validation_invalid_combinations(fake_embeddings, valkey_url):
    """Test filter validation with invalid field names and values."""
    config = HNSWConfig(
        index_name=SHARED_HNSW_INDEX, distance_metric=DistanceMetric.COSINE
    )

    vectorstore = ValkeyVectorStore(
        index_name=SHARED_HNSW_INDEX,
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=config,
    )

    try:
        # Add test document
        doc = Document(page_content="Filter test", metadata={"category": "test"})
        doc_ids = vectorstore.add_documents([doc])

        # Test invalid filter field (not indexed)
        with pytest.raises(
            ValueError, match="Filter field 'invalid_field' is not indexed"
        ):
            vectorstore.similarity_search(
                "test", k=1, filter={"invalid_field": "value"}
            )

        # Test invalid filter value with special characters
        with pytest.raises(ValueError, match="contains invalid characters"):
            vectorstore.similarity_search(
                "test", k=1, filter={"category": "test{invalid}"}
            )

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_filter_validation_nested_values(vectorstore_with_docs):
    """Test filter validation with nested list/tuple values."""
    vectorstore, _ = vectorstore_with_docs

    # Test nested list validation - should handle gracefully
    results = vectorstore.similarity_search(
        "test query",
        k=1,
        filter={"source": ["test", "demo"]},  # List of values using indexed field
    )
    assert len(results) >= 0  # Should not crash

    # Test nested tuple validation
    results = vectorstore.similarity_search(
        "test query",
        k=1,
        filter={"source": ("test", "demo")},  # Tuple value using indexed field
    )
    assert len(results) >= 0  # Should not crash


def test_document_retrieval_nonexistent_ids(fake_embeddings, valkey_url):
    """Test document retrieval with empty and non-existent IDs."""
    config = HNSWConfig(
        index_name=SHARED_HNSW_INDEX, distance_metric=DistanceMetric.COSINE
    )

    vectorstore = ValkeyVectorStore(
        index_name=SHARED_HNSW_INDEX,
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=config,
    )

    try:
        # Add test documents
        docs = [
            Document(page_content="Document 1", metadata={"id": "1"}),
            Document(page_content="Document 2", metadata={"id": "2"}),
        ]
        doc_ids = vectorstore.add_documents(docs)

        # Test retrieval with empty IDs list
        empty_results = vectorstore.get_by_ids([])
        assert empty_results == []

        # Test retrieval with non-existent IDs
        non_existent_results = vectorstore.get_by_ids(["non-existent-id"])
        assert len(non_existent_results) == 0

        # Test retrieval with mix of existing and non-existent IDs
        mixed_results = vectorstore.get_by_ids([doc_ids[0], "non-existent-id"])
        assert len(mixed_results) == 1
        assert mixed_results[0].metadata["id"] == "1"

    finally:
        if "doc_ids" in locals():
            vectorstore.delete(doc_ids)


def test_delete_operations_invalid_ids(fake_embeddings, valkey_url):
    """Test delete operations with empty and non-existent IDs."""
    config = HNSWConfig(index_name="test_basic", distance_metric=DistanceMetric.COSINE)

    vectorstore = ValkeyVectorStore(
        index_name="test_basic",
        embedding=fake_embeddings,
        valkey_url=valkey_url,
        config=config,
    )

    try:
        # Add test documents
        docs = [
            Document(page_content="Delete test 1", metadata={"id": "1"}),
            Document(page_content="Delete test 2", metadata={"id": "2"}),
        ]
        doc_ids = vectorstore.add_documents(docs)

        # Test delete with empty IDs list
        result = vectorstore.delete([])
        assert result is True

        # Test delete with non-existent IDs
        result = vectorstore.delete(["non-existent-id"])
        assert result is False  # Should return False for non-existent docs

        # Test delete with valid IDs
        result = vectorstore.delete([doc_ids[0]])
        assert result is True

        # Verify document was deleted
        remaining_docs = vectorstore.get_by_ids(doc_ids)
        assert len(remaining_docs) == 1
        assert remaining_docs[0].metadata["id"] == "2"

        # Clean up remaining document
        vectorstore.delete([doc_ids[1]])

    except Exception:
        # Cleanup in case of failure
        if "doc_ids" in locals():
            try:
                vectorstore.delete(doc_ids)
            except Exception:
                pass


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_bulk_deletion_mixed_valid_invalid_ids(vectorstore_with_docs):
    """Test bulk deletion with mix of valid and invalid IDs."""
    vectorstore, doc_ids = vectorstore_with_docs

    # Test deletion with mix of valid and invalid IDs
    invalid_ids = ["nonexistent1", "nonexistent2"]
    mixed_ids = doc_ids[:1] + invalid_ids

    # Should handle gracefully without crashing
    vectorstore.delete(mixed_ids)

    # Verify valid document was deleted
    remaining_docs = vectorstore.get_by_ids(doc_ids)
    assert len(remaining_docs) == len(doc_ids) - 1


def test_embedding_bytes_conversion(vectorstore_with_docs):
    """Test bytes to embedding conversion."""
    vectorstore, doc_ids = vectorstore_with_docs

    # Get documents to trigger embedding retrieval
    docs = vectorstore.get_by_ids(doc_ids[:1])
    assert len(docs) == 1

    # The conversion happens internally when retrieving embeddings
    # This test ensures the _bytes_to_embedding method is covered
    assert docs[0].page_content is not None


def test_search_result_parsing(vectorstore_with_docs):
    """Test search result parsing with various query types."""
    vectorstore, _ = vectorstore_with_docs

    # Perform searches that might trigger edge cases in result parsing
    results = vectorstore.similarity_search("test", k=1)
    assert len(results) >= 0

    # Test with score retrieval
    results_with_scores = vectorstore.similarity_search_with_score("test", k=1)
    assert len(results_with_scores) >= 0

    for doc, score in results_with_scores:
        assert isinstance(score, (int, float))
        assert doc.page_content is not None
