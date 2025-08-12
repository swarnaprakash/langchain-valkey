"""Standard LangChain interface tests for ValkeyVectorStore.

These tests ensure our ValkeyVectorStore properly implements the LangChain VectorStore interface
and behaves consistently with other LangChain vector store implementations.

For more information about LangChain standard tests, see:
https://github.com/langchain-ai/langchain/blob/master/docs/docs/contributing/how_to/integrations/standard_tests.mdx
"""

from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_valkey import ValkeyVectorStore


class TestValkeyVectorStoreStandard(VectorStoreIntegrationTests):
    """Standard LangChain integration tests for ValkeyVectorStore."""

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for testing."""
        # Use the standard test embeddings (small size for faster tests)
        embeddings = self.get_embeddings()

        store = ValkeyVectorStore(
            valkey_url="valkey://localhost:6379",
            index_name="test_standard_langchain",
            embedding=embeddings,
        )

        # Ensure store is empty at start of each test
        try:
            # Clean up any existing data
            keys = store.client.keys(f"{store.key_prefix}:*")
            if keys:
                store.client.delete(*keys)
        except Exception:
            pass

        # Initialize empty vectorstore (like postgres does with create_tables/create_collection)
        try:
            # Set dimension and create index to ensure vectorstore is properly initialized
            test_embedding = embeddings.embed_query("test")
            store._vector_dimension = len(test_embedding)
            store._create_index()
        except Exception:
            pass  # Index might already exist

        try:
            yield store
        finally:
            # Cleanup after test
            try:
                keys = store.client.keys(f"{store.key_prefix}:*")
                if keys:
                    store.client.delete(*keys)
            except Exception:
                pass
