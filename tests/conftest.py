import os
import pytest
import valkey


@pytest.fixture(scope="session")
def valkey_url() -> str:
    return os.getenv("VALKEY_URL", "valkey://localhost:6379")


@pytest.fixture(scope="session", autouse=True)
def flush_valkey_before_tests(valkey_url):
    """Flush Valkey database before running integration tests."""
    client = valkey.from_url(valkey_url)
    client.flushall()
    yield
    # Optional: cleanup after all tests
    client.flushall()
