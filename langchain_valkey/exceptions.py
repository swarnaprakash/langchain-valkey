"""Exception hierarchy for Valkey vector store."""


class ValkeyVectorStoreError(Exception):
    """Base exception for all Valkey vector store errors."""

    pass


class ValkeyConnectionError(ValkeyVectorStoreError):
    """Raised when connection to Valkey fails.

    This typically indicates:
    - Valkey server is not running
    - Network connectivity issues
    - Authentication failures
    - Invalid connection parameters

    Suggested actions:
    - Verify Valkey server is running and accessible
    - Check connection URL format and credentials
    - Ensure network connectivity to Valkey instance
    """

    pass


class ValkeySearchModuleError(ValkeyVectorStoreError):
    """Raised when Valkey search module is not available or configured incorrectly.

    This typically indicates:
    - Valkey search module is not loaded
    - Module version incompatibility
    - Search functionality is disabled

    Suggested actions:
    - Ensure Valkey is compiled with search module support
    - Load the search module: MODULE LOAD search.so
    - Verify module version compatibility
    """

    pass


class ValkeySchemaError(ValkeyVectorStoreError):
    """Raised when there are schema-related issues.

    This typically indicates:
    - Vector dimension mismatch
    - Index schema conflicts
    - Invalid field types or configurations

    Suggested actions:
    - Verify embedding dimensions match index configuration
    - Check for existing indexes with conflicting schemas
    - Ensure metadata field types are compatible
    """

    pass


class ValkeyQueryError(ValkeyVectorStoreError):
    """Raised when query execution fails.

    This typically indicates:
    - Invalid search syntax
    - Malformed filter expressions
    - Query parameter errors

    Suggested actions:
    - Verify query syntax and parameters
    - Check filter expressions for correct format
    - Ensure search parameters are within valid ranges
    """

    pass
