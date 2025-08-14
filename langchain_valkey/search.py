"""Internal Valkey Search utilities.

This module provides UTF-8-safe search functionality that bypasses valkey-py's
Result class parsing which corrupts binary vector data through forced UTF-8 decoding.

Technical Background:
- valkey-py's Result.__init__() (valkey/commands/search/result.py:15-70) applies
  to_string() to all field values via map(to_string, res[i + fields_offset][1::2])
- to_string() (valkey/commands/search/_util.py:1-6) uses decode("utf-8", "ignore")
  which silently drops invalid UTF-8 bytes, corrupting binary float32 vector data
- All valkey-py search methods (ft().search(), ft().aggregate()) funnel through
  Result class, making binary data corruption unavoidable within valkey-py's API
- Raw execute_command("FT.SEARCH", ...) bypasses Result class entirely
"""

from typing import Any, Dict, List, Optional, Tuple

import valkey
from langchain_core.documents import Document


class ValkeySearchClient:
    """UTF-8-safe wrapper for Valkey Search operations.

    This class provides search functionality that preserves binary vector data
    by bypassing valkey-py's Result class which corrupts float32 vectors through
    inappropriate UTF-8 decoding with error suppression.

    Limitations of valkey-py search interface:
    - client.ft(index).search() → Result.__init__() → corrupts binary data
    - client.ft(index).aggregate() → Result.__init__() → corrupts binary data
    - No alternative parsing methods exist that preserve binary fields
    """

    def __init__(self, client: valkey.Valkey):
        """Initialize with Valkey client."""
        self.client = client

    def search_with_vectors(
        self,
        index_name: str,
        query: str,
        params: dict,
        include_vectors: bool = False,
    ) -> List[Tuple[Document, float, Optional[List[float]]]]:
        """Execute FT.SEARCH preserving binary vector data.

        Args:
            index_name: Name of the search index
            query: Search query string
            params: Query parameters
            include_vectors: Whether to include vector data in results

        Returns:
            List of (document, score, vector) tuples
        """
        # Build raw command parameters
        cmd_params = []
        for key, value in params.items():
            cmd_params.extend([key, value])

        raw_result = self.client.execute_command(
            "FT.SEARCH",
            index_name,
            query,
            "PARAMS",
            str(len(params) * 2),
            *cmd_params,
        )

        parsed_results = self._parse_raw_search_result(raw_result, include_vectors)
        return parsed_results

    def _parse_raw_search_result(
        self, raw_result: List, include_vectors: bool = False
    ) -> List[Tuple[str, Dict[str, Any], float, Optional[List[float]]]]:
        """Parse raw FT.SEARCH result - pure valkey-py compatible parser.

        This method is largely based on valkey-py's Result.__init__()
        (valkey/commands/search/result.py:15-70) but with key differences:
        - Preserves binary 'embedding' field data without UTF-8 decoding
        - Only applies UTF-8 decoding to text fields (page_content, metadata)
        - Avoids map(to_string, ...) which corrupts binary data via decode("utf-8", "ignore")

        Raw FT.SEARCH returns: [count, doc_id1, [field1, value1, field2, value2, ...], doc_id2, ...]

        Args:
            raw_result: Raw result from FT.SEARCH command
            include_vectors: Whether to parse vector data

        Returns:
            List of (doc_id, fields_dict, score, vector) tuples
        """
        parsed_results = []

        # Skip count (first element), process document pairs
        for i in range(1, len(raw_result), 2):
            if i + 1 >= len(raw_result):
                break

            doc_id = raw_result[i]
            if isinstance(doc_id, bytes):
                doc_id = doc_id.decode("utf-8")
            field_data = raw_result[i + 1]

            # Parse field data: [field_name, field_value, field_name, field_value, ...]
            fields = {}
            score = 0.0
            vector = None

            for j in range(0, len(field_data), 2):
                if j + 1 >= len(field_data):
                    break

                field_name = field_data[j]
                field_value = field_data[j + 1]

                # Handle both bytes and string field names for compatibility
                if isinstance(field_name, bytes):
                    field_name = field_name.decode("utf-8")

                if field_name == "vector_score":
                    score = float(field_value)
                elif field_name == "page_content":
                    fields["page_content"] = (
                        field_value.decode("utf-8")
                        if isinstance(field_value, bytes)
                        else str(field_value)
                    )
                elif field_name == "metadata":
                    fields["metadata"] = (
                        field_value.decode("utf-8")
                        if isinstance(field_value, bytes)
                        else str(field_value)
                    )
                elif (
                    field_name == "embedding"
                    and include_vectors
                    and isinstance(field_value, bytes)
                ):
                    # KEY DIFFERENCE: Binary vector data preserved - no UTF-8 corruption
                    import numpy as np

                    vector = np.frombuffer(field_value, dtype=np.float32).tolist()

            parsed_results.append((doc_id, fields, score, vector))

        return parsed_results
