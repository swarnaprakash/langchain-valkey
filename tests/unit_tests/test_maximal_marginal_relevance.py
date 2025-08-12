"""Unit tests for MaximalMarginalRelevanceSelector and cosine similarity."""

import numpy as np
import pytest

from langchain_valkey.maximal_marginal_relevance import (
    cosine_similarity,
    MaximalMarginalRelevanceSelector,
)


def test_cosine_similarity_mathematical_correctness():
    """Test cosine similarity produces mathematically correct results."""
    # Test orthogonal vectors (should be 0)
    X = np.array([[1.0, 0.0, 0.0]])
    Y = np.array([[0.0, 1.0, 0.0]])
    result = cosine_similarity(X, Y)
    assert abs(result[0, 0] - 0.0) < 1e-10

    # Test identical vectors (should be 1)
    X = np.array([[1.0, 0.0, 0.0]])
    Y = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity(X, Y)
    assert abs(result[0, 0] - 1.0) < 1e-10

    # Test opposite vectors (should be -1)
    X = np.array([[1.0, 0.0, 0.0]])
    Y = np.array([[-1.0, 0.0, 0.0]])
    result = cosine_similarity(X, Y)
    assert abs(result[0, 0] - (-1.0)) < 1e-10


def test_cosine_similarity_handles_zero_vectors():
    """Test cosine similarity handles zero vectors gracefully."""
    # Zero vector vs normal vector should be 0
    X = np.array([[0.0, 0.0, 0.0]])
    Y = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity(X, Y)
    assert result[0, 0] == 0.0

    # Zero vector vs zero vector should be 0
    X = np.array([[0.0, 0.0, 0.0]])
    Y = np.array([[0.0, 0.0, 0.0]])
    result = cosine_similarity(X, Y)
    assert result[0, 0] == 0.0


def test_cosine_similarity_edge_cases():
    """Test cosine similarity edge cases."""
    # Empty arrays
    X = np.array([]).reshape(0, 3)
    Y = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity(X, Y)
    assert result.size == 0

    # Dimension mismatch
    X = np.array([[1.0, 0.0]])  # 2D
    Y = np.array([[1.0, 0.0, 0.0]])  # 3D
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        cosine_similarity(X, Y)


def test_mmr_selector_basic_functionality():
    """Test MMR selector basic selection logic."""
    selector = MaximalMarginalRelevanceSelector()

    # Simple test case: query similar to first candidate
    query = np.array([1.0, 0.0, 0.0])
    candidates = [
        np.array([1.0, 0.0, 0.0]),  # Most similar to query
        np.array([0.0, 1.0, 0.0]),  # Orthogonal to query
        np.array([0.5, 0.5, 0.0]),  # Moderate similarity
    ]

    # With lambda=1.0 (pure relevance), should select most similar first
    indices = selector.select(query, candidates, k=2, lambda_mult=1.0)
    assert indices[0] == 0  # Most similar to query selected first
    assert len(indices) == 2


def test_mmr_selector_diversity_vs_relevance():
    """Test MMR selector balances diversity and relevance."""
    selector = MaximalMarginalRelevanceSelector()

    query = np.array([1.0, 0.0])
    candidates = [
        np.array([1.0, 0.0]),  # Most similar to query
        np.array([0.9, 0.1]),  # Very similar to query and first candidate
        np.array([0.0, 1.0]),  # Orthogonal to query (diverse)
    ]

    # With high diversity (lambda=0.1), should prefer diverse documents
    indices = selector.select(query, candidates, k=2, lambda_mult=0.1)
    assert indices[0] == 0  # Still starts with most relevant
    assert indices[1] == 2  # Should pick diverse document over similar one


def test_mmr_selector_edge_cases():
    """Test MMR selector handles edge cases."""
    selector = MaximalMarginalRelevanceSelector()

    # Empty candidates
    assert selector.select(np.array([1.0, 0.0]), [], k=5) == []

    # k larger than candidates
    query = np.array([1.0, 0.0])
    candidates = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    indices = selector.select(query, candidates, k=10)
    assert len(indices) == 2  # Should return all candidates


def test_mmr_selector_safety_break():
    """Test MMR selector safety break when no valid candidates found."""
    selector = MaximalMarginalRelevanceSelector()

    # Create a scenario where all candidates are identical (edge case)
    query = np.array([1.0, 0.0])
    candidates = [
        np.array([1.0, 0.0]),  # Identical to query
        np.array([1.0, 0.0]),  # Identical to query
        np.array([1.0, 0.0]),  # Identical to query
    ]

    # Should still work and return indices
    indices = selector.select(query, candidates, k=3)
    assert len(indices) <= 3
    assert all(isinstance(idx, int) for idx in indices)


def test_mmr_selector_parameter_validation():
    """Test MMR selector validates parameters."""
    selector = MaximalMarginalRelevanceSelector()
    query = np.array([1.0, 0.0])
    candidates = [np.array([1.0, 0.0])]

    # Invalid k (zero)
    with pytest.raises(ValueError, match="k must be positive"):
        selector.select(query, candidates, k=0)

    # Invalid k (negative)
    with pytest.raises(ValueError, match="k must be positive"):
        selector.select(query, candidates, k=-1)

    # Invalid lambda_mult
    with pytest.raises(ValueError, match="lambda_mult must be in"):
        selector.select(query, candidates, k=1, lambda_mult=1.5)

    with pytest.raises(ValueError, match="lambda_mult must be in"):
        selector.select(query, candidates, k=1, lambda_mult=-0.1)
