"""Maximal Marginal Relevance (MMR) implementation for diverse search results.

This module implements the MMR algorithm to balance relevance and diversity in search results.
MMR addresses the problem where similarity search returns highly relevant but redundant documents
by selecting results that are both relevant to the query and diverse from each other.

Algorithm Overview:
MMR iteratively selects documents that maximize:
λ * sim(Di, Q) - (1-λ) * max(sim(Di, Dj)) for Dj in selected documents

Where:
- λ (lambda_mult): Controls relevance vs diversity tradeoff (0=max diversity, 1=max relevance)
- sim(Di, Q): Similarity between candidate document Di and query Q
- max(sim(Di, Dj)): Maximum similarity between Di and any already selected document Dj

References:
- Carbonell, J. and Goldstein, J. (1998). "The use of MMR, diversity-based reranking for 
  reordering documents and producing summaries." SIGIR '98.
- Implementation follows langchain_community.vectorstores.utils.maximal_marginal_relevance
  pattern for consistency with other LangChain vector stores.
"""

from typing import List
import numpy as np


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two matrices.

    Uses the mathematical definition: cos(θ) = (A·B) / (||A|| ||B||)
    Handles edge cases where vectors have zero magnitude.

    Implementation follows the same approach as scikit-learn's cosine_similarity
    and scipy.spatial.distance.cosine (1 - cosine_distance).

    Args:
        X: First matrix of shape (n_samples_X, n_features)
        Y: Second matrix of shape (n_samples_Y, n_features)

    Returns:
        Similarity matrix of shape (n_samples_X, n_samples_Y)

    Raises:
        ValueError: If matrices have different number of features
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]} features, "
            f"Y has {Y.shape[1]} features"
        )

    # Compute norms
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)

    # Compute cosine similarity, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)

    # Replace NaN and inf values with 0 (occurs when vectors have zero magnitude)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0

    return similarity


class MaximalMarginalRelevanceSelector:
    """Maximal Marginal Relevance selector for diverse document retrieval.

    This class implements the MMR algorithm to select a diverse subset of documents
    that balances relevance to a query with diversity among selected results.

    The algorithm works by:
    1. Starting with the most relevant document to the query
    2. Iteratively selecting documents that maximize the MMR score
    3. MMR score balances query relevance against similarity to already selected docs

    Example:
        selector = MaximalMarginalRelevanceSelector()
        indices = selector.select(
            query_embedding=query_vector,
            candidate_embeddings=document_vectors,
            k=5,
            lambda_mult=0.5
        )
    """

    def select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        """Select diverse documents using Maximal Marginal Relevance.

        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            candidate_embeddings: List of candidate document vectors
            k: Number of documents to select
            lambda_mult: Relevance vs diversity tradeoff parameter
                        0.0 = maximum diversity, 1.0 = maximum relevance

        Returns:
            List of indices of selected documents in order of selection

        Raises:
            ValueError: If k <= 0 or lambda_mult not in [0, 1]
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if not 0 <= lambda_mult <= 1:
            raise ValueError(f"lambda_mult must be in [0, 1], got {lambda_mult}")

        if not candidate_embeddings:
            return []

        # Ensure query is 2D for cosine similarity computation
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        # Convert to numpy array for efficient computation
        candidates = np.array(candidate_embeddings)

        # Compute similarity between query and all candidates
        query_similarities = cosine_similarity(query_embedding, candidates)[0]

        # Start with most similar document to query
        selected_indices = [int(np.argmax(query_similarities))]
        selected_embeddings = np.array([candidates[selected_indices[0]]])

        # Iteratively select remaining documents
        while len(selected_indices) < min(k, len(candidate_embeddings)):
            best_score = -np.inf
            best_idx = -1

            # Compute similarity between candidates and already selected documents
            candidate_to_selected_sim = cosine_similarity(
                candidates, selected_embeddings
            )

            # Find candidate that maximizes MMR score
            for i, query_sim in enumerate(query_similarities):
                if i in selected_indices:
                    continue

                # Maximum similarity to any selected document
                max_selected_sim = np.max(candidate_to_selected_sim[i])

                # MMR score: balance relevance and diversity
                mmr_score = (
                    lambda_mult * query_sim - (1 - lambda_mult) * max_selected_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx == -1:  # Safety check
                break

            selected_indices.append(best_idx)
            selected_embeddings = np.vstack([selected_embeddings, candidates[best_idx]])

        return selected_indices
