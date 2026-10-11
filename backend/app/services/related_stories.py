"""
NewsMap Related-Story Detection Service (Phase 5, Module 4).

Discovers and retrieves the top-k most similar news articles for a given source story
using cosine similarity over sparse TF-IDF feature vectors.

Architectural Guarantees:
1. Purely Unsupervised: Uses only TF-IDF document vectors. Ground-truth category labels
   are strictly never accessed, used for ranking, or evaluated during retrieval.
2. Sparse-Compatible Performance: Computes similarity via single-query sparse dot product
   against the corpus. Avoids creating any full N x N dense similarity matrix.
3. Article-ID Traceability: Preserves reliable 1-to-1 mapping between matrix rows and article IDs.
4. Source Exclusion: Explicitly excludes the source article from candidate recommendations.
5. Deterministic Tie-Breaking: Sorts by descending similarity score with article_id ascending
   as the deterministic tie-breaker.
6. Optional Cluster Context: Integrates Module 3 cluster assignments as supplementary metadata
   without altering cosine similarity ranking.
7. Robust Input Validation: Handles unknown IDs, zero-feature documents, and threshold filters.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.ml.input_validation import validate_corpus_for_tfidf
from backend.app.ml.feature_extraction import extract_tfidf_features, load_tfidf_vectorizer
from backend.app.services.data_loader import (
    get_project_root,
    load_processed_articles,
)

DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.0
DEFAULT_PREVIEW_LENGTH = 140


def compute_similarity_scores(
    query_vector: csr_matrix,
    corpus_matrix: csr_matrix,
) -> np.ndarray:
    """
    Compute cosine similarity scores between a single query vector and a sparse corpus matrix.

    Memory Efficiency:
        Only produces a 1D result array of length N (number of documents in corpus).
        Never constructs an N x N pairwise dense matrix.

    Args:
        query_vector: Sparse row vector of shape (1, n_features).
        corpus_matrix: Sparse matrix of shape (n_documents, n_features).

    Returns:
        1D numpy array of float cosine similarity scores of length n_documents.
    """
    if not issparse(query_vector) or not issparse(corpus_matrix):
        raise TypeError("Both query_vector and corpus_matrix must be scipy.sparse matrices.")

    if query_vector.shape[1] != corpus_matrix.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: query has {query_vector.shape[1]} features, "
            f"corpus has {corpus_matrix.shape[1]} features."
        )

    # If query vector has no non-zero features, cosine similarity is zero everywhere
    if query_vector.nnz == 0:
        return np.zeros(corpus_matrix.shape[0], dtype=np.float64)

    # sklearn cosine_similarity handles sparse CSR inputs efficiently
    sims = cosine_similarity(query_vector, corpus_matrix).ravel()

    # Clip numerical precision artifacts (e.g. -1e-16 or > 1.0)
    return np.clip(sims, 0.0, 1.0)


class RelatedStoryService:
    """
    Service for retrieving related news stories using TF-IDF cosine similarity.
    """

    def __init__(
        self,
        articles_df: Optional[pd.DataFrame] = None,
        tfidf_matrix: Optional[csr_matrix] = None,
        article_ids: Optional[List[int]] = None,
        cluster_assignments_df: Optional[pd.DataFrame] = None,
        project_root: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize RelatedStoryService with corpus data and TF-IDF representations.

        If arguments are omitted, data is automatically loaded and vectorized using
        the project's standard processed files and baseline TF-IDF configuration.

        Args:
            articles_df: Preprocessed DataFrame containing 'article_id' and 'cleaned_text'.
            tfidf_matrix: Sparse CSR matrix of TF-IDF features.
            article_ids: List of integer article IDs matching matrix rows 1-to-1.
            cluster_assignments_df: Optional DataFrame with 'article_id' and 'cluster_id'.
            project_root: Optional path to project root.
        """
        self.root = get_project_root(project_root)

        # 1. Load articles if not provided
        if articles_df is None:
            raw_articles = load_processed_articles(project_root=self.root)
            self.articles_df = validate_corpus_for_tfidf(raw_articles)
        else:
            self.articles_df = validate_corpus_for_tfidf(articles_df)

        # Standardize ID lookup DataFrame
        self.articles_df = self.articles_df.copy()
        self.articles_df["article_id"] = self.articles_df["article_id"].astype(int)
        self._articles_by_id = self.articles_df.set_index("article_id").to_dict(orient="index")

        # 2. Extract or assign TF-IDF matrix
        if tfidf_matrix is None or article_ids is None:
            _, matrix, meta = extract_tfidf_features(self.articles_df)
            self.tfidf_matrix = matrix
            self.article_ids = meta["article_ids"]
        else:
            if not issparse(tfidf_matrix):
                raise TypeError("tfidf_matrix must be a scipy.sparse matrix.")
            if not isinstance(tfidf_matrix, csr_matrix):
                tfidf_matrix = tfidf_matrix.tocsr()
            if len(article_ids) != tfidf_matrix.shape[0]:
                raise ValueError(
                    f"Mismatch between article_ids length ({len(article_ids)}) "
                    f"and tfidf_matrix rows ({tfidf_matrix.shape[0]})."
                )
            self.tfidf_matrix = tfidf_matrix
            self.article_ids = [int(aid) for aid in article_ids]

        # Build fast ID-to-matrix-row index
        self._id_to_row = {aid: idx for idx, aid in enumerate(self.article_ids)}

        # 3. Load optional cluster assignments if available
        self.cluster_map: Dict[int, int] = {}
        if cluster_assignments_df is not None:
            self._load_cluster_assignments_df(cluster_assignments_df)
        else:
            self._try_load_default_cluster_assignments()

    def _load_cluster_assignments_df(self, df: pd.DataFrame) -> None:
        """Internal helper to populate cluster map from DataFrame."""
        if "article_id" in df.columns and "cluster_id" in df.columns:
            clean_df = df.dropna(subset=["article_id", "cluster_id"])
            for _, row in clean_df.iterrows():
                try:
                    self.cluster_map[int(row["article_id"])] = int(row["cluster_id"])
                except (ValueError, TypeError):
                    continue

    def _try_load_default_cluster_assignments(self) -> None:
        """Attempt to load default cluster assignments CSV if present."""
        default_path = self.root / "backend" / "data" / "processed" / "cluster_assignments.csv"
        if default_path.exists():
            try:
                cdf = pd.read_csv(default_path)
                self._load_cluster_assignments_df(cdf)
            except Exception:
                self.cluster_map = {}

    def get_article_metadata(self, article_id: int) -> Dict[str, Any]:
        """
        Retrieve article metadata by article_id.

        Args:
            article_id: Integer article ID.

        Returns:
            Dictionary containing article text and metadata.

        Raises:
            KeyError: If article_id does not exist in the corpus.
        """
        aid = int(article_id)
        if aid not in self._articles_by_id:
            raise KeyError(f"Article ID {aid} was not found in the corpus.")
        meta = dict(self._articles_by_id[aid])
        meta["article_id"] = aid
        meta["cluster_id"] = self.cluster_map.get(aid)
        return meta

    def find_related_stories(
        self,
        source_article_id: int,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        preview_length: int = DEFAULT_PREVIEW_LENGTH,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar articles to a source article.

        Ranking & Exclusion Rules:
        - The source article itself is strictly excluded from recommendations.
        - Results are ranked strictly in descending order of cosine similarity score.
        - Articles with similarity scores below min_similarity are filtered out.
        - Articles with zero similarity (0.0 topical overlap) are excluded.
        - Deterministic tie-breaking orders ties by article_id ascending.
        - Cluster IDs are joined as supplementary metadata; cluster membership never alters ranking.

        Args:
            source_article_id: Integer ID of the source article.
            top_k: Maximum number of related articles to return (positive integer >= 1).
            min_similarity: Minimum cosine similarity threshold (float in [0.0, 1.0]).
            preview_length: Character length for text preview snippet.

        Returns:
            List of dictionaries containing rank, related_article_id, similarity_score,
            preview, word_count, source, cluster_id, and same_cluster flag.

        Raises:
            KeyError: If source_article_id is not in the corpus.
            ValueError: If top_k or min_similarity are out of valid bounds.
        """
        # 1. Parameter validations
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, but received {top_k}.")

        if not isinstance(min_similarity, (int, float)) or min_similarity < 0.0 or min_similarity > 1.0:
            raise ValueError(
                f"min_similarity must be a float between 0.0 and 1.0, but received {min_similarity}."
            )

        src_id = int(source_article_id)
        if src_id not in self._id_to_row:
            raise KeyError(
                f"Source article ID {src_id} not found in the TF-IDF feature matrix. "
                f"Ensure the ID belongs to the processed corpus."
            )

        source_row_idx = self._id_to_row[src_id]
        source_vector = self.tfidf_matrix[source_row_idx]
        source_cluster = self.cluster_map.get(src_id)

        # 2. Check for zero TF-IDF features
        if source_vector.nnz == 0:
            # Source article has no active vocabulary features; no meaningful similarity exists
            return []

        # 3. Compute cosine similarity against all documents in corpus
        sim_scores = compute_similarity_scores(source_vector, self.tfidf_matrix)

        # 4. Strictly exclude the source article itself
        sim_scores[source_row_idx] = -1.0

        # 5. Filter candidates: must meet min_similarity and have strictly positive similarity
        effective_min = max(float(min_similarity), 1e-9)
        candidate_indices = np.where(sim_scores >= effective_min)[0]

        if len(candidate_indices) == 0:
            return []

        # 6. Sort candidate indices: primary key = -similarity (descending), secondary key = article_id (ascending)
        candidate_article_ids = [self.article_ids[idx] for idx in candidate_indices]
        candidate_scores = sim_scores[candidate_indices]

        # Use structured sorting for guaranteed deterministic tie-breaking
        sort_keys = [(-score, aid) for score, aid in zip(candidate_scores, candidate_article_ids)]
        sorted_order = sorted(range(len(candidate_indices)), key=lambda i: sort_keys[i])

        top_indices = [candidate_indices[i] for i in sorted_order[:top_k]]

        # 7. Build structured result records
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            related_id = self.article_ids[idx]
            score = float(round(float(sim_scores[idx]), 4))

            meta = self._articles_by_id.get(related_id, {})
            cleaned_text = str(meta.get("cleaned_text", "")).strip()

            preview = cleaned_text[:preview_length]
            if len(cleaned_text) > preview_length:
                preview = preview + "..."

            related_cluster = self.cluster_map.get(related_id)
            same_cluster = (
                (related_cluster == source_cluster)
                if (related_cluster is not None and source_cluster is not None)
                else None
            )

            result_item: Dict[str, Any] = {
                "rank": rank,
                "source_article_id": src_id,
                "related_article_id": related_id,
                "similarity_score": score,
                "preview": preview,
                "word_count": meta.get("word_count"),
                "source": meta.get("source"),
                "cluster_id": related_cluster,
                "same_cluster": same_cluster,
            }
            results.append(result_item)

        return results


def get_related_stories(
    source_article_id: int,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    articles_df: Optional[pd.DataFrame] = None,
    tfidf_matrix: Optional[csr_matrix] = None,
    article_ids: Optional[List[int]] = None,
    cluster_assignments_df: Optional[pd.DataFrame] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience functional API to retrieve related stories for an article ID.

    Args:
        source_article_id: Integer ID of the source article.
        top_k: Number of related stories to return. Defaults to 5.
        min_similarity: Minimum cosine similarity threshold. Defaults to 0.0.
        articles_df: Optional corpus DataFrame.
        tfidf_matrix: Optional sparse TF-IDF feature matrix.
        article_ids: Optional list of integer IDs aligned with matrix rows.
        cluster_assignments_df: Optional cluster assignments DataFrame.
        project_root: Optional project root path.

    Returns:
        List of related story dictionaries.
    """
    service = RelatedStoryService(
        articles_df=articles_df,
        tfidf_matrix=tfidf_matrix,
        article_ids=article_ids,
        cluster_assignments_df=cluster_assignments_df,
        project_root=project_root,
    )
    return service.find_related_stories(
        source_article_id=source_article_id,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def format_related_stories_report(
    source_article_id: int,
    results: List[Dict[str, Any]],
    source_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format related-story retrieval results as a readable markdown/text report.

    Args:
        source_article_id: Integer ID of the source story.
        results: List of related story records.
        source_meta: Optional metadata dict for the source story.

    Returns:
        Formatted string report.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"NewsMap Related-Story Detection Report: Source Article ID {source_article_id}")
    lines.append("=" * 80)

    if source_meta:
        src_cluster = source_meta.get("cluster_id")
        src_words = source_meta.get("word_count", "N/A")
        src_preview = str(source_meta.get("cleaned_text", "")).strip()[:100]
        cluster_str = f"Cluster {src_cluster}" if src_cluster is not None else "Unassigned"
        lines.append(f"Source Metadata : [{cluster_str}] {src_words} words")
        lines.append(f"Source Preview  : \"{src_preview}...\"")
        lines.append("-" * 80)

    if not results:
        lines.append("No related stories found meeting the similarity threshold.")
        lines.append("=" * 80)
        return "\n".join(lines)

    lines.append(f"Retrieved {len(results)} related stories (sorted by descending cosine similarity):\n")

    for r in results:
        rank = r["rank"]
        rel_id = r["related_article_id"]
        sim = r["similarity_score"]
        cluster_id = r.get("cluster_id")
        same_c = r.get("same_cluster")

        cluster_tag = f"Cluster {cluster_id}" if cluster_id is not None else "No Cluster"
        same_tag = " (Same Topic Cluster)" if same_c else (" (Cross-Cluster)" if same_c is False else "")
        preview = r.get("preview", "")

        lines.append(f"  [Rank {rank}] Article ID {rel_id:<4} | Similarity: {sim:.4f} | [{cluster_tag}{same_tag}]")
        lines.append(f"         Preview: \"{preview}\"")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    """Demonstrate Module 4 Related-Story Retrieval on the real BBC News corpus."""
    print("Initializing NewsMap Related-Story Service on real corpus...")
    service = RelatedStoryService()

    # Query demonstration: WorldCom Trial article (ID 1833)
    demo_ids = [1833, 1500]

    for aid in demo_ids:
        source_meta = service.get_article_metadata(aid)
        results = service.find_related_stories(source_article_id=aid, top_k=5)
        report = format_related_stories_report(aid, results, source_meta)
        print("\n" + report)


if __name__ == "__main__":
    main()
