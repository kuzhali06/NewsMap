"""
NewsMap K-Means Topic Clustering Module (Phase 5, Module 3).

Discovers latent topics across news articles using K-Means clustering on sparse TF-IDF features.

Architectural Guarantees:
1. Purely Unsupervised: Model fitting and topic discovery never access or use ground-truth category labels.
2. Sparse Matrix Preservation: Operates directly on scipy.sparse.csr_matrix. No dense array conversion.
3. Strict Article ID Mapping: Every row in the matrix and its assigned cluster strictly maps to its article_id.
4. Deterministic Baseline: Default settings (n_clusters=5, random_state=42, n_init=10) guarantee reproducible results.
5. Interpretable Topic Summaries: Extracts top centroid terms and closest representative articles via sparse distance.
6. Decoupled Post-Hoc Evaluation: Evaluates clustering against external evaluation labels without coupling to training.
7. Safe Model Persistence: Persists model and outputs using standard joblib/JSON/CSV formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

from backend.app.ml.input_validation import validate_corpus_for_tfidf, validate_evaluation_labels
from backend.app.ml.feature_extraction import extract_tfidf_features
from backend.app.services.data_loader import (
    get_project_root,
    load_processed_articles,
    load_processed_evaluation_labels,
)

# Baseline K-Means Hyperparameters
DEFAULT_N_CLUSTERS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10
DEFAULT_MAX_ITER = 300
DEFAULT_TOL = 1e-4
DEFAULT_TOP_TERMS_COUNT = 10
DEFAULT_TOP_REPRESENTATIVE_COUNT = 3


def fit_kmeans(
    tfidf_matrix: csr_matrix,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: Optional[int] = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> KMeans:
    """
    Fit K-Means clustering model to a sparse TF-IDF matrix.

    Args:
        tfidf_matrix: Sparse matrix of shape (n_samples, n_features).
        n_clusters: Number of clusters to form. Defaults to 5.
        random_state: Random seed for deterministic centroid initialization. Defaults to 42.
        n_init: Number of centroid initializations to run. Defaults to 10.
        max_iter: Maximum number of iterations per run. Defaults to 300.
        tol: Relative tolerance with respect to inertia to declare convergence. Defaults to 1e-4.

    Returns:
        Fitted sklearn.cluster.KMeans instance.

    Raises:
        TypeError: If input matrix is not a sparse matrix.
        ValueError: If matrix is empty, n_clusters is invalid, or hyperparameters are out of bounds.
    """
    if not issparse(tfidf_matrix):
        raise TypeError(
            f"Expected a scipy.sparse matrix for tfidf_matrix, but received '{type(tfidf_matrix).__name__}'. "
            f"Sparse matrix input is required to avoid memory exhaustion."
        )

    if not isinstance(tfidf_matrix, csr_matrix):
        tfidf_matrix = tfidf_matrix.tocsr()

    n_samples, n_features = tfidf_matrix.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError(
            f"TF-IDF matrix is empty with shape ({n_samples}, {n_features}). "
            f"At least one sample and one feature are required for clustering."
        )

    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError(
            f"n_clusters must be a positive integer >= 1, but received {n_clusters}."
        )

    if n_clusters > n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed the number of input samples ({n_samples})."
        )

    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(
            f"max_iter must be a positive integer >= 1, but received {max_iter}."
        )

    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError(
            f"tol must be a non-negative float, but received {tol}."
        )

    if not isinstance(n_init, int) or n_init < 1:
        raise ValueError(
            f"n_init must be a positive integer >= 1, but received {n_init}."
        )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )

    kmeans.fit(tfidf_matrix)
    return kmeans


def predict_cluster_assignments(
    kmeans_model: KMeans,
    tfidf_matrix: csr_matrix,
    article_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Predict cluster assignments for a sparse TF-IDF matrix and align each row with its article_id.

    Args:
        kmeans_model: Fitted KMeans model.
        tfidf_matrix: Sparse matrix of shape (n_samples, n_features).
        article_ids: Optional list of integer article IDs corresponding 1-to-1 with matrix rows.

    Returns:
        pd.DataFrame with columns ['article_id', 'cluster_id'].

    Raises:
        TypeError: If inputs are invalid types.
        ValueError: If dimensions do not match or article_ids length differs from matrix rows.
    """
    if not hasattr(kmeans_model, "cluster_centers_"):
        raise ValueError("Provided KMeans model is not fitted.")

    if not issparse(tfidf_matrix):
        raise TypeError(
            f"Expected a scipy.sparse matrix for tfidf_matrix, received '{type(tfidf_matrix).__name__}'."
        )

    if not isinstance(tfidf_matrix, csr_matrix):
        tfidf_matrix = tfidf_matrix.tocsr()

    n_samples, n_features = tfidf_matrix.shape
    if n_features != kmeans_model.cluster_centers_.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: matrix has {n_features} features, but "
            f"KMeans model expects {kmeans_model.cluster_centers_.shape[1]} features."
        )

    cluster_labels = kmeans_model.predict(tfidf_matrix)

    if article_ids is not None:
        if len(article_ids) != n_samples:
            raise ValueError(
                f"Length of article_ids ({len(article_ids)}) does not match "
                f"number of samples in TF-IDF matrix ({n_samples})."
            )
        resolved_ids = [int(aid) for aid in article_ids]
    else:
        resolved_ids = list(range(n_samples))

    assignments_df = pd.DataFrame(
        {
            "article_id": resolved_ids,
            "cluster_id": [int(label) for label in cluster_labels],
        }
    )
    return assignments_df


def get_cluster_centroid_top_terms(
    kmeans_model: KMeans,
    feature_names: List[str],
    top_n: int = DEFAULT_TOP_TERMS_COUNT,
) -> Dict[int, List[Dict[str, Union[str, float]]]]:
    """
    Extract top weighted vocabulary terms for each cluster centroid in descending weight order.

    Args:
        kmeans_model: Fitted KMeans model.
        feature_names: List of vocabulary terms matching feature columns.
        top_n: Number of top terms to return per cluster. Defaults to 10.

    Returns:
        Dictionary mapping cluster_id (int) -> list of dicts [{'term': str, 'weight': float}].

    Raises:
        ValueError: If model is not fitted or feature_names length does not match centroid features.
    """
    if not hasattr(kmeans_model, "cluster_centers_"):
        raise ValueError("Provided KMeans model is not fitted.")

    centroids = kmeans_model.cluster_centers_
    n_clusters, n_features = centroids.shape

    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature_names ({len(feature_names)}) does not match "
            f"model centroid feature dimensions ({n_features})."
        )

    if top_n <= 0:
        raise ValueError(f"top_n must be a positive integer >= 1, but received {top_n}.")

    top_n = min(top_n, n_features)
    cluster_top_terms: Dict[int, List[Dict[str, Union[str, float]]]] = {}

    for cluster_id in range(n_clusters):
        centroid_weights = centroids[cluster_id]
        top_indices = np.argsort(centroid_weights)[::-1][:top_n]
        terms_list: List[Dict[str, Union[str, float]]] = [
            {
                "term": feature_names[idx],
                "weight": float(round(centroid_weights[idx], 6)),
            }
            for idx in top_indices
        ]
        cluster_top_terms[cluster_id] = terms_list

    return cluster_top_terms


def get_representative_articles(
    tfidf_matrix: csr_matrix,
    cluster_labels: np.ndarray,
    kmeans_model: KMeans,
    article_ids: List[int],
    top_k: int = DEFAULT_TOP_REPRESENTATIVE_COUNT,
    texts: Optional[List[str]] = None,
    max_snippet_len: int = 150,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Select representative articles closest to each cluster centroid using sparse Euclidean distance.

    Mathematical Method:
        For a cluster centroid c_k and article vector x_i, Euclidean distance squared is:
            ||x_i - c_k||^2 = ||x_i||^2 + ||c_k||^2 - 2 * (x_i . c_k)
        Because TF-IDF rows are L2 normalized, ||x_i||^2 = 1.0.
        For general sparse matrices, row norms and dot products are computed strictly using
        sparse operations (X_cluster.dot(c_k) and sparse row sum of squares) WITHOUT densifying
        the full corpus matrix.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix (csr_matrix).
        cluster_labels: Array of assigned cluster IDs for each row.
        kmeans_model: Fitted KMeans model with cluster_centers_.
        article_ids: List of article IDs aligned with matrix rows.
        top_k: Number of representative articles per cluster. Defaults to 3.
        texts: Optional list of raw/cleaned text strings for creating descriptive snippets.
        max_snippet_len: Maximum length of text snippet preview.

    Returns:
        Dictionary mapping cluster_id (int) -> list of representative article metadata dicts:
        [{'article_id': int, 'distance_to_centroid': float, 'snippet': Optional[str]}].
    """
    if not hasattr(kmeans_model, "cluster_centers_"):
        raise ValueError("Provided KMeans model is not fitted.")

    if not isinstance(tfidf_matrix, csr_matrix):
        tfidf_matrix = tfidf_matrix.tocsr()

    n_samples, n_features = tfidf_matrix.shape
    if len(article_ids) != n_samples or len(cluster_labels) != n_samples:
        raise ValueError("Dimension mismatch between tfidf_matrix, article_ids, and cluster_labels.")

    centroids = kmeans_model.cluster_centers_
    n_clusters = centroids.shape[0]

    representative_dict: Dict[int, List[Dict[str, Any]]] = {}

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_row_indices = np.where(cluster_mask)[0]

        if len(cluster_row_indices) == 0:
            representative_dict[cluster_id] = []
            continue

        # Extract only the submatrix of rows belonging to this cluster (remains sparse CSR)
        X_sub = tfidf_matrix[cluster_row_indices]
        centroid = centroids[cluster_id]

        # Sparse row squared L2 norms
        row_sq_norms = np.asarray(X_sub.multiply(X_sub).sum(axis=1)).ravel()
        centroid_sq_norm = float(np.sum(centroid ** 2))

        # Sparse matrix-vector dot product (X_sub . centroid)
        dot_products = np.asarray(X_sub.dot(centroid)).ravel()

        # Euclidean distance squared with non-negative clipping for numerical stability
        sq_distances = np.maximum(row_sq_norms + centroid_sq_norm - (2.0 * dot_products), 0.0)
        distances = np.sqrt(sq_distances)

        # Sort ascending by distance (closest to centroid first)
        k_to_pick = min(top_k, len(cluster_row_indices))
        sorted_local_indices = np.argsort(distances)[:k_to_pick]

        cluster_reps: List[Dict[str, Any]] = []
        for local_idx in sorted_local_indices:
            global_idx = cluster_row_indices[local_idx]
            art_id = int(article_ids[global_idx])
            dist_val = float(round(distances[local_idx], 6))

            snippet: Optional[str] = None
            if texts is not None and global_idx < len(texts):
                raw_text = str(texts[global_idx]).strip()
                snippet = raw_text[:max_snippet_len] + ("..." if len(raw_text) > max_snippet_len else "")

            rep_item: Dict[str, Any] = {
                "article_id": art_id,
                "distance_to_centroid": dist_val,
            }
            if snippet is not None:
                rep_item["snippet"] = snippet

            cluster_reps.append(rep_item)

        representative_dict[cluster_id] = cluster_reps

    return representative_dict


def generate_cluster_summaries(
    kmeans_model: KMeans,
    tfidf_matrix: csr_matrix,
    assignments_df: pd.DataFrame,
    feature_names: List[str],
    texts: Optional[List[str]] = None,
    top_terms_count: int = DEFAULT_TOP_TERMS_COUNT,
    top_representative_count: int = DEFAULT_TOP_REPRESENTATIVE_COUNT,
) -> Dict[str, Any]:
    """
    Generate comprehensive, structured topic summaries for all clusters.

    Topic summaries are purely data-driven:
    - Cluster IDs are neutral integers (0, 1, 2, ...).
    - Top terms are ordered by centroid weight.
    - Representative articles are chosen by minimum distance to cluster centroid.

    Args:
        kmeans_model: Fitted KMeans model.
        tfidf_matrix: Sparse TF-IDF matrix.
        assignments_df: DataFrame containing 'article_id' and 'cluster_id'.
        feature_names: List of vocabulary terms.
        texts: Optional list of article texts for snippet preview.
        top_terms_count: Number of top terms per cluster.
        top_representative_count: Number of representative articles per cluster.

    Returns:
        Structured dictionary detailing corpus clustering results and cluster summaries.
    """
    total_articles = len(assignments_df)
    n_clusters = kmeans_model.n_clusters
    cluster_labels = assignments_df["cluster_id"].to_numpy()
    article_ids = assignments_df["article_id"].tolist()

    top_terms_map = get_cluster_centroid_top_terms(
        kmeans_model, feature_names=feature_names, top_n=top_terms_count
    )

    representative_map = get_representative_articles(
        tfidf_matrix=tfidf_matrix,
        cluster_labels=cluster_labels,
        kmeans_model=kmeans_model,
        article_ids=article_ids,
        top_k=top_representative_count,
        texts=texts,
    )

    cluster_counts = assignments_df["cluster_id"].value_counts().to_dict()

    clusters_list: List[Dict[str, Any]] = []
    for cluster_id in range(n_clusters):
        count = int(cluster_counts.get(cluster_id, 0))
        percentage = float(round((count / total_articles) * 100.0, 2)) if total_articles > 0 else 0.0

        clusters_list.append(
            {
                "cluster_id": cluster_id,
                "article_count": count,
                "percentage_of_corpus": percentage,
                "top_terms": top_terms_map.get(cluster_id, []),
                "representative_articles": representative_map.get(cluster_id, []),
            }
        )

    summary_output: Dict[str, Any] = {
        "metadata": {
            "algorithm": "KMeans",
            "n_clusters": n_clusters,
            "total_articles": total_articles,
            "n_features": int(kmeans_model.cluster_centers_.shape[1]),
            "inertia": float(round(float(kmeans_model.inertia_), 4)),
            "n_iter": int(kmeans_model.n_iter_),
            "random_state": kmeans_model.random_state,
        },
        "clusters": clusters_list,
    }

    return summary_output


def evaluate_clusters_against_labels(
    assignments_df: pd.DataFrame,
    evaluation_labels_df: pd.DataFrame,
    article_id_col: str = "article_id",
    cluster_col: str = "cluster_id",
    label_col: str = "category",
) -> Dict[str, Any]:
    """
    Conduct post-training external evaluation by joining cluster assignments with evaluation labels.

    IMPORTANT ARCHITECTURAL NOTICE:
    Ground-truth category labels are strictly post-training diagnostic artifacts.
    They are NEVER used during K-Means fitting, feature selection, or topic naming.

    Args:
        assignments_df: DataFrame with article_id and cluster_id.
        evaluation_labels_df: DataFrame with article_id and ground-truth category.
        article_id_col: Column name for article ID. Defaults to 'article_id'.
        cluster_col: Column name for cluster assignment. Defaults to 'cluster_id'.
        label_col: Column name for evaluation label. Defaults to 'category'.

    Returns:
        Dictionary containing match statistics, external metrics (ARI, NMI, Homogeneity,
        Completeness, V-Measure), and cluster-to-category contingency breakdown.

    Raises:
        ValueError: If required columns are missing or no matching articles are found.
    """
    if article_id_col not in assignments_df.columns or cluster_col not in assignments_df.columns:
        raise ValueError(
            f"assignments_df must contain '{article_id_col}' and '{cluster_col}' columns."
        )

    if article_id_col not in evaluation_labels_df.columns or label_col not in evaluation_labels_df.columns:
        raise ValueError(
            f"evaluation_labels_df must contain '{article_id_col}' and '{label_col}' columns."
        )

    # Standardize ID types for join
    assign_clean = assignments_df[[article_id_col, cluster_col]].copy()
    assign_clean[article_id_col] = assign_clean[article_id_col].astype(int)

    labels_clean = evaluation_labels_df[[article_id_col, label_col]].copy()
    labels_clean[article_id_col] = labels_clean[article_id_col].astype(int)

    merged = pd.merge(assign_clean, labels_clean, on=article_id_col, how="inner")

    total_assignments = len(assign_clean)
    total_labels = len(labels_clean)
    matched_count = len(merged)
    unmatched_assignments_count = total_assignments - matched_count
    unmatched_labels_count = total_labels - matched_count

    if matched_count == 0:
        raise ValueError(
            "No matching article IDs found between cluster assignments and evaluation labels."
        )

    y_true = merged[label_col].to_numpy()
    y_pred = merged[cluster_col].to_numpy()

    # Calculate standard external clustering evaluation metrics
    ari_score = float(round(float(adjusted_rand_score(y_true, y_pred)), 4))
    nmi_score = float(round(float(normalized_mutual_info_score(y_true, y_pred)), 4))
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)

    # Build contingency table (clusters x categories)
    contingency_df = pd.crosstab(merged[cluster_col], merged[label_col])
    contingency_dict = contingency_df.to_dict(orient="index")

    # Identify dominant category for each cluster for transparency
    cluster_dominance: Dict[int, Dict[str, Any]] = {}
    for cluster_id, row in contingency_df.iterrows():
        dominant_cat = str(row.idxmax())
        dominant_count = int(row.max())
        cluster_total = int(row.sum())
        purity = float(round((dominant_count / cluster_total) * 100.0, 2)) if cluster_total > 0 else 0.0
        cluster_dominance[int(cluster_id)] = {
            "dominant_category": dominant_cat,
            "dominant_count": dominant_count,
            "cluster_evaluated_total": cluster_total,
            "purity_percentage": purity,
        }

    return {
        "notice": (
            "Post-training diagnostic evaluation only. Ground-truth labels were "
            "never used during model fitting or topic discovery."
        ),
        "coverage": {
            "total_assignments": total_assignments,
            "total_evaluation_labels": total_labels,
            "matched_articles": matched_count,
            "unmatched_assignments": unmatched_assignments_count,
            "unmatched_evaluation_labels": unmatched_labels_count,
            "alignment_ratio": float(round(matched_count / total_assignments, 4)),
        },
        "metrics": {
            "adjusted_rand_index": ari_score,
            "normalized_mutual_info": nmi_score,
            "homogeneity": float(round(float(homogeneity), 4)),
            "completeness": float(round(float(completeness), 4)),
            "v_measure": float(round(float(v_measure), 4)),
        },
        "cluster_dominance": cluster_dominance,
        "contingency_table": contingency_dict,
    }


def save_kmeans_model(
    kmeans_model: KMeans,
    output_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Persist the fitted KMeans model to disk using joblib.

    Security Notice:
        joblib uses pickle under the hood. Only unpickle/load model files from trusted sources.

    Args:
        kmeans_model: Fitted KMeans model.
        output_path: Optional path to save model file. Defaults to backend/models/kmeans_model.joblib.
        project_root: Optional project root path.

    Returns:
        Path to the saved model file.
    """
    if not hasattr(kmeans_model, "cluster_centers_"):
        raise ValueError("Cannot save an unfitted KMeans model.")

    if output_path is None:
        root = get_project_root(project_root)
        save_file = root / "backend" / "models" / "kmeans_model.joblib"
    else:
        save_file = Path(output_path).resolve()

    save_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, save_file)
    return save_file


def load_kmeans_model(
    model_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> KMeans:
    """
    Load a fitted KMeans model from disk.

    Security Notice:
        Only load trusted model artifacts. Deserializing untrusted joblib files can execute
        arbitrary code.

    Args:
        model_path: Path to the joblib model file.
        project_root: Optional project root path.

    Returns:
        Fitted KMeans instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the loaded object is not a fitted KMeans model.
    """
    if model_path is None:
        root = get_project_root(project_root)
        resolved_path = root / "backend" / "models" / "kmeans_model.joblib"
    else:
        resolved_path = Path(model_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"KMeans model file not found at: '{resolved_path}'.")

    model = joblib.load(resolved_path)

    if not isinstance(model, KMeans) or not hasattr(model, "cluster_centers_"):
        raise ValueError(f"Object loaded from '{resolved_path}' is not a valid fitted KMeans model.")

    return model


def save_cluster_assignments(
    assignments_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save cluster assignments DataFrame to CSV.

    Args:
        assignments_df: DataFrame with article_id and cluster_id.
        output_path: Optional target path. Defaults to backend/data/processed/cluster_assignments.csv.
        project_root: Optional project root path.

    Returns:
        Path to saved CSV file.
    """
    if output_path is None:
        root = get_project_root(project_root)
        file_path = root / "backend" / "data" / "processed" / "cluster_assignments.csv"
    else:
        file_path = Path(output_path).resolve()

    file_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(file_path, index=False, encoding="utf-8")
    return file_path


def save_cluster_summaries(
    summaries_dict: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save cluster summaries dictionary to JSON.

    Args:
        summaries_dict: Dictionary containing metadata and cluster descriptions.
        output_path: Optional target path. Defaults to backend/data/processed/cluster_summaries.json.
        project_root: Optional project root path.

    Returns:
        Path to saved JSON file.
    """
    if output_path is None:
        root = get_project_root(project_root)
        file_path = root / "backend" / "data" / "processed" / "cluster_summaries.json"
    else:
        file_path = Path(output_path).resolve()

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summaries_dict, f, indent=2, ensure_ascii=False)
    return file_path


def run_clustering_pipeline(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_init: int = DEFAULT_N_INIT,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    project_root: Optional[Union[str, Path]] = None,
    evaluate_against_labels: bool = True,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Execute the complete Phase 5 Module 3 topic clustering pipeline.

    Workflow:
    1. Load preprocessed articles from backend/data/processed/articles_clean.csv.
    2. Extract TF-IDF features using Phase 5 Module 2 baseline.
    3. Fit K-Means on sparse TF-IDF matrix (unsupervised).
    4. Predict cluster assignments and align with article IDs.
    5. Extract top centroid terms and sparse representative articles.
    6. Generate interpretable cluster summaries.
    7. Optionally run post-hoc external evaluation against evaluation_labels.csv.
    8. Save model, cluster assignments, and cluster summaries.

    Returns:
        Dictionary containing pipeline results, summaries, and evaluation metrics.
    """
    root = get_project_root(project_root)

    # 1. Load preprocessed articles
    articles_df = load_processed_articles(project_root=root)
    validated_corpus = validate_corpus_for_tfidf(articles_df)

    # 2. Extract TF-IDF sparse features
    vectorizer, tfidf_matrix, tfidf_meta = extract_tfidf_features(validated_corpus)
    feature_names = tfidf_meta["feature_names"]
    article_ids = tfidf_meta["article_ids"]
    texts = validated_corpus["cleaned_text"].tolist()

    # 3. Fit K-Means on sparse matrix
    kmeans_model = fit_kmeans(
        tfidf_matrix=tfidf_matrix,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
    )

    # 4. Predict assignments
    assignments_df = predict_cluster_assignments(
        kmeans_model=kmeans_model,
        tfidf_matrix=tfidf_matrix,
        article_ids=article_ids,
    )

    # 5. Generate structured topic summaries
    summaries = generate_cluster_summaries(
        kmeans_model=kmeans_model,
        tfidf_matrix=tfidf_matrix,
        assignments_df=assignments_df,
        feature_names=feature_names,
        texts=texts,
    )

    # 6. Optional evaluation against external labels
    eval_results = None
    if evaluate_against_labels:
        labels_df = load_processed_evaluation_labels(project_root=root)
        eval_results = evaluate_clusters_against_labels(
            assignments_df=assignments_df,
            evaluation_labels_df=labels_df,
        )

    # 7. Persist artifacts
    saved_paths: Dict[str, str] = {}
    if save_artifacts:
        model_path = save_kmeans_model(kmeans_model, project_root=root)
        assign_path = save_cluster_assignments(assignments_df, project_root=root)
        sum_path = save_cluster_summaries(summaries, project_root=root)
        saved_paths = {
            "model_path": str(model_path),
            "assignments_path": str(assign_path),
            "summaries_path": str(sum_path),
        }

    return {
        "summaries": summaries,
        "assignments_df": assignments_df,
        "evaluation": eval_results,
        "saved_artifacts": saved_paths,
    }


def main():
    """Runnable entry point for Phase 5 Module 3 K-Means Topic Clustering."""
    print("=" * 75)
    print("NewsMap — Phase 5, Module 3: K-Means Topic Clustering Baseline")
    print("=" * 75)

    results = run_clustering_pipeline()
    summaries = results["summaries"]
    meta = summaries["metadata"]
    evaluation = results["evaluation"]
    saved_artifacts = results["saved_artifacts"]

    print(f"\n[1] Clustering Execution Summary:")
    print(f"    - Total Articles Clustered : {meta['total_articles']:,}")
    print(f"    - Number of Clusters (k)   : {meta['n_clusters']}")
    print(f"    - Vocabulary Dimension     : {meta['n_features']:,} features")
    print(f"    - Inertia (Sum of Sq Dists): {meta['inertia']:,.4f}")
    print(f"    - Convergence Iterations   : {meta['n_iter']}")
    print(f"    - Random State             : {meta['random_state']}")

    print(f"\n[2] Discovered Topic Clusters:")
    for cluster in summaries["clusters"]:
        cid = cluster["cluster_id"]
        count = cluster["article_count"]
        pct = cluster["percentage_of_corpus"]
        top_terms = ", ".join([t["term"] for t in cluster["top_terms"][:8]])
        print(f"\n    Cluster {cid} ({count} articles, {pct}%):")
        print(f"      Top Terms: {top_terms}")
        if cluster["representative_articles"]:
            rep = cluster["representative_articles"][0]
            rep_id = rep["article_id"]
            rep_dist = rep["distance_to_centroid"]
            rep_snip = rep.get("snippet", "").replace("\n", " ")[:80]
            print(f"      Representative Article: ID {rep_id} (dist: {rep_dist:.4f}) | \"{rep_snip}...\"")

    if evaluation:
        print(f"\n[3] External Post-Hoc Evaluation (Strictly Post-Training):")
        cov = evaluation["coverage"]
        met = evaluation["metrics"]
        print(f"    - Evaluated Articles       : {cov['matched_articles']:,} / {cov['total_assignments']:,} ({cov['alignment_ratio']*100:.1f}% coverage)")
        print(f"    - Adjusted Rand Index (ARI): {met['adjusted_rand_index']:.4f}")
        print(f"    - Normalized Mutual Info   : {met['normalized_mutual_info']:.4f}")
        print(f"    - Homogeneity              : {met['homogeneity']:.4f}")
        print(f"    - Completeness             : {met['completeness']:.4f}")
        print(f"    - V-Measure                : {met['v_measure']:.4f}")

        print(f"\n    Cluster Category Dominance Breakdown:")
        for cid, dom in evaluation["cluster_dominance"].items():
            print(
                f"      Cluster {cid} -> {dom['dominant_category'].upper():<13} "
                f"({dom['dominant_count']}/{dom['cluster_evaluated_total']} articles, {dom['purity_percentage']}% purity)"
            )

    print(f"\n[4] Saved Artifacts:")
    for name, path in saved_artifacts.items():
        print(f"    - {name:<17}: {path}")

    print("\n" + "=" * 75)
    print("Phase 5, Module 3 Execution Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
