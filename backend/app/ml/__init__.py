"""
NewsMap Machine Learning Package.
Modules for text preprocessing, input validation, feature extraction, and topic clustering.
"""

__all__ = [
    # Preprocessing
    "clean_text",
    "clean_series",
    "extract_tokens",
    "get_text_statistics",
    # Input Validation
    "validate_corpus_for_tfidf",
    "validate_evaluation_labels",
    "prepare_validated_tfidf_inputs",
    # Feature Extraction
    "extract_tfidf_features",
    "get_top_terms_for_document",
    "save_tfidf_vectorizer",
    "load_tfidf_vectorizer",
    # Topic Clustering (Module 3)
    "fit_kmeans",
    "predict_cluster_assignments",
    "get_cluster_centroid_top_terms",
    "get_representative_articles",
    "generate_cluster_summaries",
    "evaluate_clusters_against_labels",
    "save_kmeans_model",
    "load_kmeans_model",
    "save_cluster_assignments",
    "save_cluster_summaries",
    "run_clustering_pipeline",
]


def __getattr__(name: str):
    if name in ("clean_text", "clean_series", "extract_tokens", "get_text_statistics"):
        from backend.app.ml import preprocessing
        return getattr(preprocessing, name)
    if name in (
        "validate_corpus_for_tfidf",
        "validate_evaluation_labels",
        "prepare_validated_tfidf_inputs",
    ):
        from backend.app.ml import input_validation
        return getattr(input_validation, name)
    if name in (
        "extract_tfidf_features",
        "get_top_terms_for_document",
        "save_tfidf_vectorizer",
        "load_tfidf_vectorizer",
    ):
        from backend.app.ml import feature_extraction
        return getattr(feature_extraction, name)
    if name in (
        "fit_kmeans",
        "predict_cluster_assignments",
        "get_cluster_centroid_top_terms",
        "get_representative_articles",
        "generate_cluster_summaries",
        "evaluate_clusters_against_labels",
        "save_kmeans_model",
        "load_kmeans_model",
        "save_cluster_assignments",
        "save_cluster_summaries",
        "run_clustering_pipeline",
    ):
        from backend.app.ml import clustering
        return getattr(clustering, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
