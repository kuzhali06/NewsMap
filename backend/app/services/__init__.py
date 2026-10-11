"""
NewsMap Services Package.
Modules for data ingestion, quality inspection, dataset preparation, and related-story detection.
"""

__all__ = [
    # Data loaders
    "load_training_data",
    "load_test_data",
    "load_sample_solution",
    "load_evaluation_labels",
    "load_processed_articles",
    "load_processed_evaluation_labels",
    "inspect_dataset_quality",
    "prepare_article_collection",
    "save_processed_articles",
    "save_evaluation_labels",
    "get_project_root",
    # Related story detection (Module 4)
    "RelatedStoryService",
    "get_related_stories",
    "compute_similarity_scores",
    "format_related_stories_report",
]


def __getattr__(name: str):
    if name in (
        "load_training_data",
        "load_test_data",
        "load_sample_solution",
        "load_evaluation_labels",
        "load_processed_articles",
        "load_processed_evaluation_labels",
        "inspect_dataset_quality",
        "prepare_article_collection",
        "save_processed_articles",
        "save_evaluation_labels",
        "get_project_root",
    ):
        from backend.app.services import data_loader
        return getattr(data_loader, name)
    if name in (
        "RelatedStoryService",
        "get_related_stories",
        "compute_similarity_scores",
        "format_related_stories_report",
    ):
        from backend.app.services import related_stories
        return getattr(related_stories, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
