"""
Input Validation Module for NewsMap (Phase 5).

Provides verified, decoupled input interfaces for unsupervised feature extraction (TF-IDF):
1. validate_corpus_for_tfidf: Validates schema, types, absence of nulls/empty strings,
   uniqueness of article IDs, and strictly prevents supervised category label leakage.
2. validate_evaluation_labels: Audits ground-truth category labels, assessing coverage
   against the article corpus without assuming all articles must be labeled.
3. prepare_validated_tfidf_inputs: High-level entrypoint that loads, audits, and returns
   verified corpus and decoupled evaluation labels for downstream modeling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from backend.app.services.data_loader import (
    get_project_root,
    load_processed_articles,
    load_processed_evaluation_labels,
)

# Mandatory columns for unsupervised TF-IDF
TFIDF_REQUIRED_COLUMNS = ["article_id", "cleaned_text"]

# Forbidden columns that would cause label leakage in unsupervised training
FORBIDDEN_UNSUPERVISED_COLUMNS = [
    "category",
    "Category",
    "label",
    "Label",
    "target",
    "Target",
]

# Mandatory columns for ground-truth evaluation
EVALUATION_REQUIRED_COLUMNS = ["article_id", "category"]


def validate_corpus_for_tfidf(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    id_col: str = "article_id",
) -> pd.DataFrame:
    """
    Validate that an article DataFrame is structurally sound and strictly decoupled
    from category labels before passing to TF-IDF feature extraction.

    Args:
        df: Input pandas DataFrame containing cleaned articles.
        text_col: Column name containing preprocessed text. Defaults to 'cleaned_text'.
        id_col: Column name containing unique article identifiers. Defaults to 'article_id'.

    Returns:
        The validated DataFrame.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If input is empty, missing required columns, contains nulls,
                    contains empty text, contains duplicate IDs, or leaks category labels.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Corpus input must be a pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Corpus DataFrame is empty. At least one article record is required.")

    # 1. Validate required columns exist
    missing_cols = [col for col in (id_col, text_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Corpus is missing required column(s): {missing_cols}. Found columns: {list(df.columns)}"
        )

    # 2. Enforce strict label separation: prevent supervised leakage
    leakage_cols = [col for col in FORBIDDEN_UNSUPERVISED_COLUMNS if col in df.columns]
    if leakage_cols:
        raise ValueError(
            f"Category label column(s) detected in unsupervised corpus: {leakage_cols}. "
            f"Category labels must remain strictly decoupled from TF-IDF inputs."
        )

    # 3. Validate article IDs (no nulls, all unique)
    if df[id_col].isnull().any():
        null_count = int(df[id_col].isnull().sum())
        raise ValueError(f"Corpus column '{id_col}' contains {null_count} null/missing value(s).")

    duplicate_ids_count = int(df[id_col].duplicated().sum())
    if duplicate_ids_count > 0:
        sample_dups = df.loc[df[id_col].duplicated(keep=False), id_col].unique()[:5].tolist()
        raise ValueError(
            f"Corpus contains {duplicate_ids_count} duplicate '{id_col}' values. Sample duplicates: {sample_dups}"
        )

    # 4. Validate cleaned text (no nulls, no empty strings)
    if df[text_col].isnull().any():
        null_text_count = int(df[text_col].isnull().sum())
        raise ValueError(f"Corpus column '{text_col}' contains {null_text_count} null/missing value(s).")

    empty_mask = df[text_col].astype(str).str.strip() == ""
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        sample_empty_ids = df.loc[empty_mask, id_col].tolist()[:5]
        raise ValueError(
            f"Corpus contains {empty_count} empty or whitespace-only text record(s). "
            f"Sample article IDs with empty text: {sample_empty_ids}"
        )

    return df


def validate_evaluation_labels(
    labels_df: pd.DataFrame,
    articles_df: Optional[pd.DataFrame] = None,
    id_col: str = "article_id",
    category_col: str = "category",
) -> Dict[str, Any]:
    """
    Validate ground-truth category evaluation labels and audit coverage against the article corpus.

    Note: Does NOT require every article to have a label. Partial label coverage is supported
    and reported transparently for real-world unsupervised pipelines.

    Args:
        labels_df: Input pandas DataFrame containing evaluation labels.
        articles_df: Optional DataFrame of article records to check ID overlap.
        id_col: Column name containing article identifiers.
        category_col: Column name containing ground-truth category labels.

    Returns:
        Dictionary containing label audit metrics, categories breakdown, and coverage ratio.

    Raises:
        TypeError: If labels_df is not a pandas DataFrame.
        ValueError: If labels_df is empty, missing required columns, or contains nulls/duplicates.
    """
    if not isinstance(labels_df, pd.DataFrame):
        raise TypeError(f"Labels input must be a pandas DataFrame, got {type(labels_df).__name__}.")

    if labels_df.empty:
        raise ValueError("Evaluation labels DataFrame is empty.")

    missing_cols = [col for col in (id_col, category_col) if col not in labels_df.columns]
    if missing_cols:
        raise ValueError(
            f"Evaluation labels missing required column(s): {missing_cols}. Found: {list(labels_df.columns)}"
        )

    if labels_df[id_col].isnull().any():
        null_count = int(labels_df[id_col].isnull().sum())
        raise ValueError(f"Evaluation labels column '{id_col}' contains {null_count} null value(s).")

    if labels_df[category_col].isnull().any():
        null_count = int(labels_df[category_col].isnull().sum())
        raise ValueError(f"Evaluation labels column '{category_col}' contains {null_count} null value(s).")

    dup_ids_count = int(labels_df[id_col].duplicated().sum())
    if dup_ids_count > 0:
        sample_dups = labels_df.loc[labels_df[id_col].duplicated(keep=False), id_col].unique()[:5].tolist()
        raise ValueError(f"Evaluation labels contain {dup_ids_count} duplicate '{id_col}' values: {sample_dups}")

    summary: Dict[str, Any] = {
        "total_labels": len(labels_df),
        "unique_categories": sorted(labels_df[category_col].unique().tolist()),
        "category_counts": {str(k): int(v) for k, v in labels_df[category_col].value_counts().items()},
    }

    if articles_df is not None and not articles_df.empty:
        article_ids = set(articles_df[id_col])
        label_ids = set(labels_df[id_col])

        overlap = article_ids.intersection(label_ids)
        unlabeled = article_ids - label_ids
        extra = label_ids - article_ids

        coverage = len(overlap) / len(article_ids) if len(article_ids) > 0 else 0.0

        summary.update({
            "total_articles": len(articles_df),
            "overlapping_count": len(overlap),
            "unlabeled_articles_count": len(unlabeled),
            "unlabeled_article_ids_sample": sorted(list(unlabeled))[:5],
            "extra_labels_count": len(extra),
            "coverage_ratio": float(coverage),
            "has_complete_coverage": len(unlabeled) == 0,
        })

    return summary


def prepare_validated_tfidf_inputs(
    project_root: Optional[Union[str, Path]] = None,
    articles_df: Optional[pd.DataFrame] = None,
    labels_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load, validate, and package input datasets for the Phase 5 TF-IDF module.

    Guarantees:
    - Corpus is valid, non-empty, and free of null/empty text.
    - Category labels are completely decoupled from the training corpus.
    - Relationship between articles and evaluation labels is documented in audit summary.

    Returns:
        Tuple of:
        - Validated articles DataFrame [article_id, cleaned_text, ...]
        - Validated evaluation labels DataFrame [article_id, category, ...] (or None)
        - Audit summary dictionary detailing counts, schemas, and coverage.
    """
    root = get_project_root(project_root)

    # 1. Load articles if not supplied
    if articles_df is None:
        articles_df = load_processed_articles(root)

    # 2. Validate unsupervised corpus
    validated_articles = validate_corpus_for_tfidf(articles_df)

    # 3. Load and validate evaluation labels if available
    validated_labels: Optional[pd.DataFrame] = None
    labels_summary: Optional[Dict[str, Any]] = None

    if labels_df is None:
        labels_file = root / "backend" / "data" / "processed" / "evaluation_labels.csv"
        if labels_file.exists():
            labels_df = load_processed_evaluation_labels(root)

    if labels_df is not None:
        labels_summary = validate_evaluation_labels(labels_df, validated_articles)
        validated_labels = labels_df

    audit_summary: Dict[str, Any] = {
        "article_count": len(validated_articles),
        "columns": list(validated_articles.columns),
        "source_breakdown": {str(k): int(v) for k, v in validated_articles["source"].value_counts().items()}
        if "source" in validated_articles.columns
        else {},
        "evaluation_audit": labels_summary,
        "is_ready_for_tfidf": True,
    }

    return validated_articles, validated_labels, audit_summary
