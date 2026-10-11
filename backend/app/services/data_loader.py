"""
Data Loader and Ingestion Service for NewsMap.

This module provides maintainable functions to:
1. Load train, test, and sample solution CSV files using relative project paths.
2. Validate required columns for each file.
3. Preserve article IDs as distinct identifiers.
4. Inspect dataset quality (missing values, empty text, duplicate IDs, duplicate content).
5. Prepare a consolidated unlabeled article collection with explicit deduplication policies.
6. Decouple category evaluation labels from unsupervised topic modeling inputs.
7. Save cleaned data to backend/data/processed/articles_clean.csv.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from backend.app.ml.preprocessing import clean_text, get_text_statistics

# Expected CSV Column Schemas
TRAIN_REQUIRED_COLUMNS = ["ArticleId", "Text", "Category"]
TEST_REQUIRED_COLUMNS = ["ArticleId", "Text"]
SAMPLE_SOLUTION_REQUIRED_COLUMNS = ["ArticleId", "Category"]


def get_project_root(custom_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Determine the project root directory without hard-coding absolute paths.

    Args:
        custom_root: Optional explicit project root path.

    Returns:
        Resolved Path pointing to the project root directory.
    """
    if custom_root:
        return Path(custom_root).resolve()

    # Search upward for project root containing backend/data/raw
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "backend" / "data" / "raw").exists():
            return candidate

    # Fallback using relative structure (backend/app/services/data_loader.py -> parents[3])
    parents = Path(__file__).resolve().parents
    if len(parents) >= 4:
        return parents[3]
    return Path.cwd()


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    source_name: str,
) -> None:
    """
    Verify that all mandatory columns exist in the DataFrame.

    Args:
        df: Input pandas DataFrame.
        required_columns: List of column names that must be present.
        source_name: Name of the dataset file or source for error reporting.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{source_name}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _load_csv_file(
    file_path: Path,
    required_columns: List[str],
    source_name: str,
) -> pd.DataFrame:
    """
    Internal helper to load a CSV file, validate columns, and handle errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset file not found at: '{file_path}'. "
            f"Please verify that the file exists relative to the project root."
        )

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file '{source_name}' at '{file_path}' is empty.")
    except Exception as exc:
        raise ValueError(
            f"Failed to read CSV file '{source_name}' at '{file_path}': {exc}"
        ) from exc

    validate_required_columns(df, required_columns, source_name)

    # Ensure ArticleId is preserved as integer/numeric identifier
    if "ArticleId" in df.columns:
        if df["ArticleId"].isnull().any():
            raise ValueError(f"Dataset '{source_name}' contains null ArticleId values.")
        df["ArticleId"] = df["ArticleId"].astype(int)

    return df


def load_training_data(
    project_root: Optional[Union[str, Path]] = None,
    relative_path: str = "backend/data/raw/BBC News Train.csv",
) -> pd.DataFrame:
    """
    Load BBC News Train dataset.

    Args:
        project_root: Optional project root path.
        relative_path: Relative path to training CSV from project root.

    Returns:
        DataFrame containing ArticleId, Text, and Category.
    """
    root = get_project_root(project_root)
    file_path = root / relative_path
    return _load_csv_file(file_path, TRAIN_REQUIRED_COLUMNS, "BBC News Train.csv")


def load_test_data(
    project_root: Optional[Union[str, Path]] = None,
    relative_path: str = "backend/data/raw/BBC News Test.csv",
) -> pd.DataFrame:
    """
    Load BBC News Test dataset.

    Args:
        project_root: Optional project root path.
        relative_path: Relative path to test CSV from project root.

    Returns:
        DataFrame containing ArticleId and Text.
    """
    root = get_project_root(project_root)
    file_path = root / relative_path
    return _load_csv_file(file_path, TEST_REQUIRED_COLUMNS, "BBC News Test.csv")


def load_sample_solution(
    project_root: Optional[Union[str, Path]] = None,
    relative_path: str = "backend/data/raw/BBC News Sample Solution.csv",
) -> pd.DataFrame:
    """
    Load BBC News Sample Solution dataset (ground-truth test labels).

    Args:
        project_root: Optional project root path.
        relative_path: Relative path to sample solution CSV from project root.

    Returns:
        DataFrame containing ArticleId and Category.
    """
    root = get_project_root(project_root)
    file_path = root / relative_path
    return _load_csv_file(
        file_path, SAMPLE_SOLUTION_REQUIRED_COLUMNS, "BBC News Sample Solution.csv"
    )


def load_evaluation_labels(
    project_root: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load and assemble ground-truth category evaluation labels from train and sample solution.
    Keeps evaluation labels decoupled from the unsupervised training inputs.

    Returns:
        DataFrame with columns: article_id, category, source.
    """
    train_df = load_training_data(project_root)
    sample_df = load_sample_solution(project_root)

    train_labels = train_df[["ArticleId", "Category"]].copy()
    train_labels["source"] = "train"

    test_labels = sample_df[["ArticleId", "Category"]].copy()
    test_labels["source"] = "test"

    combined = pd.concat([train_labels, test_labels], ignore_index=True)
    combined.rename(
        columns={"ArticleId": "article_id", "Category": "category"}, inplace=True
    )
    return combined


def inspect_dataset_quality(
    df: pd.DataFrame,
    id_col: str = "ArticleId",
    text_col: str = "Text",
    dataset_name: str = "dataset",
) -> Dict[str, Any]:
    """
    Perform a comprehensive data quality audit on a dataset.

    Reports:
    - Total row count
    - Missing/null counts per column
    - Empty or whitespace-only text count
    - Duplicate ID count and list of repeated IDs
    - Duplicate text count and list of duplicated texts
    """
    total_rows = len(df)
    missing_by_col = {col: int(df[col].isnull().sum()) for col in df.columns}

    # Empty text check
    empty_text_count = 0
    if text_col in df.columns:
        empty_text_count = int(
            df[text_col].apply(lambda x: x is None or not str(x).strip()).sum()
        )

    # Duplicate IDs
    duplicate_ids_count = 0
    duplicate_ids: List[int] = []
    if id_col in df.columns:
        dup_mask = df[id_col].duplicated(keep=False)
        duplicate_ids_count = int(df[id_col].duplicated().sum())
        if duplicate_ids_count > 0:
            duplicate_ids = sorted(df.loc[dup_mask, id_col].unique().tolist())

    # Duplicate text
    duplicate_text_count = 0
    unique_text_count = total_rows
    if text_col in df.columns:
        duplicate_text_count = int(df[text_col].duplicated().sum())
        unique_text_count = int(df[text_col].nunique())

    return {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "missing_by_column": missing_by_col,
        "empty_text_count": empty_text_count,
        "duplicate_ids_count": duplicate_ids_count,
        "duplicate_ids": duplicate_ids,
        "unique_text_count": unique_text_count,
        "duplicate_text_count": duplicate_text_count,
    }


def prepare_article_collection(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    project_root: Optional[Union[str, Path]] = None,
    deduplicate_strategy: str = "keep_first",
    clean_fn: Optional[Callable[[Any], str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare the unlabeled article collection from train and test datasets.

    Deduplication Policy:
    - Identifies exact duplicate texts and repeated article IDs separately.
    - Does NOT silently discard records; all duplicate findings are tracked and returned
      in the audit report.
    - 'keep_first': Retains the first occurrence of identical cleaned texts and logs
      dropped duplicates in the audit report.
    - 'keep_all': Preserves all records, adding an 'is_duplicate_text' boolean flag.

    Safety Guarantees:
    - Verifies disjointness of ArticleIds between train and test.
    - Adds 'source' column ('train' or 'test') for complete traceability.
    - Never merges 'Category' labels into article text or the unsupervised collection.

    Args:
        train_df: Optional preloaded train DataFrame.
        test_df: Optional preloaded test DataFrame.
        project_root: Optional project root path.
        deduplicate_strategy: 'keep_first' or 'keep_all'. Defaults to 'keep_first'.
        clean_fn: Optional text cleaning function. Defaults to clean_text.

    Returns:
        Tuple of:
        - Cleaned, prepared DataFrame with columns:
          [article_id, cleaned_text, source, word_count, char_count]
        - Audit report dictionary detailing quality checks and deduplication metrics.
    """
    if clean_fn is None:
        clean_fn = clean_text

    if deduplicate_strategy not in ("keep_first", "keep_all"):
        raise ValueError(
            f"Invalid deduplicate_strategy '{deduplicate_strategy}'. "
            f"Must be one of: 'keep_first', 'keep_all'."
        )

    # 1. Load data if not provided
    if train_df is None:
        train_df = load_training_data(project_root)
    if test_df is None:
        test_df = load_test_data(project_root)

    # 2. Audit individual sets
    train_audit = inspect_dataset_quality(train_df, dataset_name="train")
    test_audit = inspect_dataset_quality(test_df, dataset_name="test")

    # 3. Check for overlapping Article IDs
    train_ids: Set[int] = set(train_df["ArticleId"])
    test_ids: Set[int] = set(test_df["ArticleId"])
    overlapping_ids = sorted(train_ids.intersection(test_ids))
    if overlapping_ids:
        raise ValueError(
            f"Detected {len(overlapping_ids)} overlapping ArticleId(s) between "
            f"train and test sets: {overlapping_ids[:10]}... Article IDs must be distinct."
        )

    # 4. Check for overlapping texts between splits
    train_texts: Set[str] = set(train_df["Text"])
    test_texts: Set[str] = set(test_df["Text"])
    cross_split_duplicates_count = len(train_texts.intersection(test_texts))

    # 5. Build combined collection with traceability (excluding category labels)
    train_records = train_df[["ArticleId", "Text"]].copy()
    train_records["source"] = "train"

    test_records = test_df[["ArticleId", "Text"]].copy()
    test_records["source"] = "test"

    combined_df = pd.concat([train_records, test_records], ignore_index=True)
    raw_total_records = len(combined_df)

    # 6. Apply text preprocessing
    combined_df["cleaned_text"] = combined_df["Text"].apply(clean_fn)

    # 7. Compute justified metadata (word count, character count)
    stats = combined_df["cleaned_text"].apply(get_text_statistics)
    combined_df["char_count"] = stats.apply(lambda s: s["char_count"])
    combined_df["word_count"] = stats.apply(lambda s: s["word_count"])

    # 8. Rename ArticleId to article_id for standardization
    combined_df.rename(columns={"ArticleId": "article_id"}, inplace=True)

    # 9. Duplicate detection on cleaned text
    is_duplicate_mask = combined_df["cleaned_text"].duplicated(keep="first")
    total_duplicate_texts = int(is_duplicate_mask.sum())
    unique_cleaned_texts = raw_total_records - total_duplicate_texts

    # Collect details on duplicate records for transparent reporting
    dropped_records_info: List[Dict[str, Any]] = []
    if total_duplicate_texts > 0:
        # Find groups of identical cleaned text
        for text_val, group in combined_df.groupby("cleaned_text"):
            if len(group) > 1:
                primary_id = int(group.iloc[0]["article_id"])
                for idx, row in group.iloc[1:].iterrows():
                    dropped_records_info.append(
                        {
                            "article_id": int(row["article_id"]),
                            "retained_primary_article_id": primary_id,
                            "source": row["source"],
                            "text_preview": text_val[:80],
                        }
                    )

    # 10. Apply explicit deduplication strategy
    if deduplicate_strategy == "keep_first":
        final_df = combined_df[~is_duplicate_mask].copy()
    else:  # keep_all
        final_df = combined_df.copy()
        final_df["is_duplicate_text"] = is_duplicate_mask

    # Select and order final output columns
    output_columns = ["article_id", "cleaned_text", "source", "word_count", "char_count"]
    if "is_duplicate_text" in final_df.columns:
        output_columns.append("is_duplicate_text")
    final_df = final_df[output_columns].reset_index(drop=True)

    # 11. Compile comprehensive audit report
    audit_report = {
        "raw_train_records": len(train_df),
        "raw_test_records": len(test_df),
        "total_raw_records": raw_total_records,
        "train_quality": train_audit,
        "test_quality": test_audit,
        "overlapping_article_ids_count": len(overlapping_ids),
        "cross_split_raw_text_duplicates": cross_split_duplicates_count,
        "unique_cleaned_texts": unique_cleaned_texts,
        "duplicate_cleaned_texts_count": total_duplicate_texts,
        "deduplicate_strategy": deduplicate_strategy,
        "final_records_count": len(final_df),
        "dropped_records_count": len(dropped_records_info) if deduplicate_strategy == "keep_first" else 0,
        "dropped_records_details": dropped_records_info,
    }

    return final_df, audit_report


def save_processed_articles(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save the cleaned article collection to CSV.

    Args:
        df: Processed DataFrame containing article_id, cleaned_text, and metadata.
        output_path: Optional target path. Defaults to backend/data/processed/articles_clean.csv.
        project_root: Optional project root path.

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        root = get_project_root(project_root)
        output_file = root / "backend" / "data" / "processed" / "articles_clean.csv"
    else:
        output_file = Path(output_path).resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    return output_file


def save_evaluation_labels(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save ground-truth category evaluation labels to a separate CSV.

    Args:
        df: DataFrame containing article_id, category, source.
        output_path: Target path. Defaults to backend/data/processed/evaluation_labels.csv.
        project_root: Optional project root path.

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        root = get_project_root(project_root)
        output_file = root / "backend" / "data" / "processed" / "evaluation_labels.csv"
    else:
        output_file = Path(output_path).resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    return output_file


# Processed Dataset Schemas
PROCESSED_ARTICLES_REQUIRED_COLUMNS = ["article_id", "cleaned_text"]
EVALUATION_LABELS_REQUIRED_COLUMNS = ["article_id", "category"]


def load_processed_articles(
    project_root: Optional[Union[str, Path]] = None,
    relative_path: str = "backend/data/processed/articles_clean.csv",
) -> pd.DataFrame:
    """
    Load preprocessed articles CSV file.

    Args:
        project_root: Optional project root path.
        relative_path: Relative path to processed articles CSV.

    Returns:
        DataFrame containing article_id, cleaned_text, and justified metadata.
    """
    root = get_project_root(project_root)
    file_path = root / relative_path
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed articles file not found at: '{file_path}'. "
            f"Ensure Phase 4 pipeline has run."
        )

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Processed articles file at '{file_path}' is empty.")
    except Exception as exc:
        raise ValueError(f"Failed to read processed articles CSV at '{file_path}': {exc}") from exc

    validate_required_columns(
        df, PROCESSED_ARTICLES_REQUIRED_COLUMNS, "articles_clean.csv"
    )

    if "article_id" in df.columns:
        if df["article_id"].isnull().any():
            raise ValueError("Processed articles contain null article_id values.")
        df["article_id"] = df["article_id"].astype(int)

    return df


def load_processed_evaluation_labels(
    project_root: Optional[Union[str, Path]] = None,
    relative_path: str = "backend/data/processed/evaluation_labels.csv",
) -> pd.DataFrame:
    """
    Load saved ground-truth evaluation labels CSV file.

    Args:
        project_root: Optional project root path.
        relative_path: Relative path to evaluation labels CSV.

    Returns:
        DataFrame containing article_id, category, and source.
    """
    root = get_project_root(project_root)
    file_path = root / relative_path
    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation labels file not found at: '{file_path}'. "
            f"Ensure Phase 4 pipeline has run."
        )

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Evaluation labels file at '{file_path}' is empty.")
    except Exception as exc:
        raise ValueError(f"Failed to read evaluation labels CSV at '{file_path}': {exc}") from exc

    validate_required_columns(
        df, EVALUATION_LABELS_REQUIRED_COLUMNS, "evaluation_labels.csv"
    )

    if "article_id" in df.columns:
        if df["article_id"].isnull().any():
            raise ValueError("Evaluation labels contain null article_id values.")
        df["article_id"] = df["article_id"].astype(int)

    return df
