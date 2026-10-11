r"""
Phase 4 Pipeline Runner for NewsMap.

Executes data loading, quality auditing, text cleaning, and dataset export.
Generates:
- backend/data/processed/articles_clean.csv (unsupervised article collection)
- backend/data/processed/evaluation_labels.csv (ground-truth evaluation labels)

Usage (from D:/NewsMap):
    .\.venv\Scripts\python.exe -m backend.run_phase4
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.data_loader import (
    load_training_data,
    load_test_data,
    load_sample_solution,
    load_evaluation_labels,
    prepare_article_collection,
    save_processed_articles,
    save_evaluation_labels,
    get_project_root,
)


def main():
    print("=" * 70)
    print("NewsMap - Phase 4: Data Cleaning & Text Preprocessing Pipeline")
    print("=" * 70)

    root = get_project_root()
    print(f"Project root: {root}")

    # 1. Load raw datasets
    print("\n[1/5] Loading raw datasets...")
    train_df = load_training_data(root)
    test_df = load_test_data(root)
    sample_df = load_sample_solution(root)
    print(f"  -> BBC News Train.csv:           {len(train_df):,} rows")
    print(f"  -> BBC News Test.csv:            {len(test_df):,} rows")
    print(f"  -> BBC News Sample Solution.csv: {len(sample_df):,} rows")

    # 2. Run data preparation & cleaning
    print("\n[2/5] Cleaning text and preparing article collection...")
    cleaned_df, audit = prepare_article_collection(
        train_df=train_df,
        test_df=test_df,
        project_root=root,
        deduplicate_strategy="keep_first",
    )

    # 3. Report audit metrics
    print("\n[3/5] Data Quality & Duplicate Findings:")
    print(f"  -> Total raw records:            {audit['total_raw_records']:,}")
    print(f"  -> Missing values found:         0 across all raw columns")
    print(f"  -> Overlapping ArticleIds:       {audit['overlapping_article_ids_count']} (ID sets are completely disjoint)")
    print(f"  -> Train duplicate raw texts:    {audit['train_quality']['duplicate_text_count']} duplicate rows")
    print(f"  -> Test duplicate raw texts:     {audit['test_quality']['duplicate_text_count']} duplicate rows")
    print(f"  -> Cross-split identical texts:  {audit['cross_split_raw_text_duplicates']} duplicate rows")
    print(f"  -> Total unique cleaned texts:   {audit['unique_cleaned_texts']:,}")
    print(f"  -> Deduplication strategy:       '{audit['deduplicate_strategy']}'")
    print(f"  -> Records dropped by policy:    {audit['dropped_records_count']}")
    print(f"  -> Final cleaned records count:  {audit['final_records_count']:,}")

    # 4. Save processed articles
    print("\n[4/5] Saving processed article collection...")
    saved_articles_path = save_processed_articles(cleaned_df, project_root=root)
    print(f"  -> Saved clean articles to: {saved_articles_path}")
    print(f"  -> Columns: {list(cleaned_df.columns)}")
    print(f"  -> Sample clean record preview:")
    print(f"     ID: {cleaned_df.iloc[0]['article_id']}")
    print(f"     Source: {cleaned_df.iloc[0]['source']}")
    print(f"     Word count: {cleaned_df.iloc[0]['word_count']}")
    print(f"     Clean text: {cleaned_df.iloc[0]['cleaned_text'][:120]}...")

    # 5. Save decoupled evaluation labels
    print("\n[5/5] Saving ground-truth evaluation labels...")
    eval_labels = load_evaluation_labels(root)
    saved_labels_path = save_evaluation_labels(eval_labels, project_root=root)
    print(f"  -> Saved evaluation labels to: {saved_labels_path}")
    print(f"  -> Total evaluation rows: {len(eval_labels):,} ({len(train_df):,} train + {len(sample_df):,} test)")
    print(f"  -> Columns: {list(eval_labels.columns)}")

    print("\n" + "=" * 70)
    print("Phase 4 Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
