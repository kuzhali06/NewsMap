"""
Focused Unit Tests for Phase 5 Module 1: Input Validation and Label Decoupling.

Tests:
1. Valid corpus validation (schema, types, non-empty).
2. Type error rejection for non-DataFrame inputs.
3. Empty corpus rejection.
4. Missing required column rejection.
5. Null or empty text rejection.
6. Duplicate article ID rejection.
7. Category label leakage rejection (supervised target strictly forbidden).
8. Evaluation label schema validation.
9. Non-mandatory label coverage handling (partial labels supported without error).
10. Extra labels tracking (from deduplication).
11. Real articles_clean.csv integration test.
12. Real evaluation_labels.csv integration and alignment test.
13. End-to-end prepare_validated_tfidf_inputs pipeline test.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.input_validation import (
    validate_corpus_for_tfidf,
    validate_evaluation_labels,
    prepare_validated_tfidf_inputs,
)
from backend.app.services.data_loader import (
    load_processed_articles,
    load_processed_evaluation_labels,
    get_project_root,
)


class TestTFIDFCorpusValidation(unittest.TestCase):
    """Unit tests for unsupervised corpus validation."""

    def test_valid_corpus_passes_validation(self):
        """A well-formed corpus with article_id and cleaned_text passes validation."""
        df = pd.DataFrame({
            "article_id": [1, 2, 3],
            "cleaned_text": [
                "economy grew substantially in the third quarter",
                "tech company launches cutting-edge mobile operating system",
                "football team secured victory in the cup final",
            ],
            "source": ["train", "train", "test"],
        })
        validated = validate_corpus_for_tfidf(df)
        self.assertEqual(len(validated), 3)

    def test_non_dataframe_raises_type_error(self):
        """Passing a non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            validate_corpus_for_tfidf(["not", "a", "dataframe"])

    def test_empty_dataframe_raises_value_error(self):
        """An empty DataFrame raises ValueError."""
        df = pd.DataFrame({"article_id": [], "cleaned_text": []})
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_missing_required_column_raises_value_error(self):
        """Missing article_id or cleaned_text raises ValueError."""
        df_no_id = pd.DataFrame({"cleaned_text": ["text sample"]})
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df_no_id)
        self.assertIn("missing required column", str(ctx.exception).lower())

        df_no_text = pd.DataFrame({"article_id": [101]})
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df_no_text)
        self.assertIn("missing required column", str(ctx.exception).lower())

    def test_null_or_empty_text_raises_value_error(self):
        """Null values or whitespace-only strings in cleaned_text raise ValueError."""
        df_null = pd.DataFrame({
            "article_id": [1, 2],
            "cleaned_text": ["valid text story", None],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df_null)
        self.assertIn("null", str(ctx.exception).lower())

        df_whitespace = pd.DataFrame({
            "article_id": [1, 2],
            "cleaned_text": ["valid text story", "   \t\n  "],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df_whitespace)
        self.assertIn("empty or whitespace-only", str(ctx.exception).lower())

    def test_duplicate_article_ids_raises_value_error(self):
        """Duplicate article_id values raise ValueError."""
        df_dup = pd.DataFrame({
            "article_id": [100, 200, 100],
            "cleaned_text": ["story one", "story two", "story three"],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df_dup)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_category_label_leakage_is_strictly_forbidden(self):
        """Passing a DataFrame containing category labels raises ValueError to prevent leakage."""
        df_leaked = pd.DataFrame({
            "article_id": [1, 2],
            "cleaned_text": ["story one", "story two"],
            "category": ["tech", "sport"],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_corpus_for_tfidf(df_leaked)
        self.assertIn("label column(s) detected in unsupervised corpus", str(ctx.exception).lower())


class TestEvaluationLabelsValidation(unittest.TestCase):
    """Unit tests for evaluation labels validation and corpus alignment."""

    def test_valid_labels_schema_passes(self):
        """A well-formed labels DataFrame passes validation."""
        df_labels = pd.DataFrame({
            "article_id": [1, 2, 3],
            "category": ["business", "tech", "sport"],
        })
        summary = validate_evaluation_labels(df_labels)
        self.assertEqual(summary["total_labels"], 3)
        self.assertEqual(len(summary["unique_categories"]), 3)

    def test_missing_category_column_raises_value_error(self):
        """Missing category column raises ValueError."""
        df_bad = pd.DataFrame({"article_id": [1, 2]})
        with self.assertRaises(ValueError) as ctx:
            validate_evaluation_labels(df_bad)
        self.assertIn("missing required column", str(ctx.exception).lower())

    def test_partial_label_coverage_handled_without_error(self):
        """Articles without labels do NOT trigger an error; coverage ratio is reported."""
        articles_df = pd.DataFrame({
            "article_id": [1, 2, 3, 4],
            "cleaned_text": ["a", "b", "c", "d"],
        })
        labels_df = pd.DataFrame({
            "article_id": [1, 2],
            "category": ["sport", "tech"],
        })
        summary = validate_evaluation_labels(labels_df, articles_df)
        self.assertEqual(summary["total_articles"], 4)
        self.assertEqual(summary["overlapping_count"], 2)
        self.assertEqual(summary["unlabeled_articles_count"], 2)
        self.assertEqual(summary["coverage_ratio"], 0.5)
        self.assertFalse(summary["has_complete_coverage"])

    def test_extra_labels_tracked(self):
        """Labels for articles dropped during deduplication are tracked as extra_labels."""
        articles_df = pd.DataFrame({
            "article_id": [10],
            "cleaned_text": ["retained story"],
        })
        labels_df = pd.DataFrame({
            "article_id": [10, 20, 30],
            "category": ["tech", "tech", "tech"],
        })
        summary = validate_evaluation_labels(labels_df, articles_df)
        self.assertEqual(summary["extra_labels_count"], 2)
        self.assertEqual(summary["overlapping_count"], 1)
        self.assertTrue(summary["has_complete_coverage"])


class TestRealProcessedDataIntegration(unittest.TestCase):
    """Integration tests verifying actual disk files backend/data/processed."""

    def test_load_and_validate_real_articles_clean(self):
        """Verify real articles_clean.csv loads and passes all validation checks."""
        root = get_project_root()
        articles_df = load_processed_articles(root)
        self.assertEqual(len(articles_df), 2123)
        self.assertIn("article_id", articles_df.columns)
        self.assertIn("cleaned_text", articles_df.columns)
        self.assertNotIn("category", articles_df.columns)

        validated = validate_corpus_for_tfidf(articles_df)
        self.assertEqual(len(validated), 2123)

    def test_load_and_validate_real_evaluation_labels(self):
        """Verify real evaluation_labels.csv loads and aligns with real articles."""
        root = get_project_root()
        labels_df = load_processed_evaluation_labels(root)
        articles_df = load_processed_articles(root)

        self.assertEqual(len(labels_df), 2225)
        summary = validate_evaluation_labels(labels_df, articles_df)
        self.assertEqual(summary["total_articles"], 2123)
        self.assertEqual(summary["overlapping_count"], 2123)
        self.assertEqual(summary["unlabeled_articles_count"], 0)
        self.assertEqual(summary["extra_labels_count"], 102)
        self.assertEqual(summary["coverage_ratio"], 1.0)
        self.assertTrue(summary["has_complete_coverage"])

    def test_prepare_validated_tfidf_inputs_pipeline(self):
        """Verify prepare_validated_tfidf_inputs executes cleanly and prepares TF-IDF inputs."""
        root = get_project_root()
        articles, labels, audit = prepare_validated_tfidf_inputs(root)
        self.assertEqual(len(articles), 2123)
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), 2225)
        self.assertTrue(audit["is_ready_for_tfidf"])
        self.assertEqual(audit["article_count"], 2123)


if __name__ == "__main__":
    unittest.main()
