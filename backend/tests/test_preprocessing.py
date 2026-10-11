"""
Unit tests for NewsMap Phase 4 Text Preprocessing and Data Ingestion.

Test categories:
1. Normal text cleaning.
2. Null, empty, and non-string inputs.
3. Whitespace normalization.
4. Case normalization.
5. Meaningful punctuation preservation (intra-word hyphens, decimals, currency, percent).
6. Meaningful negation preservation (contractions, negative words).
7. Repeated article IDs detection.
8. Duplicate text detection and handling.
9. CSV loading and required column validation.
10. Evaluation label separation.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Ensure backend package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.preprocessing import clean_text, extract_tokens, get_text_statistics
from backend.app.services.data_loader import (
    TRAIN_REQUIRED_COLUMNS,
    TEST_REQUIRED_COLUMNS,
    SAMPLE_SOLUTION_REQUIRED_COLUMNS,
    validate_required_columns,
    inspect_dataset_quality,
    prepare_article_collection,
    get_project_root,
)


class TestTextPreprocessing(unittest.TestCase):
    """Tests for text cleaning and normalization functions."""

    def test_normal_text(self):
        """Standard English sentence cleaning."""
        raw = "The economy grew substantially in the third quarter of this fiscal year."
        expected = "the economy grew substantially in the third quarter of this fiscal year"
        self.assertEqual(clean_text(raw), expected)

    def test_null_and_empty_input(self):
        """Safe handling of None, empty strings, and non-string types."""
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")
        self.assertEqual(clean_text("\t\n  \r"), "")
        self.assertEqual(clean_text(float("nan")), "")
        self.assertEqual(clean_text(False), "")
        self.assertEqual(clean_text(12345), "12345")
        self.assertEqual(clean_text(99.5), "99.5")

    def test_whitespace_normalization(self):
        """Collapse multiple spaces, tabs, and newlines into single spaces."""
        raw = "Headline:\t Breaking   News!\n\nThis is a report.\r\nNext   line.   "
        expected = "headline breaking news this is a report next line"
        self.assertEqual(clean_text(raw), expected)

    def test_case_normalization(self):
        """Verify default lowercase normalization and optional preserve_case."""
        raw = "BBC News REPORT on London Technology SUMMIT"
        self.assertEqual(
            clean_text(raw), "bbc news report on london technology summit"
        )
        self.assertEqual(
            clean_text(raw, preserve_case=True),
            "BBC News REPORT on London Technology SUMMIT",
        )

    def test_meaningful_punctuation_hyphens(self):
        """Preserve intra-word hyphens for compound concepts while stripping structural dashes."""
        raw = "The company deployed peer-to-peer, cost-effective, and cutting-edge software -- with zero downtime."
        expected = "the company deployed peer-to-peer cost-effective and cutting-edge software with zero downtime"
        self.assertEqual(clean_text(raw), expected)

    def test_meaningful_punctuation_decimals_and_currency(self):
        """Preserve decimal numbers, currency symbols, and percentages."""
        raw = "Revenue rose by 3.8% to £5.7bn ($11bn) from 95.5m in 2004."
        expected = "revenue rose by 3.8% to £5.7bn $11bn from 95.5m in 2004"
        self.assertEqual(clean_text(raw), expected)

    def test_meaningful_negation_and_contractions(self):
        """Preserve negations, negative words, and intra-word apostrophe contractions."""
        raw = "The minister won't resign, didn't agree, and cannot accept the terms. No, never!"
        expected = "the minister won't resign didn't agree and cannot accept the terms no never"
        self.assertEqual(clean_text(raw), expected)

    def test_html_and_unwanted_formatting(self):
        """Strip HTML tags and unescape entities without corrupting words."""
        raw = "<p>Tech giant &amp; partner announce &quot;new device&quot; (model #42); see [link].</p>"
        expected = "tech giant partner announce new device model 42 see link"
        self.assertEqual(clean_text(raw), expected)

    def test_helper_functions(self):
        """Test extract_tokens and get_text_statistics helpers."""
        cleaned = "tech giant partner announce new device"
        tokens = extract_tokens(cleaned)
        stats = get_text_statistics(cleaned)
        self.assertEqual(len(tokens), 6)
        self.assertEqual(stats["word_count"], 6)
        self.assertEqual(stats["char_count"], len(cleaned))


class TestDatasetQualityAndDuplicates(unittest.TestCase):
    """Tests for duplicate detection, ID collision prevention, and quality inspection."""

    def test_repeated_article_ids_detected(self):
        """Verify duplicate ArticleIds are identified and reported."""
        df = pd.DataFrame(
            {
                "ArticleId": [101, 102, 101, 103],
                "Text": ["First story", "Second story", "Third story", "Fourth story"],
            }
        )
        audit = inspect_dataset_quality(df)
        self.assertEqual(audit["duplicate_ids_count"], 1)
        self.assertIn(101, audit["duplicate_ids"])

    def test_duplicate_text_detection(self):
        """Verify identical text across different IDs is detected."""
        df = pd.DataFrame(
            {
                "ArticleId": [201, 202, 203],
                "Text": [
                    "Identical breaking news article body.",
                    "Completely unique sports commentary.",
                    "Identical breaking news article body.",
                ],
            }
        )
        audit = inspect_dataset_quality(df)
        self.assertEqual(audit["duplicate_text_count"], 1)
        self.assertEqual(audit["unique_text_count"], 2)

    def test_overlapping_article_ids_between_splits_raises(self):
        """Verify that an error is raised if train and test share ArticleIds."""
        train_df = pd.DataFrame(
            {"ArticleId": [1, 2], "Text": ["Story A", "Story B"], "Category": ["tech", "sport"]}
        )
        test_df = pd.DataFrame(
            {"ArticleId": [2, 3], "Text": ["Story C", "Story D"]}
        )
        with self.assertRaises(ValueError) as ctx:
            prepare_article_collection(train_df=train_df, test_df=test_df)
        self.assertIn("overlapping ArticleId", str(ctx.exception))

    def test_deduplication_policy_keep_first(self):
        """Verify 'keep_first' strategy retains 1 copy and reports dropped duplicates."""
        train_df = pd.DataFrame(
            {"ArticleId": [10, 20], "Text": ["Exact same content", "Unique content"], "Category": ["a", "b"]}
        )
        test_df = pd.DataFrame(
            {"ArticleId": [30, 40], "Text": ["Exact same content", "Another unique"],}
        )
        clean_df, audit = prepare_article_collection(
            train_df=train_df, test_df=test_df, deduplicate_strategy="keep_first"
        )
        self.assertEqual(len(clean_df), 3)  # 4 total - 1 duplicate = 3
        self.assertEqual(audit["dropped_records_count"], 1)
        self.assertEqual(audit["dropped_records_details"][0]["article_id"], 30)
        self.assertEqual(audit["dropped_records_details"][0]["retained_primary_article_id"], 10)

    def test_deduplication_policy_keep_all(self):
        """Verify 'keep_all' strategy preserves all rows and flags duplicates."""
        train_df = pd.DataFrame(
            {"ArticleId": [10, 20], "Text": ["Exact same content", "Unique content"], "Category": ["a", "b"]}
        )
        test_df = pd.DataFrame(
            {"ArticleId": [30, 40], "Text": ["Exact same content", "Another unique"]}
        )
        clean_df, audit = prepare_article_collection(
            train_df=train_df, test_df=test_df, deduplicate_strategy="keep_all"
        )
        self.assertEqual(len(clean_df), 4)
        self.assertIn("is_duplicate_text", clean_df.columns)
        self.assertEqual(int(clean_df["is_duplicate_text"].sum()), 1)


class TestDataLoaderAndValidation(unittest.TestCase):
    """Tests for CSV loading, error handling, and schema validation."""

    def test_required_column_validation_missing_column(self):
        """Verify ValueError is raised when required columns are missing."""
        bad_df = pd.DataFrame({"ArticleId": [1, 2], "WrongColumn": ["a", "b"]})
        with self.assertRaises(ValueError) as ctx:
            validate_required_columns(bad_df, TRAIN_REQUIRED_COLUMNS, "test_file.csv")
        self.assertIn("missing required column", str(ctx.exception))
        self.assertIn("Text", str(ctx.exception))
        self.assertIn("Category", str(ctx.exception))

    def test_required_column_validation_passes(self):
        """Verify validation succeeds when all columns are present."""
        good_train = pd.DataFrame(
            {"ArticleId": [1], "Text": ["text"], "Category": ["tech"]}
        )
        validate_required_columns(good_train, TRAIN_REQUIRED_COLUMNS, "train.csv")

        good_test = pd.DataFrame({"ArticleId": [1], "Text": ["text"]})
        validate_required_columns(good_test, TEST_REQUIRED_COLUMNS, "test.csv")

        good_sample = pd.DataFrame({"ArticleId": [1], "Category": ["tech"]})
        validate_required_columns(
            good_sample, SAMPLE_SOLUTION_REQUIRED_COLUMNS, "sample.csv"
        )

    def test_missing_csv_file_raises_not_found(self):
        """Verify FileNotFoundError is raised when file does not exist."""
        from backend.app.services.data_loader import _load_csv_file
        non_existent = Path("D:/NewsMap/non_existent_path.csv")
        with self.assertRaises(FileNotFoundError):
            _load_csv_file(non_existent, ["ArticleId"], "missing.csv")

    def test_categories_never_merged_into_cleaned_text(self):
        """Verify category labels are not leaked into the unsupervised corpus."""
        train_df = pd.DataFrame(
            {
                "ArticleId": [100],
                "Text": ["UK economic growth rebounded."],
                "Category": ["business"],
            }
        )
        test_df = pd.DataFrame(
            {"ArticleId": [200], "Text": ["Tech startup secures series A."]}
        )
        clean_df, _ = prepare_article_collection(train_df=train_df, test_df=test_df)
        self.assertNotIn("category", clean_df.columns)
        self.assertNotIn("Category", clean_df.columns)
        self.assertNotIn("business", clean_df["cleaned_text"].iloc[0])


if __name__ == "__main__":
    unittest.main()
