"""
Unit and Integration Tests for Phase 5 Module 2: TF-IDF Feature Extraction.

Tests:
1. Normal TF-IDF matrix creation (shape, csr_matrix format, non-zero entries).
2. Row-to-article-ID alignment (direct index-to-ID mapping and term relevance).
3. Deterministic behavior (identical inputs + config -> bit-for-bit identical matrix).
4. Invalid and empty inputs rejection (empty corpus, non-DataFrame/non-list, nulls).
5. Configurable vectorizer settings (max_features, min_df, ngram_range).
6. Sparse matrix preservation (verifies CSR format, no dense array conversion).
7. Empty-vocabulary handling (overly aggressive filtering raises clear ValueError).
8. Category-label isolation (category column in corpus rejected via input validation).
9. Vectorizer persistence (save/load via joblib yields identical transform).
10. Top terms extraction helper (correct ordering and values).
11. Real BBC corpus integration (2,123 articles, 0 empty vectors, valid dimensions).
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

# Ensure backend package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.feature_extraction import (
    extract_tfidf_features,
    get_top_terms_for_document,
    save_tfidf_vectorizer,
    load_tfidf_vectorizer,
)
from backend.app.services.data_loader import (
    get_project_root,
    load_processed_articles,
)


class TestTFIDFFeatureExtraction(unittest.TestCase):
    """Unit tests for TF-IDF feature extraction module."""

    def setUp(self):
        """Prepare sample corpus for unit testing."""
        self.sample_df = pd.DataFrame({
            "article_id": [101, 102, 103, 104],
            "cleaned_text": [
                "economy grew substantially in the third quarter interest rates remained stable",
                "tech company launches cutting-edge mobile operating system with cloud storage",
                "football team secured victory in the domestic cup final after extra time",
                "government announces tax reforms and public spending cuts in budget report",
            ],
            "source": ["train", "train", "test", "test"],
        })

    def test_normal_tfidf_matrix_creation(self):
        """Verify normal creation of TF-IDF matrix, CSR format, and metadata."""
        vec, X, meta = extract_tfidf_features(
            self.sample_df,
            min_df=1,
            max_df=1.0,
            max_features=50,
            ngram_range=(1, 1),
        )
        self.assertTrue(isspmatrix_csr(X))
        self.assertEqual(X.shape[0], 4)
        self.assertGreater(X.shape[1], 0)
        self.assertEqual(meta["n_documents"], 4)
        self.assertEqual(meta["n_features"], X.shape[1])
        self.assertEqual(meta["matrix_format"], "csr_matrix")
        self.assertEqual(meta["empty_vector_count"], 0)
        self.assertGreater(meta["nnz"], 0)
        self.assertGreater(meta["sparsity"], 0.0)

    def test_row_to_article_id_alignment(self):
        """Verify that row index i aligns with article_id and corresponds to relevant terms."""
        vec, X, meta = extract_tfidf_features(
            self.sample_df,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
        )
        # Check ID mapping
        self.assertEqual(meta["article_ids"], [101, 102, 103, 104])

        # Row 1 (article 102) is about 'tech company mobile cutting-edge'
        row_1_terms = dict(get_top_terms_for_document(X, meta["feature_names"], doc_index=1, top_n=10))
        self.assertIn("mobile", row_1_terms)
        self.assertIn("cutting-edge", row_1_terms)
        self.assertIn("tech", row_1_terms)

        # Row 2 (article 103) is about 'football team cup final'
        row_2_terms = dict(get_top_terms_for_document(X, meta["feature_names"], doc_index=2, top_n=10))
        self.assertIn("football", row_2_terms)
        self.assertIn("cup", row_2_terms)

    def test_deterministic_behavior(self):
        """Two runs on identical corpus and config produce bit-for-bit identical matrices."""
        vec1, X1, meta1 = extract_tfidf_features(self.sample_df, min_df=1, max_df=1.0)
        vec2, X2, meta2 = extract_tfidf_features(self.sample_df, min_df=1, max_df=1.0)

        self.assertEqual(meta1["feature_names"], meta2["feature_names"])
        self.assertEqual(meta1["article_ids"], meta2["article_ids"])
        np.testing.assert_array_equal(X1.toarray(), X2.toarray())

    def test_invalid_and_empty_inputs(self):
        """Rejects empty, non-DataFrame/non-list, and malformed inputs with clear errors."""
        empty_df = pd.DataFrame({"article_id": [], "cleaned_text": []})
        with self.assertRaises(ValueError):
            extract_tfidf_features(empty_df)

        with self.assertRaises(TypeError):
            extract_tfidf_features(12345)

        with self.assertRaises(ValueError):
            extract_tfidf_features([])

        with self.assertRaises(ValueError):
            extract_tfidf_features(["valid story", ""])

    def test_configurable_vectorizer_settings(self):
        """Verifies max_features, ngram_range, and min_df are respected."""
        vec, X, meta = extract_tfidf_features(
            self.sample_df,
            min_df=1,
            max_features=10,
            ngram_range=(1, 2),
        )
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(meta["n_features"], 10)

        # Check bigrams presence when ngram_range=(1, 2)
        has_bigram = any(" " in term for term in meta["feature_names"])
        self.assertTrue(has_bigram)

    def test_sparse_matrix_preservation(self):
        """Verifies that the matrix remains a sparse CSR matrix and is not cast to dense."""
        vec, X, meta = extract_tfidf_features(self.sample_df, min_df=1, max_df=1.0)
        self.assertTrue(isspmatrix_csr(X))
        self.assertFalse(isinstance(X, np.ndarray))
        self.assertEqual(X.format, "csr")

    def test_empty_vocabulary_errors(self):
        """Filtering that eliminates all terms raises a clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_tfidf_features(self.sample_df, min_df=100)
        self.assertIn("vectorization failed", str(ctx.exception).lower())

    def test_category_label_isolation(self):
        """A DataFrame containing a category label column is strictly rejected."""
        leaked_df = self.sample_df.copy()
        leaked_df["category"] = ["business", "tech", "sport", "politics"]
        with self.assertRaises(ValueError) as ctx:
            extract_tfidf_features(leaked_df)
        self.assertIn("label column(s) detected", str(ctx.exception).lower())

    def test_save_and_load_vectorizer_joblib(self):
        """Saving and loading via joblib preserves vectorizer and vocabulary."""
        vec, X, meta = extract_tfidf_features(self.sample_df, min_df=1, max_df=1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "test_vec.joblib"
            saved = save_tfidf_vectorizer(vec, output_path=temp_path)
            self.assertTrue(saved.exists())

            loaded_vec = load_tfidf_vectorizer(model_path=saved)
            self.assertEqual(
                vec.get_feature_names_out().tolist(),
                loaded_vec.get_feature_names_out().tolist(),
            )

            # Test transforming new text with loaded vectorizer
            new_text = ["economy interest rates"]
            trans_orig = vec.transform(new_text)
            trans_loaded = loaded_vec.transform(new_text)
            np.testing.assert_array_equal(trans_orig.toarray(), trans_loaded.toarray())

    def test_get_top_terms_helper(self):
        """Verify get_top_terms_for_document returns correct top terms sorted descending."""
        vec, X, meta = extract_tfidf_features(self.sample_df, min_df=1, max_df=1.0)
        top_terms = get_top_terms_for_document(X, meta["feature_names"], doc_index=0, top_n=3)
        self.assertEqual(len(top_terms), 3)
        # Weights should be descending
        self.assertGreaterEqual(top_terms[0][1], top_terms[1][1])
        self.assertGreaterEqual(top_terms[1][1], top_terms[2][1])

        # Test index out of bounds
        with self.assertRaises(IndexError):
            get_top_terms_for_document(X, meta["feature_names"], doc_index=99)


class TestRealBBCFeatureExtractionIntegration(unittest.TestCase):
    """Integration test executing feature extraction on real articles_clean.csv."""

    def test_real_corpus_feature_extraction(self):
        """Extract features from the real 2,123 articles and verify dimensions and sparsity."""
        root = get_project_root()
        articles_df = load_processed_articles(root)
        self.assertEqual(len(articles_df), 2123)

        vec, X, meta = extract_tfidf_features(articles_df)

        self.assertEqual(X.shape[0], 2123)
        self.assertEqual(X.shape[1], 5000)
        self.assertTrue(isspmatrix_csr(X))
        self.assertEqual(meta["empty_vector_count"], 0)
        self.assertGreater(meta["nnz"], 200000)
        self.assertGreater(meta["sparsity"], 0.95)

        # Verify fitted vectorizer can be saved and loaded
        saved_file = save_tfidf_vectorizer(vec, project_root=root)
        self.assertTrue(saved_file.exists())
        loaded = load_tfidf_vectorizer(project_root=root)
        self.assertEqual(len(loaded.get_feature_names_out()), 5000)


if __name__ == "__main__":
    unittest.main()
