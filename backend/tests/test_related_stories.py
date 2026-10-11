"""
Unit and integration tests for Phase 5 Module 4: Related-Story Detection.

Validates:
1. Correct descending similarity score ordering.
2. Strict exclusion of the source article from results.
3. Correct article-ID-to-row mapping.
4. Top-k limiting and min_similarity threshold filtering.
5. Error handling for unknown article IDs and invalid parameters.
6. Safe handling of zero-feature / empty TF-IDF vectors.
7. No duplicate article IDs returned.
8. Deterministic tie-breaking for identical similarity scores.
9. Cluster metadata joining (cluster_id, same_cluster).
10. Complete independence from ground-truth category labels.
11. Integration test on the real cleaned BBC News corpus (2,123 articles).
"""

import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from backend.app.services.related_stories import (
    RelatedStoryService,
    compute_similarity_scores,
    get_related_stories,
)
from backend.app.services.data_loader import get_project_root


class TestRelatedStoryServiceUnit(unittest.TestCase):
    """Unit tests using a small synthetic corpus with known similarity relationships."""

    def setUp(self):
        """Create a synthetic 5-document corpus with known vocabulary and clusters."""
        # Doc 101: highly related to Doc 102 (shares features 0, 1)
        # Doc 102: shares features 0, 1 with Doc 101
        # Doc 103: shares feature 0 with 101, but also feature 2
        # Doc 201: completely different topic (features 3, 4)
        # Doc 202: all-zero vector (zero active features)
        dense_data = np.array(
            [
                [0.8, 0.6, 0.0, 0.0, 0.0],  # 101
                [0.8, 0.6, 0.0, 0.0, 0.0],  # 102 (identical direction to 101)
                [0.6, 0.0, 0.8, 0.0, 0.0],  # 103 (partial overlap with 101)
                [0.0, 0.0, 0.0, 0.7, 0.7],  # 201 (zero overlap with 101)
                [0.0, 0.0, 0.0, 0.0, 0.0],  # 202 (all-zero vector)
            ],
            dtype=np.float64,
        )
        self.tfidf_matrix = csr_matrix(dense_data)
        self.article_ids = [101, 102, 103, 201, 202]

        self.articles_df = pd.DataFrame(
            {
                "article_id": self.article_ids,
                "cleaned_text": [
                    "football goal scored in thrilling football match",
                    "football match goal scored penalty",
                    "football referee yellow card stadium",
                    "bank interest rate economy inflation",
                    "random empty unseen token xyz",
                ],
                "source": ["train", "train", "test", "train", "test"],
                "word_count": [7, 5, 5, 5, 5],
            }
        )

        self.clusters_df = pd.DataFrame(
            {
                "article_id": [101, 102, 103, 201, 202],
                "cluster_id": [0, 0, 0, 1, 1],
            }
        )

        self.service = RelatedStoryService(
            articles_df=self.articles_df,
            tfidf_matrix=self.tfidf_matrix,
            article_ids=self.article_ids,
            cluster_assignments_df=self.clusters_df,
        )

    def test_descending_similarity_ordering(self):
        """Verify results are sorted strictly in descending order of similarity score."""
        results = self.service.find_related_stories(source_article_id=101, top_k=5)
        self.assertGreater(len(results), 0)
        scores = [r["similarity_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_source_article_exclusion(self):
        """Verify the source article itself is never returned in related stories."""
        for aid in [101, 102, 103, 201]:
            results = self.service.find_related_stories(source_article_id=aid, top_k=5)
            retrieved_ids = [r["related_article_id"] for r in results]
            self.assertNotIn(aid, retrieved_ids)

    def test_top_k_limits(self):
        """Verify top_k bounds the number of returned stories correctly."""
        results_k1 = self.service.find_related_stories(source_article_id=101, top_k=1)
        self.assertEqual(len(results_k1), 1)

        results_k2 = self.service.find_related_stories(source_article_id=101, top_k=2)
        self.assertEqual(len(results_k2), 2)

        # Asking for more than available candidates returns all valid non-zero matches
        results_k10 = self.service.find_related_stories(source_article_id=101, top_k=10)
        # Articles 102 (sim=1.0) and 103 (sim=0.48) have similarity > 0; 201 and 202 have 0.0
        self.assertEqual(len(results_k10), 2)

    def test_min_similarity_threshold(self):
        """Verify stories below min_similarity are filtered out."""
        # 102 has score 1.0; 103 has score 0.48
        results_high_thresh = self.service.find_related_stories(
            source_article_id=101, min_similarity=0.7
        )
        self.assertEqual(len(results_high_thresh), 1)
        self.assertEqual(results_high_thresh[0]["related_article_id"], 102)

        results_low_thresh = self.service.find_related_stories(
            source_article_id=101, min_similarity=0.3
        )
        self.assertEqual(len(results_low_thresh), 2)

    def test_unknown_article_id_raises_key_error(self):
        """Verify querying an unknown article ID raises KeyError with clear message."""
        with self.assertRaises(KeyError):
            self.service.find_related_stories(source_article_id=9999)

    def test_invalid_parameters_raise_value_error(self):
        """Verify invalid top_k and min_similarity values raise ValueError."""
        with self.assertRaises(ValueError):
            self.service.find_related_stories(source_article_id=101, top_k=0)

        with self.assertRaises(ValueError):
            self.service.find_related_stories(source_article_id=101, top_k=-2)

        with self.assertRaises(ValueError):
            self.service.find_related_stories(source_article_id=101, min_similarity=-0.1)

        with self.assertRaises(ValueError):
            self.service.find_related_stories(source_article_id=101, min_similarity=1.5)

    def test_zero_feature_article_returns_empty_list(self):
        """Verify an article with an all-zero TF-IDF vector returns an empty list cleanly."""
        results = self.service.find_related_stories(source_article_id=202, top_k=5)
        self.assertEqual(results, [])

    def test_no_duplicate_result_ids(self):
        """Verify each related article ID is unique in the returned list."""
        results = self.service.find_related_stories(source_article_id=101, top_k=5)
        retrieved_ids = [r["related_article_id"] for r in results]
        self.assertEqual(len(retrieved_ids), len(set(retrieved_ids)))

    def test_deterministic_tie_breaking(self):
        """Verify tied similarity scores are broken deterministically by article_id ascending."""
        # Create synthetic corpus where docs 20 and 10 have identical vectors
        dense = np.array(
            [
                [1.0, 0.0],  # 1 (source)
                [1.0, 0.0],  # 20 (tied with 10)
                [1.0, 0.0],  # 10 (tied with 20)
            ]
        )
        sparse = csr_matrix(dense)
        df = pd.DataFrame(
            {
                "article_id": [1, 20, 10],
                "cleaned_text": ["text a", "text b", "text c"],
            }
        )
        svc = RelatedStoryService(articles_df=df, tfidf_matrix=sparse, article_ids=[1, 20, 10])
        res = svc.find_related_stories(source_article_id=1, top_k=2)

        # Both have similarity 1.0; deterministic secondary tie-breaker orders article 10 before 20
        self.assertEqual(res[0]["related_article_id"], 10)
        self.assertEqual(res[1]["related_article_id"], 20)

    def test_cluster_metadata_join(self):
        """Verify cluster_id and same_cluster boolean are accurately joined."""
        results = self.service.find_related_stories(source_article_id=101, top_k=5)
        self.assertGreater(len(results), 0)

        # 101 is in Cluster 0. 102 and 103 are also in Cluster 0.
        for r in results:
            self.assertEqual(r["cluster_id"], 0)
            self.assertTrue(r["same_cluster"])

    def test_category_labels_never_affect_retrieval(self):
        """Verify category labels are not used in retrieval or ranking."""
        # Corpus DataFrame has no category column
        self.assertNotIn("category", self.articles_df.columns)
        results = self.service.find_related_stories(source_article_id=101)
        for r in results:
            self.assertNotIn("category", r)

    def test_convenience_function_api(self):
        """Verify top-level get_related_stories function works identically to service method."""
        res_func = get_related_stories(
            source_article_id=101,
            top_k=2,
            articles_df=self.articles_df,
            tfidf_matrix=self.tfidf_matrix,
            article_ids=self.article_ids,
            cluster_assignments_df=self.clusters_df,
        )
        res_svc = self.service.find_related_stories(source_article_id=101, top_k=2)
        self.assertEqual(len(res_func), len(res_svc))
        self.assertEqual(res_func[0]["related_article_id"], res_svc[0]["related_article_id"])
        self.assertEqual(res_func[0]["similarity_score"], res_svc[0]["similarity_score"])


class TestRelatedStoryServiceIntegration(unittest.TestCase):
    """Integration test using the real BBC News corpus (2,123 articles)."""

    def setUp(self):
        root = get_project_root()
        articles_file = root / "backend" / "data" / "processed" / "articles_clean.csv"
        if not articles_file.exists():
            self.skipTest("Processed corpus not found; skipping integration test.")
        self.service = RelatedStoryService(project_root=root)

    def test_real_corpus_retrieval_worldcom_trial(self):
        """Test retrieving related stories for Article 1833 (WorldCom ex-boss Bernie Ebbers)."""
        source_id = 1833
        results = self.service.find_related_stories(source_article_id=source_id, top_k=5)

        self.assertEqual(len(results), 5)
        # Verify rank sequence
        self.assertListEqual([r["rank"] for r in results], [1, 2, 3, 4, 5])

        # Verify source article is excluded
        retrieved_ids = [r["related_article_id"] for r in results]
        self.assertNotIn(source_id, retrieved_ids)

        # High similarity for top related stories on this prominent legal story (> 0.50)
        self.assertGreater(results[0]["similarity_score"], 0.50)

        # Top related articles should include known WorldCom/Ebbers trial stories (e.g. 1238, 1413, 1143)
        known_related = {1238, 1413, 1143, 1573, 2}
        self.assertTrue(any(r["related_article_id"] in known_related for r in results))

        # Check metadata fields
        for r in results:
            self.assertIn("similarity_score", r)
            self.assertIn("preview", r)
            self.assertIn("cluster_id", r)
            self.assertIsInstance(r["same_cluster"], bool)


if __name__ == "__main__":
    unittest.main()
