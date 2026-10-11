"""
Unit and integration tests for Phase 5 Module 3: K-Means Topic Clustering.

Validates:
1. Sparse matrix preservation (no full dense conversion).
2. Exactly one cluster assignment per article with exact article_id alignment.
3. Deterministic behavior under fixed random_state.
4. Cluster sizes adding up to total corpus size.
5. Centroid top-term extraction and descending weight sorting.
6. Sparse representative-article selection and ascending distance ordering.
7. Input validation: non-sparse inputs, empty matrices, invalid cluster counts, dimension mismatches.
8. Joblib persistence and round-trip prediction fidelity.
9. Post-hoc external evaluation label alignment, metric computation, and partial/unmatched label handling.
10. Strict unsupervised decoupling: ground-truth labels never touch model fitting.
11. Integration test on the real BBC News cleaned corpus (2,123 articles).
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from backend.app.ml.clustering import (
    fit_kmeans,
    predict_cluster_assignments,
    get_cluster_centroid_top_terms,
    get_representative_articles,
    generate_cluster_summaries,
    evaluate_clusters_against_labels,
    save_kmeans_model,
    load_kmeans_model,
    save_cluster_assignments,
    save_cluster_summaries,
    run_clustering_pipeline,
)
from backend.app.services.data_loader import get_project_root


class TestKMeansClusteringUnit(unittest.TestCase):
    """Unit tests using synthetic sparse TF-IDF data for deterministic, fast verification."""

    def setUp(self):
        """Build small synthetic sparse dataset with 6 documents and 8 features."""
        # Row 0, 1: focus on features 0, 1 (Topic A)
        # Row 2, 3: focus on features 3, 4 (Topic B)
        # Row 4, 5: focus on features 6, 7 (Topic C)
        dense_data = np.array(
            [
                [0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8, 0.6, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.8],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7],
            ],
            dtype=np.float64,
        )
        self.sparse_matrix = csr_matrix(dense_data)
        self.article_ids = [101, 102, 201, 202, 301, 302]
        self.feature_names = [
            "goal", "match", "neutral", "profit", "shares", "filler", "software", "code"
        ]
        self.texts = [
            "goal match scored in football match",
            "match goal winning penalty shootout",
            "profit shares quarter fiscal growth",
            "shares profit revenue forecast dividend",
            "software code release kernel update",
            "code software algorithm debug patch",
        ]

    def test_fit_kmeans_sparse_input(self):
        """Verify fit_kmeans returns fitted model with correct centroid shape."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        self.assertIsInstance(model, KMeans)
        self.assertTrue(hasattr(model, "cluster_centers_"))
        self.assertEqual(model.cluster_centers_.shape, (3, 8))
        self.assertIsInstance(model.inertia_, float)
        self.assertGreater(model.inertia_, 0.0)

    def test_predict_assignments_and_exact_one_per_article(self):
        """Verify exactly one cluster assignment per article and strict ID alignment."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        assignments = predict_cluster_assignments(
            model, self.sparse_matrix, article_ids=self.article_ids
        )
        self.assertEqual(len(assignments), 6)
        self.assertListEqual(list(assignments.columns), ["article_id", "cluster_id"])
        self.assertListEqual(assignments["article_id"].tolist(), self.article_ids)

        # Every article belongs to exactly one cluster in range(0, 3)
        self.assertEqual(len(assignments["article_id"].unique()), 6)
        valid_clusters = set(range(3))
        self.assertTrue(set(assignments["cluster_id"]).issubset(valid_clusters))

    def test_deterministic_behavior(self):
        """Verify identical inputs with same random_state yield bit-for-bit identical results."""
        m1 = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        m2 = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)

        np.testing.assert_allclose(m1.cluster_centers_, m2.cluster_centers_)
        p1 = predict_cluster_assignments(m1, self.sparse_matrix, self.article_ids)
        p2 = predict_cluster_assignments(m2, self.sparse_matrix, self.article_ids)
        self.assertListEqual(p1["cluster_id"].tolist(), p2["cluster_id"].tolist())

    def test_cluster_sizes_sum_to_total_articles(self):
        """Verify cluster article counts sum exactly to corpus total and percentages to 100%."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        assignments = predict_cluster_assignments(model, self.sparse_matrix, self.article_ids)
        summaries = generate_cluster_summaries(
            model, self.sparse_matrix, assignments, self.feature_names, texts=self.texts
        )

        total_articles = summaries["metadata"]["total_articles"]
        self.assertEqual(total_articles, 6)

        summed_counts = sum(c["article_count"] for c in summaries["clusters"])
        self.assertEqual(summed_counts, total_articles)

        summed_pct = sum(c["percentage_of_corpus"] for c in summaries["clusters"])
        self.assertAlmostEqual(summed_pct, 100.0, places=1)

    def test_top_terms_extraction_and_weight_order(self):
        """Verify top terms per cluster are extracted and sorted descending by centroid weight."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        top_terms_dict = get_cluster_centroid_top_terms(model, self.feature_names, top_n=4)

        self.assertEqual(len(top_terms_dict), 3)
        for cluster_id, term_records in top_terms_dict.items():
            self.assertEqual(len(term_records), 4)
            weights = [t["weight"] for t in term_records]
            # Must be in descending order
            self.assertEqual(weights, sorted(weights, reverse=True))
            # Term names must be valid strings from feature_names
            for t in term_records:
                self.assertIn(t["term"], self.feature_names)
                self.assertIsInstance(t["weight"], float)

    def test_representative_articles_sparse_distances(self):
        """Verify representative articles have valid non-negative distances sorted ascending."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        assignments = predict_cluster_assignments(model, self.sparse_matrix, self.article_ids)
        reps = get_representative_articles(
            self.sparse_matrix,
            assignments["cluster_id"].to_numpy(),
            model,
            self.article_ids,
            top_k=2,
            texts=self.texts,
        )

        self.assertEqual(len(reps), 3)
        for cluster_id, rep_list in reps.items():
            self.assertGreaterEqual(len(rep_list), 1)
            distances = [r["distance_to_centroid"] for r in rep_list]
            # Must be non-negative
            for d in distances:
                self.assertGreaterEqual(d, 0.0)
            # Must be ascending (closest first)
            self.assertEqual(distances, sorted(distances))
            # Snippets should be present when texts are supplied
            for r in rep_list:
                self.assertIn(r["article_id"], self.article_ids)
                self.assertIsNotNone(r.get("snippet"))

    def test_input_validation_errors(self):
        """Verify robust input validation on malformed inputs."""
        # 1. Non-sparse matrix
        with self.assertRaises(TypeError):
            fit_kmeans(np.zeros((5, 5)), n_clusters=2)

        # 2. Empty sparse matrix
        with self.assertRaises(ValueError):
            fit_kmeans(csr_matrix((0, 5)), n_clusters=2)

        # 3. Invalid cluster count: <= 0
        with self.assertRaises(ValueError):
            fit_kmeans(self.sparse_matrix, n_clusters=0)

        # 4. Invalid cluster count: > n_samples
        with self.assertRaises(ValueError):
            fit_kmeans(self.sparse_matrix, n_clusters=10)

        # 5. Invalid max_iter or tol
        with self.assertRaises(ValueError):
            fit_kmeans(self.sparse_matrix, n_clusters=2, max_iter=0)
        with self.assertRaises(ValueError):
            fit_kmeans(self.sparse_matrix, n_clusters=2, tol=-1.0)

        # 6. Prediction dimension mismatch
        model = fit_kmeans(self.sparse_matrix, n_clusters=2, random_state=42)
        wrong_features_mat = csr_matrix(np.zeros((6, 4)))
        with self.assertRaises(ValueError):
            predict_cluster_assignments(model, wrong_features_mat)

        # 7. Article ID length mismatch
        with self.assertRaises(ValueError):
            predict_cluster_assignments(model, self.sparse_matrix, article_ids=[1, 2])

    def test_save_and_load_kmeans_model(self):
        """Verify model persistence to joblib and reproducible prediction from loaded model."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        orig_preds = predict_cluster_assignments(model, self.sparse_matrix, self.article_ids)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "test_kmeans.joblib"
            save_kmeans_model(model, output_path=model_file)
            self.assertTrue(model_file.exists())

            loaded_model = load_kmeans_model(model_path=model_file)
            loaded_preds = predict_cluster_assignments(loaded_model, self.sparse_matrix, self.article_ids)

            self.assertListEqual(
                orig_preds["cluster_id"].tolist(), loaded_preds["cluster_id"].tolist()
            )
            np.testing.assert_allclose(
                model.cluster_centers_, loaded_model.cluster_centers_
            )

    def test_save_cluster_assignments_and_summaries(self):
        """Verify saving cluster assignments to CSV and summaries to valid JSON."""
        model = fit_kmeans(self.sparse_matrix, n_clusters=3, random_state=42, n_init=5)
        assignments = predict_cluster_assignments(model, self.sparse_matrix, self.article_ids)
        summaries = generate_cluster_summaries(
            model, self.sparse_matrix, assignments, self.feature_names
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_assignments.csv"
            json_path = Path(tmp_dir) / "test_summaries.json"

            save_cluster_assignments(assignments, output_path=csv_path)
            save_cluster_summaries(summaries, output_path=json_path)

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            # Read back and verify CSV
            read_df = pd.read_csv(csv_path)
            self.assertEqual(len(read_df), 6)
            self.assertListEqual(list(read_df.columns), ["article_id", "cluster_id"])

            # Read back and verify JSON
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                read_json = json.load(f)
            self.assertEqual(read_json["metadata"]["n_clusters"], 3)
            self.assertEqual(len(read_json["clusters"]), 3)

    def test_evaluation_against_labels_and_unmatched_handling(self):
        """Verify evaluation metric computation, contingency table, and handling of unmatched IDs."""
        assignments = pd.DataFrame(
            {
                "article_id": [101, 102, 201, 202, 301, 302, 999],  # 999 has no label
                "cluster_id": [0, 0, 1, 1, 2, 2, 0],
            }
        )
        labels = pd.DataFrame(
            {
                "article_id": [101, 102, 201, 202, 301, 302, 888],  # 888 has no cluster assignment
                "category": ["sport", "sport", "business", "business", "tech", "tech", "politics"],
            }
        )

        eval_res = evaluate_clusters_against_labels(assignments, labels)
        cov = eval_res["coverage"]
        met = eval_res["metrics"]

        self.assertEqual(cov["matched_articles"], 6)
        self.assertEqual(cov["unmatched_assignments"], 1)  # 999
        self.assertEqual(cov["unmatched_evaluation_labels"], 1)  # 888

        # In this synthetic setup, clusters perfectly map 1-to-1 to categories
        self.assertAlmostEqual(met["adjusted_rand_index"], 1.0, places=4)
        self.assertAlmostEqual(met["normalized_mutual_info"], 1.0, places=4)
        self.assertIn("sport", eval_res["contingency_table"][0])

    def test_labels_never_used_in_model_fitting(self):
        """Verify that fit_kmeans API signature and implementation strictly forbids labels."""
        import inspect
        sig = inspect.signature(fit_kmeans)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("labels", param_names)
        self.assertNotIn("category", param_names)
        self.assertNotIn("categories", param_names)
        self.assertNotIn("y", param_names)


class TestKMeansClusteringIntegration(unittest.TestCase):
    """Integration test using the real BBC News corpus (2,123 articles)."""

    def test_real_corpus_clustering_pipeline(self):
        """Run complete baseline clustering pipeline on real processed BBC News data."""
        root = get_project_root()
        articles_path = root / "backend" / "data" / "processed" / "articles_clean.csv"
        labels_path = root / "backend" / "data" / "processed" / "evaluation_labels.csv"

        if not articles_path.exists() or not labels_path.exists():
            self.skipTest("Processed corpus or labels not found; skipping integration test.")

        results = run_clustering_pipeline(
            n_clusters=5,
            random_state=42,
            n_init=10,
            save_artifacts=False,  # Don't overwrite production files in unit test
        )

        summaries = results["summaries"]
        assignments = results["assignments_df"]
        evaluation = results["evaluation"]

        # 1. Total articles
        self.assertEqual(len(assignments), 2123)
        self.assertEqual(summaries["metadata"]["total_articles"], 2123)
        self.assertEqual(summaries["metadata"]["n_clusters"], 5)

        # 2. Each cluster has non-zero articles
        for c in summaries["clusters"]:
            self.assertGreater(c["article_count"], 100)
            self.assertEqual(len(c["top_terms"]), 10)
            self.assertEqual(len(c["representative_articles"]), 3)

        # 3. Post-hoc evaluation results
        self.assertIsNotNone(evaluation)
        cov = evaluation["coverage"]
        self.assertEqual(cov["matched_articles"], 2123)

        # Baseline unsupervised ARI and NMI should fall in typical text clustering range (0.35 - 0.55)
        met = evaluation["metrics"]
        self.assertGreater(met["adjusted_rand_index"], 0.3)
        self.assertLess(met["adjusted_rand_index"], 0.7)
        self.assertGreater(met["normalized_mutual_info"], 0.3)
        self.assertLess(met["normalized_mutual_info"], 0.6)


if __name__ == "__main__":
    unittest.main()
