"""
NewsMap Application State Manager.

Pre-loads and caches expensive artifacts (corpus, TF-IDF matrix, cluster assignments,
and related-story service) once at application startup.
Provides clean query interfaces for routes without repeated disk reads or refitting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from scipy.sparse import csr_matrix

from backend.app.core.config import settings
from backend.app.ml.feature_extraction import extract_tfidf_features
from backend.app.services.data_loader import get_project_root, load_processed_articles
from backend.app.services.related_stories import RelatedStoryService


class AppState:
    """Singleton-style application state holder."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or settings.PROJECT_ROOT
        self.articles_df: pd.DataFrame = pd.DataFrame()
        self.articles_by_id: Dict[int, Dict[str, Any]] = {}
        self.cluster_map: Dict[int, int] = {}
        self.cluster_summaries_data: Dict[str, Any] = {}
        self.related_service: Optional[RelatedStoryService] = None
        self.is_initialized: bool = False
        self.init_error: Optional[str] = None

    def initialize(self) -> None:
        """Load and cache artifacts once."""
        if self.is_initialized:
            return

        try:
            # 1. Load processed articles
            articles_path = self.root / settings.ARTICLES_CSV_PATH
            if articles_path.exists():
                df = load_processed_articles(project_root=self.root)
                self.articles_df = df.copy()
                self.articles_df["article_id"] = self.articles_df["article_id"].astype(int)
            else:
                self.articles_df = pd.DataFrame(columns=["article_id", "cleaned_text", "source", "word_count", "char_count"])

            # 2. Load cluster assignments
            cluster_path = self.root / settings.CLUSTER_ASSIGNMENTS_PATH
            self.cluster_map = {}
            if cluster_path.exists():
                try:
                    cdf = pd.read_csv(cluster_path)
                    for _, row in cdf.dropna(subset=["article_id", "cluster_id"]).iterrows():
                        self.cluster_map[int(row["article_id"])] = int(row["cluster_id"])
                except Exception:
                    self.cluster_map = {}

            # Attach cluster_id to articles_df
            if not self.articles_df.empty:
                self.articles_df["cluster_id"] = self.articles_df["article_id"].map(self.cluster_map)
                # Build fast O(1) dictionary by ID
                self.articles_by_id = self.articles_df.set_index("article_id").to_dict(orient="index")

            # 3. Load cluster summaries
            summaries_path = self.root / settings.CLUSTER_SUMMARIES_PATH
            if summaries_path.exists():
                try:
                    with open(summaries_path, "r", encoding="utf-8") as f:
                        self.cluster_summaries_data = json.load(f)
                except Exception:
                    self.cluster_summaries_data = {}
            else:
                self.cluster_summaries_data = {}

            # 4. Initialize RelatedStoryService if corpus exists
            if not self.articles_df.empty:
                try:
                    cluster_df_param = None
                    if cluster_path.exists():
                        cluster_df_param = pd.read_csv(cluster_path)

                    self.related_service = RelatedStoryService(
                        articles_df=self.articles_df,
                        cluster_assignments_df=cluster_df_param,
                        project_root=self.root,
                    )
                except Exception as exc:
                    self.related_service = None
                    self.init_error = f"RelatedStoryService initialization failed: {exc}"

            self.is_initialized = True
        except Exception as exc:
            self.init_error = str(exc)
            self.is_initialized = False

    def check_artifacts_status(self) -> Dict[str, bool]:
        """Check presence of all key artifacts on disk."""
        return {
            "articles_clean": (self.root / settings.ARTICLES_CSV_PATH).exists(),
            "cluster_assignments": (self.root / settings.CLUSTER_ASSIGNMENTS_PATH).exists(),
            "cluster_summaries": (self.root / settings.CLUSTER_SUMMARIES_PATH).exists(),
            "tfidf_vectorizer": (self.root / settings.TFIDF_VECTORIZER_PATH).exists(),
            "kmeans_model": (self.root / settings.KMEANS_MODEL_PATH).exists(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Compile health check diagnostics."""
        artifacts = self.check_artifacts_status()
        is_healthy = artifacts["articles_clean"] and self.is_initialized

        n_clusters = 0
        if self.cluster_summaries_data and "metadata" in self.cluster_summaries_data:
            n_clusters = self.cluster_summaries_data["metadata"].get("n_clusters", 0)
        elif self.cluster_map:
            n_clusters = len(set(self.cluster_map.values()))

        return {
            "status": "healthy" if is_healthy else "degraded",
            "version": settings.VERSION,
            "corpus_size": len(self.articles_df),
            "total_clusters": n_clusters,
            "artifacts": artifacts,
        }

    def get_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve single article detail by ID."""
        aid = int(article_id)
        if aid not in self.articles_by_id:
            return None
        rec = dict(self.articles_by_id[aid])
        full_text = str(rec.get("cleaned_text", ""))
        preview = full_text[:160] + ("..." if len(full_text) > 160 else "")
        cluster_val = rec.get("cluster_id")
        return {
            "article_id": aid,
            "text": full_text,
            "preview": preview,
            "word_count": int(rec.get("word_count", 0)),
            "char_count": int(rec.get("char_count", len(full_text))),
            "source": str(rec.get("source", "unknown")),
            "cluster_id": int(cluster_val) if pd.notnull(cluster_val) else None,
        }

    def get_articles(
        self,
        page: int = 1,
        page_size: int = 20,
        query: Optional[str] = None,
        cluster_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Query paginated articles with optional text search and cluster filtering.

        Returns (items, total_count).
        """
        if self.articles_df.empty:
            return [], 0

        filtered = self.articles_df

        # 1. Cluster filter
        if cluster_id is not None:
            filtered = filtered[filtered["cluster_id"] == int(cluster_id)]

        # 2. Text substring search (case-insensitive)
        if query:
            clean_q = query.strip().lower()
            if clean_q:
                mask = filtered["cleaned_text"].str.contains(clean_q, regex=False, na=False)
                filtered = filtered[mask]

        total = len(filtered)
        if total == 0:
            return [], 0

        # 3. Pagination slice
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered.iloc[start_idx:end_idx]

        items: List[Dict[str, Any]] = []
        for _, row in page_df.iterrows():
            aid = int(row["article_id"])
            full_text = str(row["cleaned_text"])
            preview = full_text[:160] + ("..." if len(full_text) > 160 else "")
            c_val = row.get("cluster_id")
            items.append(
                {
                    "article_id": aid,
                    "preview": preview,
                    "word_count": int(row.get("word_count", 0)),
                    "source": str(row.get("source", "unknown")),
                    "cluster_id": int(c_val) if pd.notnull(c_val) else None,
                }
            )

        return items, total

    def get_cluster_articles(
        self,
        cluster_id: int,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query paginated articles belonging to a cluster."""
        return self.get_articles(page=page, page_size=page_size, query=None, cluster_id=cluster_id)

    def get_topics(self) -> Dict[str, Any]:
        """Retrieve topic summaries."""
        if self.cluster_summaries_data:
            return self.cluster_summaries_data

        # Fallback if cluster_summaries.json was not generated yet
        clusters_present = sorted(list(set(self.cluster_map.values()))) if self.cluster_map else []
        return {
            "metadata": {
                "algorithm": "KMeans",
                "n_clusters": len(clusters_present),
                "total_articles": len(self.articles_df),
            },
            "clusters": [
                {
                    "cluster_id": cid,
                    "article_count": sum(1 for v in self.cluster_map.values() if v == cid),
                    "percentage_of_corpus": round(
                        (sum(1 for v in self.cluster_map.values() if v == cid) / len(self.articles_df)) * 100, 2
                    ) if len(self.articles_df) > 0 else 0.0,
                    "top_terms": [],
                    "representative_articles": [],
                }
                for cid in clusters_present
            ],
        }

    def get_related_stories(
        self,
        article_id: int,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Query related stories using the pre-initialized RelatedStoryService."""
        if not self.related_service:
            raise RuntimeError("RelatedStoryService is not available.")
        return self.related_service.find_related_stories(
            source_article_id=article_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )


_global_state: Optional[AppState] = None


def get_app_state(project_root: Optional[Path] = None) -> AppState:
    """Retrieve or initialize the global AppState."""
    global _global_state
    if _global_state is None:
        _global_state = AppState(project_root=project_root)
        _global_state.initialize()
    return _global_state
