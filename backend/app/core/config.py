"""
NewsMap Application Configuration.

Defines runtime settings, server parameters, CORS origins, and paths to ML artifacts.
"""

from pathlib import Path
from typing import List, Union
from pydantic import BaseModel


def get_default_project_root() -> Path:
    """Resolve project root directory reliably."""
    # backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> root
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "backend").is_dir() and (parent / "README.md").is_file():
            return parent
    return Path("D:/NewsMap").resolve()


class Settings(BaseModel):
    """Application configuration settings."""

    PROJECT_NAME: str = "NewsMap API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Explicit CORS origins for local development (Vite, React Create App, Next.js)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    PROJECT_ROOT: Path = get_default_project_root()

    # Relative paths to persisted ML and data artifacts
    ARTICLES_CSV_PATH: str = "backend/data/processed/articles_clean.csv"
    EVALUATION_LABELS_PATH: str = "backend/data/processed/evaluation_labels.csv"
    CLUSTER_ASSIGNMENTS_PATH: str = "backend/data/processed/cluster_assignments.csv"
    CLUSTER_SUMMARIES_PATH: str = "backend/data/processed/cluster_summaries.json"
    TFIDF_VECTORIZER_PATH: str = "backend/models/tfidf_vectorizer.joblib"
    KMEANS_MODEL_PATH: str = "backend/models/kmeans_model.joblib"

    def get_full_path(self, relative_path: str) -> Path:
        """Resolve full path against project root."""
        return self.PROJECT_ROOT / relative_path


settings = Settings()
