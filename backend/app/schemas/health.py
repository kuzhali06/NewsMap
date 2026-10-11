"""Health check schemas."""
from typing import Dict
from pydantic import BaseModel, Field


class ArtifactStatus(BaseModel):
    """Status of required and optional backend artifacts."""
    articles_clean: bool = Field(..., description="Processed articles CSV availability")
    cluster_assignments: bool = Field(..., description="Cluster assignments CSV availability")
    cluster_summaries: bool = Field(..., description="Cluster summaries JSON availability")
    tfidf_vectorizer: bool = Field(..., description="Fitted TF-IDF vectorizer model availability")
    kmeans_model: bool = Field(..., description="Fitted KMeans model availability")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    version: str = Field(..., description="API version")
    corpus_size: int = Field(..., description="Number of loaded articles")
    total_clusters: int = Field(..., description="Number of topic clusters")
    artifacts: ArtifactStatus = Field(..., description="Artifact readiness flags")
