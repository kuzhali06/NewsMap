"""Topic clustering schemas."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.article import ArticleSummary


class TopTermItem(BaseModel):
    """Top term with centroid weight."""
    term: str = Field(..., description="Vocabulary term")
    weight: float = Field(..., description="Centroid feature weight")


class RepresentativeArticleItem(BaseModel):
    """Representative article closest to cluster centroid."""
    article_id: int = Field(..., description="Article ID")
    distance_to_centroid: float = Field(..., description="Euclidean distance to cluster centroid")
    snippet: Optional[str] = Field(None, description="Preview snippet")


class TopicClusterItem(BaseModel):
    """Topic cluster summary."""
    cluster_id: int = Field(..., description="Cluster identifier (0-indexed integer)")
    article_count: int = Field(..., description="Number of articles assigned to this cluster")
    percentage_of_corpus: float = Field(..., description="Percentage of total corpus")
    top_terms: List[TopTermItem] = Field(..., description="Top terms ordered by centroid weight")
    representative_articles: List[RepresentativeArticleItem] = Field(
        ..., description="Representative articles closest to centroid"
    )


class TopicMetadata(BaseModel):
    """Clustering run metadata."""
    algorithm: str = Field("KMeans", description="Clustering algorithm name")
    n_clusters: int = Field(..., description="Number of clusters formed")
    total_articles: int = Field(..., description="Total articles clustered")
    n_features: Optional[int] = Field(None, description="TF-IDF vocabulary feature count")
    inertia: Optional[float] = Field(None, description="Sum of squared distances to centroids")
    n_iter: Optional[int] = Field(None, description="Iterations until convergence")
    random_state: Optional[int] = Field(None, description="Deterministic random state seed")


class TopicsResponse(BaseModel):
    """Complete topics response containing metadata and all cluster summaries."""
    metadata: TopicMetadata = Field(..., description="Clustering metadata")
    clusters: List[TopicClusterItem] = Field(..., description="List of topic clusters")


class ClusterArticlesResponse(BaseModel):
    """Paginated articles belonging to a specific cluster."""
    cluster_id: int = Field(..., description="Queried cluster ID")
    items: List[ArticleSummary] = Field(..., description="List of articles in this cluster")
    total: int = Field(..., description="Total articles in this cluster")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages in this cluster")
    has_next: bool = Field(..., description="Whether next page exists")
    has_prev: bool = Field(..., description="Whether previous page exists")
