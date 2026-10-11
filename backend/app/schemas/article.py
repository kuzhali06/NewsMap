"""Article and related story schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ArticleSummary(BaseModel):
    """Compact summary of an article for listing and search results."""
    article_id: int = Field(..., description="Unique article ID")
    preview: str = Field(..., description="Short text preview snippet")
    word_count: int = Field(..., description="Total word count")
    source: str = Field(..., description="Original split source ('train' or 'test')")
    cluster_id: Optional[int] = Field(None, description="Assigned topic cluster ID")


class ArticleDetail(BaseModel):
    """Complete detail view of an article."""
    article_id: int = Field(..., description="Unique article ID")
    text: str = Field(..., description="Full cleaned article text")
    preview: str = Field(..., description="Short text preview snippet")
    word_count: int = Field(..., description="Total word count")
    char_count: int = Field(..., description="Total character count")
    source: str = Field(..., description="Original split source ('train' or 'test')")
    cluster_id: Optional[int] = Field(None, description="Assigned topic cluster ID")


class ArticleListResponse(BaseModel):
    """Paginated article list response."""
    items: List[ArticleSummary] = Field(..., description="List of articles for the current page")
    total: int = Field(..., description="Total matching articles count")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total available pages")
    has_next: bool = Field(..., description="Whether a subsequent page exists")
    has_prev: bool = Field(..., description="Whether a previous page exists")


class RelatedStoryItem(BaseModel):
    """Individual related story result item."""
    rank: int = Field(..., description="Rank in similarity results (1-indexed)")
    related_article_id: int = Field(..., description="Article ID of the related story")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    preview: str = Field(..., description="Text preview snippet of the related story")
    word_count: Optional[int] = Field(None, description="Word count of the related story")
    source: Optional[str] = Field(None, description="Dataset split source")
    cluster_id: Optional[int] = Field(None, description="Assigned topic cluster ID")
    same_cluster: Optional[bool] = Field(None, description="True if in same cluster as source")


class RelatedStoriesResponse(BaseModel):
    """Response containing related stories for a source article."""
    source_article_id: int = Field(..., description="Source article ID queried")
    total_related: int = Field(..., description="Number of related stories returned")
    top_k: int = Field(..., description="Requested top-k limit")
    min_similarity: float = Field(..., description="Applied minimum similarity threshold")
    items: List[RelatedStoryItem] = Field(..., description="Ranked list of related stories")
