"""Article listing, details, and related story endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from backend.app.schemas.article import (
    ArticleDetail,
    ArticleListResponse,
    ArticleSummary,
    RelatedStoriesResponse,
    RelatedStoryItem,
)
from backend.app.services.app_state import AppState, get_app_state

router = APIRouter(prefix="/articles", tags=["Articles"])


@router.get("", response_model=ArticleListResponse, summary="List paginated articles with optional search and cluster filter")
def list_articles(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    query: Optional[str] = Query(None, min_length=1, max_length=200, description="Optional text search query"),
    cluster_id: Optional[int] = Query(None, ge=0, description="Optional cluster ID filter"),
    state: AppState = Depends(get_app_state),
) -> ArticleListResponse:
    """
    Retrieve a paginated list of articles from the processed corpus.
    Supports case-insensitive text search and filtering by topic cluster ID.
    """
    items, total = state.get_articles(page=page, page_size=page_size, query=query, cluster_id=cluster_id)
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 1

    return ArticleListResponse(
        items=[ArticleSummary(**item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1 and page <= total_pages + 1,
    )


@router.get("/{article_id}", response_model=ArticleDetail, summary="Get full article details by ID")
def get_article(
    article_id: int = Path(..., description="Unique article ID"),
    state: AppState = Depends(get_app_state),
) -> ArticleDetail:
    """
    Retrieve complete details for an individual news article.
    """
    article = state.get_article(article_id=article_id)
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found in the corpus.",
        )
    return ArticleDetail(**article)


@router.get("/{article_id}/related", response_model=RelatedStoriesResponse, summary="Find top related stories for an article")
def get_related_stories(
    article_id: int = Path(..., description="Source article ID"),
    top_k: int = Query(5, ge=1, le=50, description="Number of related stories to return (1-50)"),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0, description="Minimum cosine similarity threshold (0.0 to 1.0)"),
    state: AppState = Depends(get_app_state),
) -> RelatedStoriesResponse:
    """
    Retrieve top-k related news stories for a source article using cosine similarity
    over sparse TF-IDF feature vectors. The source article is strictly excluded from results.
    """
    # 1. Verify source article exists
    article = state.get_article(article_id=article_id)
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source article {article_id} not found in the corpus.",
        )

    # 2. Retrieve related stories via service
    try:
        results = state.get_related_stories(
            article_id=article_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return RelatedStoriesResponse(
        source_article_id=article_id,
        total_related=len(results),
        top_k=top_k,
        min_similarity=min_similarity,
        items=[RelatedStoryItem(**r) for r in results],
    )
