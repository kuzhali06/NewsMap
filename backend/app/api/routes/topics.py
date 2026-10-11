"""Topic clustering and cluster article endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from backend.app.schemas.article import ArticleSummary
from backend.app.schemas.topic import (
    ClusterArticlesResponse,
    TopicsResponse,
)
from backend.app.services.app_state import AppState, get_app_state

router = APIRouter(prefix="/topics", tags=["Topics"])


@router.get("", response_model=TopicsResponse, summary="Get discovered topic summaries and cluster metadata")
def get_topics(state: AppState = Depends(get_app_state)) -> TopicsResponse:
    """
    Retrieve all discovered topic clusters, top weighted terms, and representative stories.
    Topic summaries are purely unsupervised and data-driven.
    """
    topics_data = state.get_topics()
    return TopicsResponse(**topics_data)


@router.get("/{cluster_id}/articles", response_model=ClusterArticlesResponse, summary="List paginated articles belonging to a topic cluster")
def get_cluster_articles(
    cluster_id: int = Path(..., ge=0, description="Topic cluster ID (0-indexed)"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)"),
    state: AppState = Depends(get_app_state),
) -> ClusterArticlesResponse:
    """
    Retrieve paginated articles assigned to a specific topic cluster.
    """
    topics = state.get_topics()
    valid_cluster_ids = [c["cluster_id"] for c in topics.get("clusters", [])]

    if cluster_id not in valid_cluster_ids and cluster_id not in state.cluster_map.values():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic cluster {cluster_id} not found.",
        )

    items, total = state.get_cluster_articles(cluster_id=cluster_id, page=page, page_size=page_size)
    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 1

    return ClusterArticlesResponse(
        cluster_id=cluster_id,
        items=[ArticleSummary(**item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1 and page <= total_pages + 1,
    )
