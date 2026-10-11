"""Health check endpoint."""
from fastapi import APIRouter, Depends
from backend.app.schemas.health import HealthResponse
from backend.app.services.app_state import AppState, get_app_state

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Check backend health and artifact readiness")
def get_health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    """
    Returns backend service status and readiness of ML models and data artifacts.
    """
    status_data = state.get_health_status()
    return HealthResponse(**status_data)
