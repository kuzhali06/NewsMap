"""Central API router assembling all sub-routers."""
from fastapi import APIRouter
from backend.app.api.routes import articles, health, topics

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(articles.router)
api_router.include_router(topics.router)
