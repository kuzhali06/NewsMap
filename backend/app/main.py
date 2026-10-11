"""
NewsMap FastAPI Application Entry Point (Phase 6, Module 1).

Exposes NewsMap text processing, topic clustering, and related-story discovery
services via RESTful API endpoints for downstream frontend clients.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.app.api.api import api_router
from backend.app.core.config import settings
from backend.app.services.app_state import get_app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Pre-warms and caches expensive ML and corpus resources once on startup.
    """
    # Startup: Initialize AppState once
    state = get_app_state()
    yield
    # Shutdown logic (if needed)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "NewsMap REST API for unsupervised news topic clustering and related-story discovery "
        "using the BBC News dataset."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS for local development with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean, consistent validation error handler without exposing internals."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        errors.append({"field": loc, "message": err.get("msg", "Invalid value")})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": errors},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all internal server error handler.
    Prevents leaking internal filesystem paths, stack traces, or credentials to clients.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# Mount main API router under /api
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Root"], summary="API Root and Documentation Link")
def root_endpoint():
    """Root endpoint welcoming clients and directing to interactive documentation."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} v{settings.VERSION}",
        "docs_url": "/docs",
        "health_url": f"{settings.API_PREFIX}/health",
        "api_prefix": settings.API_PREFIX,
    }


def start_server():
    """Start uvicorn server programmatically."""
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
