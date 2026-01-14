"""
Main API router module.

Combines all sub-routers into a single API router.
"""

from fastapi import APIRouter

from .health import router as health_router
from .predictions import router as predictions_router
from .users import router as users_router

# Create main API router
api_router = APIRouter(prefix="/api")

# Include sub-routers
api_router.include_router(health_router, prefix="", tags=["health"])
api_router.include_router(predictions_router, prefix="/v1", tags=["predictions"])
api_router.include_router(users_router, prefix="/users", tags=["users"])

# Export for main.py
router = api_router