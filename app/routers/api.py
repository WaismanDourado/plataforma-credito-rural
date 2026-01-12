from fastapi import APIRouter
from .health import router as health_router
from .predictions import router as predictions_router
from .users import router as users_router

# Criar router principal
api_router = APIRouter(prefix="/api")

# Incluir sub-routers
api_router.include_router(health_router)
api_router.include_router(predictions_router)
api_router.include_router(users_router)

# Exportar para main.py
router = api_router