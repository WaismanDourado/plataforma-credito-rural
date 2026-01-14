"""
Main application module.

Initializes FastAPI application with all configurations,
middleware, routers, and database setup.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import database, ml_model
from .auth import router as auth_router
from .routers import router as api_router
from .database import init_db
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# INITIALIZATION
# ============================================

# Create necessary directories
os.makedirs(ml_model.MODEL_DIR, exist_ok=True)
logger.info("✅ Model directory created/verified")

# Initialize database
init_db()
logger.info("✅ Database initialized")

# Load ML model
try:
    credit_model, credit_scaler = ml_model.load_model_and_scaler()
    logger.info("✅ ML model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading ML model: {e}")
    logger.warning("⚠️  Application will continue but predictions may not work")
    credit_model = None
    credit_scaler = None

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Plataforma de Previsão de Crédito Rural",
    description="API para previsão de crédito rural utilizando FastAPI e modelos de ML.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================
# MIDDLEWARE
# ============================================

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"✅ CORS configured for origins: {settings.CORS_ORIGINS}")

# ============================================
# ROUTERS
# ============================================

# Include authentication router
app.include_router(auth_router, prefix="/auth", tags=["auth"])
logger.info("✅ Auth router included")

# Include API router (contains health, predictions, users)
app.include_router(api_router, prefix="/api", tags=["api"])
logger.info("✅ API router included")

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/", tags=["status"])
async def root():
    """
    Root endpoint - health check.
    
    Returns:
        dict: Status message
    """
    return {
        "message": "Backend OK! 🚀",
        "status": "healthy",
        "version": "1.0.0"
    }

# ============================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    Runs when the application starts.
    """
    logger.info("=" * 50)
    logger.info("🚀 Application Starting...")
    logger.info("=" * 50)
    
    # List all available endpoints
    logger.info("📋 Available Endpoints:")
    for route in app.routes:
        methods = ', '.join(route.methods) if hasattr(route, 'methods') else 'N/A'
        path = route.path if hasattr(route, 'path') else 'N/A'
        tags = route.tags if hasattr(route, 'tags') else []
        logger.info(f"   {methods:20} {path:40} {tags}")
    
    logger.info("=" * 50)
    logger.info("✅ Application Ready!")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    Runs when the application shuts down.
    """
    logger.info("🛑 Application Shutting Down...")

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    General exception handler for unhandled errors.
    
    Args:
        request: The request object
        exc: The exception
    
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "detail": "Internal server error",
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )