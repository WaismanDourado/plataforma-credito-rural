from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
def health_check():
    """Verifica se o backend está rodando."""
    return {
        "status": "healthy",
        "message": "Backend rodando!",
        "version": "1.0.0"
    }