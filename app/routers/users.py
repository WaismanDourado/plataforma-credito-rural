"""
Users router module.

Handles user-related endpoints and legacy prediction endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..schemas import user_schemas
from ..schemas.prediction_schema import (
    CreditPredictionInput,
    CreditPredictionOutput
)
from ..auth import get_current_user
from ..database import get_db
from ..services.prediction_service import PredictionService

router = APIRouter(tags=["users"])

# ============================================
# LEGACY PREDICTION ENDPOINT
# ============================================

@router.post(
    "/predict",
    response_model=CreditPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Fazer previsão de crédito (Legacy)",
    description="Endpoint legado para compatibilidade com versões anteriores"
)
async def predict(
    prediction_input: CreditPredictionInput,
    current_user: user_schemas.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Faz previsão de crédito rural (endpoint legado).
    
    **Requer autenticação JWT**
    
    **Nota:** Use `/api/v1/predictions/credit-approval` para novas implementações.
    
    **Parâmetros:**
    - `prediction_input`: Dados do cliente para previsão
    
    **Retorna:**
    - Resultado da previsão com status, probabilidades e detalhes
    
    **Exemplo de uso:**
    ```json
    {
        "age": 35,
        "income": 50000,
        "loan_amount": 15000,
        "credit_history": 1
    }
    ```
    """