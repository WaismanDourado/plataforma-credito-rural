"""
Prediction routes module.

Handles all credit prediction endpoints with database persistence
and JWT authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List

from ..schemas import user_schemas
from ..schemas.prediction_schema import (
    CreditPredictionInput,
    CreditPredictionOutput,
    PredictionStats
)
from ..auth import get_current_user
from ..database import get_db
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/v1/predictions", tags=["prediction"])

# ============================================
# CREDIT APPROVAL PREDICTION
# ============================================

@router.post(
    "/credit-approval",
    response_model=CreditPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Fazer previsão de aprovação de crédito",
    description="Recebe dados do cliente e retorna previsão de aprovação de crédito rural"
)
async def predict_credit_approval(
    prediction_input: CreditPredictionInput,
    current_user: user_schemas.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Faz previsão de aprovação de crédito rural.
    
    **Requer autenticação JWT**
    
    **Parâmetros:**
    - `prediction_input`: Dados do cliente para previsão
    
    **Retorna:**
    - `status`: Aprovado ou Negado
    - `probabilidade_aprovacao`: Probabilidade de aprovação (0-1)
    - `probabilidade_negacao`: Probabilidade de negação (0-1)
    - `modelo_utilizado`: Nome do modelo (GBT ou XGBoost)
    - `confianca`: Confiança da previsão (0-1)
    
    **Exemplo de uso:**
    ```json
    {
        "idade": 35,
        "renda_mensal": 5000.0,
        "historico_credito": "bom",
        "valor_emprestimo": 20000.0,
        "prazo_meses": 24
    }
    ```
    """