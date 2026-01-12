from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List

from ..schemas.prediction_schema import (
    CreditPredictionInput,
    CreditPredictionOutput,
    PredictionStats
)
from ..database import get_db
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/v1/predictions", tags=["prediction"])

@router.post(
    "/credit-approval",
    response_model=CreditPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Fazer previsão de aprovação de crédito",
    description="Recebe dados do cliente e retorna previsão de aprovação de crédito rural"
)
async def predict_credit_approval(
    prediction_input: CreditPredictionInput,
    user_id: int = Query(..., description="ID do usuário fazendo a previsão"),
    db: Session = Depends(get_db)
):
    """Faz previsão de aprovação de crédito rural."""
    try:
        service = PredictionService(db=db)
        result = await service.predict(prediction_input, user_id=user_id)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro na previsão: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar previsão: {str(e)}"
        )

# ... outros endpoints de prediction