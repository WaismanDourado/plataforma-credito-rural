from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..schemas import user_schemas
from ..schemas.prediction_schema import (
    CreditPredictionInput,
    CreditPredictionOutput,
    PredictionStats
)
from .. import crud
from ..database import get_db
from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/api", tags=["health", "prediction"])

# ============================================
# HEALTH CHECK
# ============================================

@router.get("/health")
def health_check():
    """Verifica se o backend está rodando."""
    return {
        "status": "healthy",
        "message": "Backend rodando!",
        "version": "1.0.0"
    }

# ============================================
# ROTAS DE PREVISÃO DE CRÉDITO
# ============================================

@router.post(
    "/v1/predictions/credit-approval",
    response_model=CreditPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Fazer previsão de aprovação de crédito",
    description="Recebe dados do cliente e retorna previsão de aprovação de crédito rural"
)
def predict_credit_approval(
    prediction_input: CreditPredictionInput,
    db: Session = Depends(get_db)
):
    """
    Faz previsão de aprovação de crédito rural.
    
    **Parâmetros:**
    - `prediction_input`: Dados do cliente para previsão
    
    **Retorna:**
    - `status`: Aprovado ou Negado
    - `probabilidade_aprovacao`: Probabilidade de aprovação (0-1)
    - `probabilidade_negacao`: Probabilidade de negação (0-1)
    - `modelo_utilizado`: Nome do modelo (GBT ou XGBoost)
    - `confianca`: Confiança da previsão (0-1)
    """
    try:
        # Usar o serviço de previsão
        service = PredictionService()
        result = service.predict(prediction_input, db)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro na previsão: {str(e)}"
        )

@router.get(
    "/v1/predictions/{prediction_id}",
    response_model=CreditPredictionOutput,
    summary="Recuperar previsão anterior",
    description="Recupera uma previsão de crédito feita anteriormente"
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """
    Recupera uma previsão de crédito anterior pelo ID.
    
    **Parâmetros:**
    - `prediction_id`: ID da previsão
    
    **Retorna:**
    - Dados da previsão armazenada no banco de dados
    """
    try:
        service = PredictionService()
        result = service.get_prediction(prediction_id, db)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Previsão com ID {prediction_id} não encontrada"
            )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao recuperar previsão: {str(e)}"
        )

@router.get(
    "/v1/predictions/stats/summary",
    response_model=PredictionStats,
    summary="Estatísticas de previsões",
    description="Retorna estatísticas gerais sobre as previsões realizadas"
)
def get_prediction_stats(db: Session = Depends(get_db)):
    """
    Retorna estatísticas sobre as previsões realizadas.
    
    **Retorna:**
    - `total_predictions`: Total de previsões realizadas
    - `approved_count`: Total de previsões aprovadas
    - `denied_count`: Total de previsões negadas
    - `approval_rate`: Taxa de aprovação (%)
    - `average_confidence`: Confiança média das previsões
    """
    try:
        service = PredictionService()
        stats = service.get_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )

# ============================================
# ROTAS DE USUÁRIO (EXISTENTES)
# ============================================

@router.post("/predict", response_model=user_schemas.PredictionResponse)
def predict(prediction: user_schemas.PredictionCreate, db: Session = Depends(get_db)):
    """Rota legada de previsão (mantida para compatibilidade)."""
    return crud.create_prediction(db=db, prediction=prediction)