from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas import user_schemas
from .. import crud
from ..database import get_db

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/predict", response_model=user_schemas.PredictionResponse)
def predict(prediction: user_schemas.PredictionCreate, db: Session = Depends(get_db)):
    """Rota legada de previsão (mantida para compatibilidade)."""
    return crud.create_prediction(db=db, prediction=prediction)