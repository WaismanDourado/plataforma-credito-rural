from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import schemas, crud
from ..database import get_db

router = APIRouter(prefix="/api", tags=["prediction"])

@router.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend rodando!"}

@router.post("/predict", response_model=schemas.PredictionResponse)
def predict(prediction: schemas.PredictionCreate, db: Session = Depends(get_db)):
    return crud.create_prediction(db=db, prediction=prediction)
