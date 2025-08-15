from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd
import os

from .import schemas
from .database import engine, Base, get_db
from .import ml_model
#from .crud import get_user, create_user

if not os.path.exists(ml_model.MODEL_DIR):
    os.makedirs(ml_model.MODEL_DIR, exist_ok=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Plataforma de Previsão de Crédito Rural")

# --- Load the ML model at application startup ---
# This ensures that the model is loaded only once when FastAPI starts
credit_model, credit_scaler = ml_model.load_model_and_scaler()
print("Modelo de previsão de crédito carregado e pronto!")

@app.get("/", tags=["Status"])
async def read_root():
    return {"message": "Bem-vindo à Plataforma de Previsão de Crédito Rural! Backend OK."}

@app.post("/predict/credit", response_model=schemas.CreditPredictionResult, tags=["Previsão ML"])
async def predict_credit_approval(
    application_data:schemas.CreditApplication
):
    """
    Makes a prediction of rural credit approval based on the data provided.
    """
    try:
        input_df=pd.DataFrame([application_data.dict()])
        input_df = input_df[['income', 'years_farming', 'area_hectares']]

        input_scaled = credit_scaler.transform(input_df)

        prediction = credit_model.predict(input_scaled)[0]
        prediction_proba = credit_model.predict_proba(input_scaled)[0][1]

        return schemas.CreditPredictionResult(
            predicted_approval=int(prediction),
            probability=float(prediction_proba)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar a previsão: {str(e)}"
        )