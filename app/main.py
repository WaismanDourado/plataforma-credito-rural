from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
import pandas as pd
import os

# ✅ Todos imports relativos (já corrigidos)
from . import schemas
from . import database
from . import ml_model
from . import crud
from . import models
from . import auth  # ✅ Tem get_current_user (não get_current_active_user)
from app.auth import router as auth_router

if not os.path.exists(ml_model.MODEL_DIR):
    os.makedirs(ml_model.MODEL_DIR, exist_ok=True)

database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Plataforma de Previsão de Crédito Rural")

# CORS
origins = [
    "http://localhost", "http://localhost:8080", "http://127.0.0.1:8080",
    "http://127.0.0.1", "http://localhost:3000", "http://127.0.0.1:3000",
    "https://meu-dominio.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML model load
credit_model, credit_scaler = ml_model.load_model_and_scaler()
print("Modelo de previsão de crédito carregado e pronto!")

@app.get("/", tags=["Status"])
async def read_root():
    return {"message": "Bem-vindo à Plataforma de Previsão de Crédito Rural! Backend OK."}

@app.post("/users/", response_model=schemas.User, tags=["Usuários"], status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    return crud.create_user(db=db, user=user)

# ✅ CORREÇÃO: auth.get_current_user (função que EXISTE no seu auth.py)
@app.get("/users/me", response_model=schemas.User, tags=["Usuários"])
async def read_users_me(current_user: schemas.User = Depends(auth.get_current_user)):
    return current_user

@app.post("/predict/credit", response_model=schemas.CreditPredictionResult, tags=["Previsão ML"])
async def predict_credit_approval(application_data: schemas.CreditApplication):
    try:
        input_df = pd.DataFrame([application_data.dict()])
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

app.include_router(auth_router)