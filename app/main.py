from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import os

# ✅ FIX 1: IMPORTS RELATIVOS (resolve 404 router)
from . import schemas, database, ml_model, crud, models, auth

# ✅ FIX 1: Relativo (não app.auth!)
from .auth import router as auth_router  # Monta /auth/*

# ML dir + tables
os.makedirs(ml_model.MODEL_DIR, exist_ok=True)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Plataforma de Previsão de Crédito Rural")

# CORS LAN + localhost
origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.1.217:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ML load
credit_model, credit_scaler = ml_model.load_model_and_scaler()
print("✅ Modelo ML carregado!")

@app.get("/", tags=["Status"])
async def root(): return {"message": "Backend OK! 🚀"}

# ✅ /users/ (SQLAlchemy + crud)
@app.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED, tags=["Usuários"])
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    return crud.create_user(db, user)

@app.get("/users/me", response_model=schemas.User, tags=["Usuários"])
async def read_users_me(current_user: schemas.User = Depends(auth.get_current_user)):
    return current_user

# ML predict
@app.post("/predict/credit", response_model=schemas.CreditPredictionResult, tags=["ML"])
async def predict_credit(application_data: schemas.CreditApplication):
    input_df = pd.DataFrame([application_data.dict()])[['income', 'years_farming', 'area_hectares']]
    input_scaled = credit_scaler.transform(input_df)
    prediction = credit_model.predict(input_scaled)[0]
    proba = credit_model.predict_proba(input_scaled)[0][1]
    return schemas.CreditPredictionResult(predicted_approval=int(prediction), probability=float(proba))

# ✅ FIX 1: Monta auth router (agora /auth/token visível)
app.include_router(auth_router, prefix="/auth", tags=["Auth"])