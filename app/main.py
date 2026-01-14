from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os

from .schemas import user_schemas

# Imports relativos
from . import database, ml_model, crud, models
from .auth import router as auth_router  # Relativo OK
from .auth import get_current_user  # Relativo OK
from .routers import routers
from .database import init_db

# Cria dirs e tables
os.makedirs(ml_model.MODEL_DIR, exist_ok=True)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Plataforma de Previsão de Crédito Rural",
    description="API para previsão de crédito rural utilizando FastAPI e modelos de ML.",
    version="1.0.0",
)

# Inicializa DB
init_db()

# CORS
origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.1.217:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ML load
credit_model, credit_scaler = ml_model.load_model_and_scaler()
print("✅ Modelo ML carregado!")

@app.get("/", tags=["Status"])
async def root():
    return {"message": "Backend OK! 🚀"}

# ✅ FIX: Register com status 201 (Created)
@app.post("/users/", response_model=user_schemas.User, status_code=201, tags=["Usuários"])  # Status 201 aqui!
def create_user(user: user_schemas.UserCreate, db: Session = Depends(database.get_db)):
    return crud.create_user(db, user)

@app.get("/users/me", response_model=user_schemas.User, status_code=200, tags=["Usuários"])
async def read_users_me(current_user: user_schemas.User = Depends(get_current_user)):
    return current_user

# ✅ FIX 404: Include router SEM prefix duplicado + debug print
app.include_router(auth_router, prefix="/auth", tags=["Auth"])

# Debug completo: Lista TODOS endpoints com métodos, paths e tags
print("✅ Routers montados! Endpoints disponíveis:")
for route in app.routes:
    methods = ', '.join(route.methods) if hasattr(route, 'methods') else 'N/A'
    path = route.path if hasattr(route, 'path') else 'N/A'
    tags = route.tags if hasattr(route, 'tags') else 'N/A'
    print(f" - Métodos: {methods} | Path: {path} | Tags: {tags}")