from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import engine, Base, get_db
from .routers import api

app = FastAPI(title="Plataforma de Previsão de Crédito Rural")

# Create DB tables
Base.metadata.create_all(bind=engine)

# Include API routers
app.include_router(api.router)
