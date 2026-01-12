"""
Application configuration.

This module loads configuration from environment variables.
Uses simple os.getenv() instead of Pydantic Settings to avoid parsing issues.
"""

import os
from typing import List
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


class Settings:
    """
    Application settings and configuration.
    
    Simple configuration class that loads from environment variables.
    No Pydantic complexity, just straightforward Python.
    """
    
    # 
    # GENERAL APPLICATION SETTINGS
    # 
    
    APP_NAME: str = os.getenv("APP_NAME", "Plataforma de Crédito Rural")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    
    # 
    # DATABASE CONFIGURATION
    # 
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # 
    # FILE PATHS
    # 
    
    MODEL_DIR: str = os.getenv("MODEL_DIR", "app/models")
    DATA_DIR: str = os.getenv("DATA_DIR", "data/raw")
    
    # 
    # LOGGING CONFIGURATION
    # 
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 
    # CORS CONFIGURATION
    # 
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
    ]
    
    # 
    # API CONFIGURATION
    # 
    
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    
    # 
    # SECURITY CONFIGURATION
    # 
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30


# Global instance of settings
settings = Settings()