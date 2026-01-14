"""
Configuration module.

Handles application settings and environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    """
    Application settings.
    
    Loads configuration from environment variables.
    """
    
    # ============================================
    # APPLICATION CONFIGURATION
    # ============================================
    
    APP_NAME: str = Field(
        default="Plataforma de Previsão de Crédito Rural",
        env="APP_NAME"
    )
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    APP_ENV: str = Field(default="development", env="APP_ENV")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # ============================================
    # DATABASE CONFIGURATION
    # ============================================
    
    DATABASE_URL: str = Field(default="sqlite:///./test.db", env="DATABASE_URL")
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")
    
    # ============================================
    # JWT CONFIGURATION
    # ============================================
    
    SECRET_KEY: str = Field(
        default="sua-chave-secreta-super-segura-aqui-mude-em-producao",
        env="SECRET_KEY"
    )
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # ============================================
    # CORS CONFIGURATION
    # ============================================
    
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://192.168.1.217:3000",
            "http://localhost:8000",
            "http://localhost:5173"
        ],
        env="CORS_ORIGINS"
    )
    
    # ============================================
    # API CONFIGURATION
    # ============================================
    
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX")
    
    # ============================================
    # FILE PATHS CONFIGURATION
    # ============================================
    
    MODEL_DIR: str = Field(default="app/models", env="MODEL_DIR")
    DATA_DIR: str = Field(default="data/raw", env="DATA_DIR")
    
    # ============================================
    # LOGGING CONFIGURATION
    # ============================================
    
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignora campos extras do .env

# Create settings instance
settings = Settings()