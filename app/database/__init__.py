"""
Database module.

Handles database connection, session management, and model initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from ..config import settings

# ============================================
# DATABASE CONFIGURATION
# ============================================

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DATABASE_ECHO
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# ============================================
# DATABASE FUNCTIONS
# ============================================

def get_db() -> Session:
    """
    Get database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize database.
    
    Creates all tables defined in models.
    """
    # Import models to register them with Base
    from .models import User, Prediction
    
    # Create all tables
    Base.metadata.create_all(bind=engine)

# ============================================
# EXPORTS
# ============================================

__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "get_db",
    "init_db"
]