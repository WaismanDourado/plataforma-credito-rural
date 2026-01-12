"""
Database connection configuration.

This module sets up the SQLAlchemy engine, session factory,
and declarative base for all database models.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# Create database engine
# The engine is responsible for managing connections to the database
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG  # Print SQL queries in debug mode
)

# Create session factory
# SessionLocal is used to create new database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base
# All models will inherit from this Base class
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    
    This function is used with FastAPI's Depends() to inject
    a database session into route handlers.
    
    Yields:
        Session: A SQLAlchemy database session
        
    Example:
        @app.get("/items/")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    
    This function should be called once at application startup
    to create all tables defined in the models.
    
    Example:
        from app.database import init_db
        init_db()
    """
    Base.metadata.create_all(bind=engine)