"""
Database module for the application.

This module initializes the database connection and provides
access to the database session and base model for SQLAlchemy.
"""

from .connection import engine, SessionLocal, Base

__all__ = ["engine", "SessionLocal", "Base"]