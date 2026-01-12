"""
SQLAlchemy models for the application.

This module defines all database models (tables) used in the application.
Each class represents a table in the database.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.connection import Base


class User(Base):
    """
    User model representing a user in the system.
    
    Attributes:
        id: Unique identifier for the user
        username: Unique username for login
        email: Unique email address
        hashed_password: Bcrypt hashed password
        is_active: Whether the user account is active
        created_at: Timestamp when user was created
        predictions: Relationship to user's predictions
    """
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to predictions
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class Prediction(Base):
    """
    Prediction model representing a credit prediction.
    
    Attributes:
        id: Unique identifier for the prediction
        user_id: Foreign key to the user who made the prediction
        input_data: JSON data containing the input parameters
        prediction_result: JSON data containing the prediction results
        created_at: Timestamp when prediction was made
        user: Relationship to the user who made the prediction
    """
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    input_data = Column(JSON, nullable=False)
    prediction_result = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship to user
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, user_id={self.user_id}, status='{self.prediction_result.get('status')}')>"