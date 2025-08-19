from sqlalchemy import Column, Integer, String, Float, Boolean
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    farmer_name = Column(String, index=True) 
    income = Column(Float)
    crop_type = Column(String)
    approval_probability = Column(Float)
    approved = Column(Boolean)
