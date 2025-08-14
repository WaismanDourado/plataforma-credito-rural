from sqlalchemy import Column, Integer, String, Float, Boolean
from .database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    farmer_name = Column(String, index=True) 
    income = Column(Float)
    crop_type = Column(String)
    approval_probability = Column(Float)
    approved = Column(Boolean)
