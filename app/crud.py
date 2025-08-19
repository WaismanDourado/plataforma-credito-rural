from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_active_status(db: Session, user_id: int, is_active: bool):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user:
        db_user.is_active = is_active
        db.commit()
        db.refresh(db_user)
    return db_user

def create_prediction(db: Session, prediction: schemas.PredictionCreate):
    approval_prob = 0.75 if prediction.income > 50000 else 0.4
    approved = approval_prob >= 0.5

    db_prediction = models.Prediction(
        farmer_name=prediction.farmer_name,
        income=prediction.income,
        crop_type=prediction.crop_type,
        approval_probability=approval_prob,
        approved=approved
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction