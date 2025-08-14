from sqlalchemy.orm import Session
from . import models, schemas

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