import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from pydantic import ValidationError

# Import from your application's modules
from app.ml_model import predict_credit_approval_detailed
from app.schemas.prediction_schema import (
    CreditPredictionInput,
    CreditPredictionOutput,
    PredictionDetails,
    PredictionStats
)
from app.database.models import Prediction, User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionService:
    """
    Service class to handle business logic for credit prediction.
    Acts as a bridge between FastAPI routes and the ML model.
    Persists predictions to the database for audit trail and analytics.
    """

    def __init__(self, db: Session):
        """
        Initializes the PredictionService with a database session.
        
        Args:
            db (Session): SQLAlchemy database session for persistence.
        """
        self.db = db
        self.logger = logger
        self.logger.info("PredictionService initialized with database session.")

    async def predict(
        self,
        prediction_input: CreditPredictionInput,
        user_id: int
    ) -> CreditPredictionOutput:
        """
        Performs credit approval prediction based on user input.
        Saves the prediction to the database for persistence.

        Args:
            prediction_input (CreditPredictionInput): Validated input data from the user.
            user_id (int): The ID of the user making the prediction.

        Returns:
            CreditPredictionOutput: A Pydantic model containing the prediction results.

        Raises:
            ValueError: If input data validation fails or an internal prediction error occurs.
        """
        self.logger.info(f"Received request for credit approval prediction from user {user_id}.")
        
        try:
            # Convert Pydantic model to dictionary for the ML model
            ml_model_input = prediction_input.model_dump()

            # Call the ML model's prediction function
            ml_prediction_result = predict_credit_approval_detailed(ml_model_input)
            self.logger.info(
                f"ML model prediction result: {ml_prediction_result.get('status')} "
                f"with confidence {ml_prediction_result.get('confianca', 0):.2f}"
            )

            # Process and format the prediction result
            response = self._process_prediction_result(ml_prediction_result)
            self.logger.debug("Prediction result processed.")

            # Save prediction to database
            prediction_db = self._save_prediction_to_database(
                user_id=user_id,
                input_data=ml_model_input,
                prediction_result=response.model_dump()
            )
            self.logger.info(f"Prediction saved to database with ID: {prediction_db.id}")

            return response

        except ValidationError as e:
            self.logger.error(f"Input data validation failed: {e.errors()}")
            raise ValueError(f"Invalid input data: {e.errors()}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
            raise ValueError(f"Internal server error during prediction: {e}")

    def _process_prediction_result(self, ml_result: Dict[str, Any]) -> CreditPredictionOutput:
        """
        Formats the raw ML model output into a structured CreditPredictionOutput.

        Args:
            ml_result (Dict[str, Any]): The raw dictionary output from the ML model.

        Returns:
            CreditPredictionOutput: A Pydantic model containing the formatted prediction.
        """
        try:
            # Create PredictionDetails
            details = PredictionDetails(
                score_credito=ml_result.get("score_credito", 0),
                taxa_juros_estimada_aa=ml_result.get("taxa_juros_estimada_aa", 0),
                relacao_garantia_credito=ml_result.get("relacao_garantia_credito", 0),
                valor_garantias_oferecidas_reais=ml_result.get("valor_garantias_oferecidas_reais", 0),
                receita_ajustada_sazonalidade=ml_result.get("receita_ajustada_sazonalidade"),
                taxa_inadimplencia_historica=ml_result.get("taxa_inadimplencia_historica"),
                tempo_desde_ultima_operacao_meses=ml_result.get("tempo_desde_ultima_operacao_meses")
            )

            # Create and return CreditPredictionOutput
            processed_result = {
                "status": ml_result.get("status", "Erro na Previsão"),
                "probabilidade_aprovacao": ml_result.get("probabilidade_aprovacao", 0.0),
                "probabilidade_negacao": ml_result.get("probabilidade_negacao", 0.0),
                "modelo_utilizado": ml_result.get("modelo_utilizado", "Desconhecido"),
                "confianca": ml_result.get("confianca", 0.0),
                "detalhes": details
            }
            return CreditPredictionOutput(**processed_result)
        except Exception as e:
            self.logger.error(f"Error processing ML prediction result: {e}", exc_info=True)
            raise ValueError(f"Failed to process prediction result: {e}")

    def _save_prediction_to_database(
        self,
        user_id: int,
        input_data: Dict[str, Any],
        prediction_result: Dict[str, Any]
    ) -> Prediction:
        """
        Saves the prediction data to the database for persistence and audit trail.

        Args:
            user_id (int): The ID of the user making the prediction.
            input_data (Dict[str, Any]): The input data used for prediction.
            prediction_result (Dict[str, Any]): The prediction result from the ML model.

        Returns:
            Prediction: The saved Prediction database object.

        Raises:
            ValueError: If the user doesn't exist or database operation fails.
        """
        try:
            # Verify user exists
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User with ID {user_id} not found")

            # Create Prediction object
            prediction = Prediction(
                user_id=user_id,
                input_data=input_data,
                prediction_result=prediction_result,
                created_at=datetime.utcnow()
            )

            # Save to database
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)

            self.logger.info(
                f"Prediction saved to database. "
                f"User ID: {user_id}, Prediction ID: {prediction.id}"
            )
            return prediction

        except ValueError as ve:
            self.logger.error(f"Validation error while saving prediction: {ve}")
            self.db.rollback()
            raise
        except Exception as e:
            self.logger.error(f"Database error while saving prediction: {e}", exc_info=True)
            self.db.rollback()
            raise ValueError(f"Failed to save prediction to database: {e}")

    async def get_prediction_stats(self) -> PredictionStats:
        """
        Returns aggregated statistics about all predictions made.
        Queries the database for accurate statistics.

        Returns:
            PredictionStats: A Pydantic model containing prediction statistics.
        """
        self.logger.info("Retrieving prediction statistics from database.")
        
        try:
            # Query all predictions from database
            all_predictions = self.db.query(Prediction).all()
            
            if not all_predictions:
                return PredictionStats(
                    total_predictions=0,
                    approved_count=0,
                    denied_count=0,
                    approval_rate=0.0,
                    average_confidence=0.0,
                    average_risk_score=0.0
                )

            # Calculate statistics
            total_predictions = len(all_predictions)
            approved_count = sum(
                1 for p in all_predictions 
                if p.prediction_result.get("status") == "Aprovado"
            )
            denied_count = sum(
                1 for p in all_predictions 
                if p.prediction_result.get("status") == "Negado"
            )
            
            approval_rate = (approved_count / total_predictions * 100) if total_predictions > 0 else 0.0
            
            # Calculate average confidence
            confidences = [
                p.prediction_result.get("confianca", 0) 
                for p in all_predictions
            ]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate average risk score
            risk_scores = [
                (1 - p.prediction_result.get("probabilidade_aprovacao", 0)) * 100 
                for p in all_predictions
            ]
            average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

            stats = PredictionStats(
                total_predictions=total_predictions,
                approved_count=approved_count,
                denied_count=denied_count,
                approval_rate=round(approval_rate, 2),
                average_confidence=round(average_confidence, 4),
                average_risk_score=round(average_risk_score, 2)
            )
            
            self.logger.debug(f"Prediction statistics: {stats.model_dump()}")
            return stats

        except Exception as e:
            self.logger.error(f"Error retrieving prediction statistics: {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve prediction statistics: {e}")

    async def get_prediction(self, prediction_id: int) -> Optional[CreditPredictionOutput]:
        """
        Retrieves a specific prediction from the database by ID.

        Args:
            prediction_id (int): The ID of the prediction to retrieve.

        Returns:
            Optional[CreditPredictionOutput]: The prediction if found, None otherwise.
        """
        self.logger.info(f"Retrieving prediction with ID: {prediction_id}")
        
        try:
            prediction = self.db.query(Prediction).filter(
                Prediction.id == prediction_id
            ).first()
            
            if not prediction:
                self.logger.warning(f"Prediction with ID {prediction_id} not found")
                return None
            
            # Convert database object to Pydantic model
            return CreditPredictionOutput(**prediction.prediction_result)

        except Exception as e:
            self.logger.error(f"Error retrieving prediction: {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve prediction: {e}")

    async def get_predictions_by_user(self, user_id: int) -> List[CreditPredictionOutput]:
        """
        Retrieves all predictions made by a specific user.

        Args:
            user_id (int): The ID of the user.

        Returns:
            List[CreditPredictionOutput]: List of predictions made by the user.
        """
        self.logger.info(f"Retrieving predictions for user ID: {user_id}")
        
        try:
            predictions = self.db.query(Prediction).filter(
                Prediction.user_id == user_id
            ).order_by(Prediction.created_at.desc()).all()
            
            return [
                CreditPredictionOutput(**p.prediction_result) 
                for p in predictions
            ]

        except Exception as e:
            self.logger.error(f"Error retrieving user predictions: {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve user predictions: {e}")

    async def get_predictions_by_user_paginated(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieves predictions for a user with pagination support.

        Args:
            user_id (int): The ID of the user.
            skip (int): Number of records to skip.
            limit (int): Maximum number of records to return.

        Returns:
            Dict[str, Any]: Dictionary containing predictions and pagination info.
        """
        self.logger.info(f"Retrieving paginated predictions for user ID: {user_id}")
        
        try:
            # Get total count
            total = self.db.query(Prediction).filter(
                Prediction.user_id == user_id
            ).count()
            
            # Get paginated results
            predictions = self.db.query(Prediction).filter(
                Prediction.user_id == user_id
            ).order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
            
            return {
                "total": total,
                "skip": skip,
                "limit": limit,
                "predictions": [
                    CreditPredictionOutput(**p.prediction_result) 
                    for p in predictions
                ]
            }

        except Exception as e:
            self.logger.error(f"Error retrieving paginated predictions: {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve paginated predictions: {e}")