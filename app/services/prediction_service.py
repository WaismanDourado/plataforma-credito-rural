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
    """

    _predictions_history: List[Dict[str, Any]] = []  # In-memory storage for demonstration
    _prediction_count: int = 0
    _total_confidence: float = 0.0
    _total_approved: int = 0
    _total_denied: int = 0
    _last_updated_stats: datetime = datetime.now()

    def __init__(self):
        """
        Initializes the PredictionService.
        """
        self.logger = logger
        self.logger.info("PredictionService initialized.")

    async def predict(
        self,
        prediction_input: CreditPredictionInput,
        db: Optional[Session] = None
    ) -> CreditPredictionOutput:
        """
        Performs credit approval prediction based on user input.

        Args:
            prediction_input (CreditPredictionInput): Validated input data from the user.
            db (Optional[Session]): Database session for saving predictions.

        Returns:
            CreditPredictionOutput: A Pydantic model containing the prediction results.

        Raises:
            ValueError: If input data validation fails or an internal prediction error occurs.
        """
        self.logger.info("Received request for credit approval prediction.")
        
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

            # Save prediction to history (for stats demonstration)
            self._save_prediction_to_history(response.model_dump())
            self.logger.debug("Prediction saved to history.")

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

    def _save_prediction_to_history(self, prediction_data: Dict[str, Any]) -> None:
        """
        Saves the prediction data to an in-memory history for statistics.
        In a production environment, this would interact with a database.

        Args:
            prediction_data (Dict[str, Any]): The prediction data to save.
        """
        self._predictions_history.append(prediction_data)
        
        # Update aggregate statistics
        self._prediction_count += 1
        self._total_confidence += prediction_data.get("confianca", 0.0)
        
        if prediction_data.get("status") == "Aprovado":
            self._total_approved += 1
        elif prediction_data.get("status") == "Negado":
            self._total_denied += 1
            
        self._last_updated_stats = datetime.now()
        
        self.logger.info(
            f"Prediction saved to in-memory history. "
            f"Total predictions: {self._prediction_count}"
        )

    async def get_prediction_stats(self) -> PredictionStats:
        """
        Returns aggregated statistics about all predictions made.
        For now, this uses in-memory data.

        Returns:
            PredictionStats: A Pydantic model containing prediction statistics.
        """
        self.logger.info("Retrieving prediction statistics.")
        
        if self._prediction_count == 0:
            return PredictionStats(
                total_predictions=0,
                approved_count=0,
                denied_count=0,
                approval_rate=0.0,
                average_confidence=0.0,
                average_risk_score=0.0
            )

        approval_rate = (self._total_approved / self._prediction_count) * 100
        average_confidence = self._total_confidence / self._prediction_count
        
        # Calculate average risk score (inverse of approval probability)
        risk_scores = [
            (1 - p.get("probabilidade_aprovacao", 0)) * 100 
            for p in self._predictions_history
        ]
        average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0

        stats = PredictionStats(
            total_predictions=self._prediction_count,
            approved_count=self._total_approved,
            denied_count=self._total_denied,
            approval_rate=round(approval_rate, 2),
            average_confidence=round(average_confidence, 4),
            average_risk_score=round(average_risk_score, 2)
        )
        
        self.logger.debug(f"Prediction statistics: {stats.model_dump()}")
        return stats

    async def get_prediction(self, prediction_id: int) -> Optional[CreditPredictionOutput]:
        """
        Retrieves a specific prediction from history by ID.

        Args:
            prediction_id (int): The ID of the prediction to retrieve.

        Returns:
            Optional[CreditPredictionOutput]: The prediction if found, None otherwise.
        """
        self.logger.info(f"Retrieving prediction with ID: {prediction_id}")
        
        if 0 <= prediction_id < len(self._predictions_history):
            prediction_data = self._predictions_history[prediction_id]
            return CreditPredictionOutput(**prediction_data)
        
        self.logger.warning(f"Prediction with ID {prediction_id} not found")
        return None

# Instantiate the service for use in routes
prediction_service = PredictionService()

if __name__ == "__main__":
    # Example usage (for testing the service directly)
    import asyncio

    async def test_service():
        # Dummy data for testing
        dummy_user_data = {
            "area_total_ha": 100.0,
            "area_produtiva_ha": 80.0,
            "tempo_de_conta_anos": 5,
            "receita_bruta_total_obtida_reais": 200000.0,
            "endividamento_reais": 100000.0,
            "valor_terras_reais": 500000.0,
            "valor_maquinario_reais": 150000.0,
            "valor_semoventes_reais": 50000.0,
            "regiao": "Sul",
            "tipo_atividade": "Agricultura",
            "cliente_iniciante_credito_rural": False,
            "indicador_cliente_com_dap_producao": True,
            "tipo_garantia_principal": "Hipoteca de Terra",
            "periodos_chuva_seca": "Chuva Normal",
            "efeitos_el_nino_la_nina": "Normal",
            "mes_operacao": "Junho",
            "qtd_operacoes_rurais_anteriores": 2,
            "valor_pretendido": 50000.0
        }

        print("\n--- Testing PredictionService ---")
        try:
            # Create input
            prediction_input = CreditPredictionInput(**dummy_user_data)
            
            # Test prediction
            prediction_result = await prediction_service.predict(prediction_input)
            print(f"\nPrediction Result: {prediction_result.model_dump_json(indent=2)}")

            # Test stats
            stats = await prediction_service.get_prediction_stats()
            print(f"\nPrediction Stats: {stats.model_dump_json(indent=2)}")

        except ValueError as ve:
            print(f"Service Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    asyncio.run(test_service())