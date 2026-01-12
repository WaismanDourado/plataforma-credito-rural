from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

# Definindo as categorias para validação (melhora a clareza e validação)
REGIAO_CHOICES = Literal['Sul', 'Sudeste', 'Centro-Oeste', 'Nordeste', 'Norte']
TIPO_ATIVIDADE_CHOICES = Literal['Agricultura', 'Pecuária', 'Misto']
TIPO_GARANTIA_CHOICES = Literal[
    'Hipoteca de Terra', 'Penhor de Maquinario', 'Alienacao Fiduciaria',
    'Fianca Bancaria', 'Aval', 'Sem Garantia Real'
]
PERIODOS_CHUVA_SECA_CHOICES = Literal['Seca', 'Chuva Normal', 'Chuva Excessiva']
EFEITOS_EL_NINO_LA_NINA_CHOICES = Literal['Seca', 'Chuva Excessiva', 'Normal']
MES_OPERACAO_CHOICES = Literal[
    'Janeiro', 'Fevereiro', 'Marco', 'Abril', 'Maio', 'Junho',
    'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
]

class CreditPredictionInput(BaseModel):
    """
    Schema para os dados de entrada da previsão de crédito rural.
    Contém os 18 campos obrigatórios que o usuário deve preencher no frontend.
    """
    area_total_ha: float = Field(..., gt=0, description="Área total da propriedade em hectares")
    area_produtiva_ha: float = Field(..., gt=0, description="Área produtiva em hectares")
    tempo_de_conta_anos: int = Field(..., ge=0, description="Tempo de conta no banco em anos")
    receita_bruta_total_obtida_reais: float = Field(..., ge=0, description="Receita bruta total obtida em Reais")
    endividamento_reais: float = Field(..., ge=0, description="Endividamento atual em Reais")
    valor_terras_reais: float = Field(..., ge=0, description="Valor das terras em Reais")
    valor_maquinario_reais: float = Field(..., ge=0, description="Valor do maquinário em Reais")
    valor_semoventes_reais: float = Field(..., ge=0, description="Valor de semoventes (animais) em Reais")
    regiao: REGIAO_CHOICES = Field(..., description="Região do produtor")
    tipo_atividade: TIPO_ATIVIDADE_CHOICES = Field(..., description="Tipo de atividade principal")
    cliente_iniciante_credito_rural: bool = Field(..., description="Indica se o cliente é iniciante em crédito rural")
    indicador_cliente_com_dap_producao: bool = Field(..., description="Indica se o cliente possui DAP de produção")
    tipo_garantia_principal: TIPO_GARANTIA_CHOICES = Field(..., description="Tipo de garantia principal oferecida")
    periodos_chuva_seca: PERIODOS_CHUVA_SECA_CHOICES = Field(..., description="Período de chuva/seca")
    efeitos_el_nino_la_nina: EFEITOS_EL_NINO_LA_NINA_CHOICES = Field(..., description="Efeitos de El Niño/La Niña")
    mes_operacao: MES_OPERACAO_CHOICES = Field(..., description="Mês da operação de crédito")
    qtd_operacoes_rurais_anteriores: int = Field(..., ge=0, description="Quantidade de operações rurais anteriores")
    valor_pretendido: float = Field(..., gt=0, description="Valor do crédito pretendido em Reais")

    # Campos opcionais que podem ser fornecidos ou calculados pelo backend
    # Estes campos são parte do dataset completo, mas não são estritamente obrigatórios
    # para o input do usuário no frontend, pois podem ser inferidos ou ter defaults.
    produtividade_por_ha: Optional[float] = Field(None, ge=0, description="Produtividade por hectare")
    preco_medio_por_unidade: Optional[float] = Field(None, ge=0, description="Preço médio por unidade")
    receita_bruta_total_prevista_reais: Optional[float] = Field(None, ge=0, description="Receita bruta total prevista em Reais")
    sfn_qtd_instituicoes_financeira: Optional[int] = Field(None, ge=0, description="Quantidade de instituições financeiras no SFN")
    custo_producao_total_obtido_reais: Optional[float] = Field(None, ge=0, description="Custo de produção total obtido em Reais")
    custo_producao_total_previsto_reais: Optional[float] = Field(None, ge=0, description="Custo de produção total previsto em Reais")
    margem_liquida_obtida_reais: Optional[float] = Field(None, ge=0, description="Margem líquida obtida em Reais")
    margem_liquida_prevista_reais: Optional[float] = Field(None, ge=0, description="Margem líquida prevista em Reais")
    renda_liquida_mensal_demais_ocupacoes_reais: Optional[float] = Field(None, ge=0, description="Renda líquida mensal de demais ocupações em Reais")
    valor_referencial_custeio_obtido_reais: Optional[float] = Field(None, ge=0, description="Valor referencial de custeio obtido em Reais")
    valor_referencial_custeio_previsto_reais: Optional[float] = Field(None, ge=0, description="Valor referencial de custeio previsto em Reais")
    qtd_cheques_devolvidos_ultimos_12_meses: Optional[int] = Field(None, ge=0, description="Quantidade de cheques devolvidos nos últimos 12 meses")
    utilizacao_media_ch_especial_ult_12_meses: Optional[float] = Field(None, ge=0, description="Utilização média do cheque especial nos últimos 12 meses")
    sld_medio_cc_aplicacao_12_meses_reais: Optional[float] = Field(None, ge=0, description="Saldo médio em conta corrente/aplicação nos últimos 12 meses em Reais")
    saldo_medio_devedor_no_ano_reais: Optional[float] = Field(None, ge=0, description="Saldo médio devedor no ano em Reais")
    agricultura_lavouras_anuais_ha: Optional[float] = Field(None, ge=0, description="Área de lavouras anuais em hectares")
    agricultura_lavouras_perenes_ha: Optional[float] = Field(None, ge=0, description="Área de lavouras perenes em hectares")
    areas_de_preservacao_rl_app_ha: Optional[float] = Field(None, ge=0, description="Área de preservação (RL/APP) em hectares")
    construcoes_e_benfeitorias_ha: Optional[float] = Field(None, ge=0, description="Área de construções e benfeitorias em hectares")
    lagos_lagoas_ha: Optional[float] = Field(None, ge=0, description="Área de lagos/lagoas em hectares")
    pastagem_nativa_ha: Optional[float] = Field(None, ge=0, description="Área de pastagem nativa em hectares")
    pastagem_cultivada_ha: Optional[float] = Field(None, ge=0, description="Área de pastagem cultivada em hectares")
    silvicultura_florestas_comerciais_ha: Optional[float] = Field(None, ge=0, description="Área de silvicultura/florestas comerciais em hectares")
    silvicultura_mata_nativa_ha: Optional[float] = Field(None, ge=0, description="Área de silvicultura/mata nativa em hectares")
    coobrigacoes_reais: Optional[float] = Field(None, ge=0, description="Coobrigações em Reais")
    valor_outros_bens_reais: Optional[float] = Field(None, ge=0, description="Valor de outros bens em Reais")
    recursos_computaveis_reais: Optional[float] = Field(None, ge=0, description="Recursos computáveis em Reais")

    @field_validator('area_produtiva_ha')
    @classmethod
    def validate_area_produtiva(cls, v, info):
        """Valida que área produtiva não excede área total."""
        if 'area_total_ha' in info.data and v > info.data['area_total_ha']:
            raise ValueError('Área produtiva não pode ser maior que área total')
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionDetails(BaseModel):
    """
    Detalhes adicionais gerados pela previsão do modelo de ML.
    """
    score_credito: float = Field(..., ge=0, le=1000, description="Score de crédito do cliente (0-1000)")
    taxa_juros_estimada_aa: float = Field(..., gt=0, description="Taxa de juros anual estimada para o crédito")
    relacao_garantia_credito: float = Field(..., ge=0, description="Relação entre o valor da garantia e o crédito pretendido")
    valor_garantias_oferecidas_reais: float = Field(..., ge=0, description="Valor total das garantias oferecidas em Reais")
    receita_ajustada_sazonalidade: Optional[float] = Field(None, ge=0, description="Receita ajustada pela sazonalidade")
    taxa_inadimplencia_historica: Optional[float] = Field(None, ge=0, description="Taxa de inadimplência histórica")
    tempo_desde_ultima_operacao_meses: Optional[float] = Field(None, ge=0, description="Tempo desde a última operação em meses")


class CreditPredictionOutput(BaseModel):
    """
    Schema para a resposta da previsão de crédito rural.
    """
    status: Literal['Aprovado', 'Negado', 'Erro na Previsão'] = Field(..., description="Status da aprovação de crédito")
    probabilidade_aprovacao: float = Field(..., ge=0, le=1, description="Probabilidade de aprovação do crédito")
    probabilidade_negacao: float = Field(..., ge=0, le=1, description="Probabilidade de negação do crédito")
    modelo_utilizado: str = Field(..., description="Nome do modelo de ML utilizado para a previsão")
    confianca: float = Field(..., ge=0, le=1, description="Confiança da previsão (maior probabilidade)")
    detalhes: PredictionDetails = Field(..., description="Detalhes adicionais da previsão")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "Aprovado",
                "probabilidade_aprovacao": 0.87,
                "probabilidade_negacao": 0.13,
                "modelo_utilizado": "XGBoost",
                "confianca": 0.87,
                "detalhes": {
                    "score_credito": 785,
                    "taxa_juros_estimada_aa": 9.5,
                    "relacao_garantia_credito": 1.45,
                    "valor_garantias_oferecidas_reais": 145000.0,
                    "receita_ajustada_sazonalidade": 210000.0,
                    "taxa_inadimplencia_historica": 0.02,
                    "tempo_desde_ultima_operacao_meses": 12.0
                }
            }
        }


class PredictionStats(BaseModel):
    """
    Schema para estatísticas de previsões realizadas.
    """
    total_predictions: int = Field(..., ge=0, description="Total de previsões realizadas")
    approved_count: int = Field(..., ge=0, description="Total de previsões aprovadas")
    denied_count: int = Field(..., ge=0, description="Total de previsões negadas")
    approval_rate: float = Field(..., ge=0, le=100, description="Taxa de aprovação em percentual")
    average_confidence: float = Field(..., ge=0, le=1, description="Confiança média das previsões")
    average_risk_score: Optional[float] = Field(None, ge=0, le=100, description="Score de risco médio")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 150,
                "approved_count": 120,
                "denied_count": 30,
                "approval_rate": 80.0,
                "average_confidence": 0.85,
                "average_risk_score": 18.5
            }
        }