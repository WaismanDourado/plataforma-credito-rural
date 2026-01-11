import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÃO DE CAMINHOS
# ============================================================================

MODEL_DIR = Path("app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GBT_MODEL_PATH = MODEL_DIR / "gbt_model.joblib"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
BEST_MODEL_NAME_PATH = MODEL_DIR / "best_model_name.txt"

# ============================================================================
# VARIÁVEIS CATEGÓRICAS E NUMÉRICAS
# ============================================================================

CATEGORICAL_FEATURES = [
    'regiao',
    'cliente_iniciante_credito_rural',
    'indicador_cliente_com_dap_producao',
    'tipo_atividade',
    'ind_existencia_anotacao_cadastral',
    'indicador_operacoes_rurais_prorrogadas',
    'indicador_existencia_operacao_rural',
    'tipo_garantia_principal',
    'periodos_chuva_seca',
    'efeitos_el_nino_la_nina',
    'mes_operacao',
    'qtd_operacoes_rurais_anteriores'
]

NUMERIC_FEATURES = [
    'area_total_ha',
    'area_produtiva_ha',
    'tempo_de_conta_anos',
    'produtividade_por_ha',
    'preco_medio_por_unidade',
    'receita_bruta_total_obtida_reais',
    'endividamento_reais',
    'coobrigacoes_reais',
    'valor_terras_reais',
    'valor_maquinario_reais',
    'valor_outros_bens_reais',
    'valor_semoventes_reais',
    'recursos_computaveis_reais',
    'receita_bruta_total_prevista_reais',
    'sfn_qtd_instituicoes_financeira',
    'custo_producao_total_obtido_reais',
    'custo_producao_total_previsto_reais',
    'margem_liquida_obtida_reais',
    'margem_liquida_prevista_reais',
    'renda_liquida_mensal_demais_ocupacoes_reais',
    'valor_referencial_custeio_obtido_reais',
    'valor_referencial_custeio_previsto_reais',
    'qtd_cheques_devolvidos_ultimos_12_meses',
    'utilizacao_media_ch_especial_ult_12_meses',
    'sld_medio_cc_aplicacao_12_meses_reais',
    'saldo_medio_devedor_no_ano_reais',
    'agricultura_lavouras_anuais_ha',
    'agricultura_lavouras_perenes_ha',
    'areas_de_preservacao_rl_app_ha',
    'construcoes_e_benfeitorias_ha',
    'lagos_lagoas_ha',
    'pastagem_nativa_ha',
    'pastagem_cultivada_ha',
    'silvicultura_florestas_comerciais_ha',
    'silvicultura_mata_nativa_ha',
    'receita_ajustada_sazonalidade',
    'taxa_inadimplencia_historica',
    'tempo_desde_ultima_operacao_meses',
    'valor_garantias_oferecidas_reais',
    'relacao_garantia_credito',
    'score_credito',
    'taxa_juros_estimada_aa'
]

# ============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ============================================================================

def load_data(filepath):
    """Carrega dados do arquivo XLSX."""
    logger.info(f"Carregando dados de {filepath}...")
    try:
        df = pd.read_excel(filepath)
        logger.info(f"Dados carregados com sucesso! Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

def preprocess_data(df, fit_encoders=True, label_encoders=None):
    """
    Faz pré-processamento dos dados:
    - Codifica variáveis categóricas
    - Remove colunas desnecessárias
    - Retorna X e y
    """
    logger.info("Iniciando pré-processamento...")
    
    # Criar cópia para não modificar original
    df_processed = df.copy()
    
    # Remover coluna 'id' se existir
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
    
    # Codificar variáveis categóricas
    if fit_encoders:
        label_encoders = {}
        logger.info("Codificando variáveis categóricas...")
        
        for col in CATEGORICAL_FEATURES:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
                logger.info(f"  ✓ {col}: {len(le.classes_)} classes")
    else:
        logger.info("Usando encoders fornecidos...")
        for col in CATEGORICAL_FEATURES:
            if col in df_processed.columns:
                df_processed[col] = label_encoders[col].transform(df_processed[col].astype(str))
    
    # Separar features e target
    target_col = 'credito_aprovado'
    
    if target_col in df_processed.columns:
        # Codificar target (sim/não -> 1/0)
        y = (df_processed[target_col] == 'sim').astype(int)
        X = df_processed.drop(target_col, axis=1)
    else:
        raise ValueError(f"Coluna '{target_col}' não encontrada!")
    
    logger.info(f"Pré-processamento concluído!")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  Distribuição do target: {y.value_counts().to_dict()}")
    
    return X, y, label_encoders if fit_encoders else None

# ============================================================================
# FUNÇÕES DE TREINAMENTO
# ============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """Treina GBT e XGBoost."""
    
    logger.info("\n" + "="*70)
    logger.info("TREINANDO MODELOS")
    logger.info("="*70)
    
    # ========== GBT ==========
    logger.info("\n[1/2] Treinando Gradient Boosting...")
    gbt_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )
    gbt_model.fit(X_train, y_train)
    logger.info("✓ GBT treinado com sucesso!")
    
    # ========== XGBoost ==========
    logger.info("\n[2/2] Treinando XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    logger.info("✓ XGBoost treinado com sucesso!")
    
    return gbt_model, xgb_model

def evaluate_model(model, X_test, y_test, model_name):
    """Avalia um modelo com múltiplas métricas."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"AVALIAÇÃO: {model_name}")
    logger.info(f"{'='*70}")
    
    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"\nMétricas de Desempenho:")
    logger.info(f"  Acurácia:  {accuracy:.4f}")
    logger.info(f"  Precisão:  {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nMatriz de Confusão:")
    logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Relatório de Classificação
    logger.info(f"\nRelatório de Classificação:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Negado', 'Aprovado'])}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def compare_models(gbt_metrics, xgb_metrics):
    """Compara os dois modelos e retorna o melhor."""
    
    logger.info(f"\n{'='*70}")
    logger.info("COMPARAÇÃO DE MODELOS")
    logger.info(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame({
        'GBT': gbt_metrics,
        'XGBoost': xgb_metrics
    }).loc[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    
    logger.info(comparison_df.to_string())
    
    # Determinar melhor modelo (baseado em F1-Score)
    best_model = 'XGBoost' if xgb_metrics['f1'] > gbt_metrics['f1'] else 'GBT'
    logger.info(f"\n✓ Melhor modelo: {best_model} (F1-Score: {max(gbt_metrics['f1'], xgb_metrics['f1']):.4f})")
    
    return best_model

# ============================================================================
# FUNÇÕES DE SALVAMENTO E CARREGAMENTO
# ============================================================================

def save_models(gbt_model, xgb_model, label_encoders, best_model_name):
    """Salva os modelos e encoders."""
    
    logger.info(f"\n{'='*70}")
    logger.info("SALVANDO MODELOS")
    logger.info(f"{'='*70}\n")
    
    try:
        joblib.dump(gbt_model, GBT_MODEL_PATH)
        logger.info(f"✓ GBT salvo em {GBT_MODEL_PATH}")
        
        joblib.dump(xgb_model, XGBOOST_MODEL_PATH)
        logger.info(f"✓ XGBoost salvo em {XGBOOST_MODEL_PATH}")
        
        joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
        logger.info(f"✓ Label Encoders salvos em {LABEL_ENCODERS_PATH}")
        
        # Salvar nome do melhor modelo
        with open(BEST_MODEL_NAME_PATH, 'w') as f:
            f.write(best_model_name)
        logger.info(f"✓ Melhor modelo ({best_model_name}) registrado")
        
        # Salvar melhor modelo
        best_model = gbt_model if best_model_name == 'GBT' else xgb_model
        joblib.dump(best_model, BEST_MODEL_PATH)
        logger.info(f"✓ Melhor modelo salvo em {BEST_MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar modelos: {e}")
        raise

def load_best_model():
    """Carrega o melhor modelo treinado."""
    try:
        if not BEST_MODEL_PATH.exists():
            logger.warning("Modelo não encontrado. Treinando novo modelo...")
            train_and_save_models()
        
        model = joblib.load(BEST_MODEL_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        
        with open(BEST_MODEL_NAME_PATH, 'r') as f:
            model_name = f.read().strip()
        
        logger.info(f"✓ Modelo {model_name} carregado com sucesso!")
        return model, label_encoders, model_name
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise

# ============================================================================
# FUNÇÃO DE PREVISÃO (PARA FASTAPI)
# ============================================================================

def predict_credit_approval(user_data: dict):
    """
    Faz previsão de aprovação de crédito.
    
    Args:
        user_data: Dicionário com dados do usuário
        
    Returns:
        Dicionário com previsão e probabilidade
    """
    try:
        # Carregar modelo e encoders
        model, label_encoders, model_name = load_best_model()
        
        # Converter dados do usuário em DataFrame
        df_user = pd.DataFrame([user_data])
        
        # Pré-processar dados
        X_user, _, _ = preprocess_data(df_user, fit_encoders=False, label_encoders=label_encoders)
        
        # Fazer previsão
        prediction = model.predict(X_user)[0]
        probability = model.predict_proba(X_user)[0]
        
        return {
            'status': 'Aprovado' if prediction == 1 else 'Negado',
            'probabilidade_aprovacao': float(probability[1]),
            'probabilidade_negacao': float(probability[0]),
            'modelo_utilizado': model_name,
            'confianca': float(max(probability))
        }
        
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        return {
            'erro': str(e),
            'status': 'Erro na previsão'
        }

# ============================================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ============================================================================

def train_and_save_models(data_path="dados_credito_rural.xlsx"):
    """Função principal que treina e salva os modelos."""
    
    logger.info("\n" + "="*70)
    logger.info("INICIANDO TREINAMENTO DE MODELOS DE CRÉDITO RURAL")
    logger.info("="*70 + "\n")
    
    try:
        # 1. Carregar dados
        df = load_data(data_path)
        
        # 2. Pré-processar
        X, y, label_encoders = preprocess_data(df, fit_encoders=True)
        
        # 3. Dividir em treino e teste
        logger.info("\nDividindo dados em treino (80%) e teste (20%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"  Treino: {X_train.shape[0]} amostras")
        logger.info(f"  Teste:  {X_test.shape[0]} amostras")
        
        # 4. Treinar modelos
        gbt_model, xgb_model = train_models(X_train, X_test, y_train, y_test)
        
        # 5. Avaliar modelos
        gbt_metrics = evaluate_model(gbt_model, X_test, y_test, "Gradient Boosting")
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        # 6. Comparar modelos
        best_model = compare_models(gbt_metrics, xgb_metrics)
        
        # 7. Salvar modelos
        save_models(gbt_model, xgb_model, label_encoders, best_model)
        
        logger.info("\n" + "="*70)
        logger.info("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("="*70 + "\n")
        
        return gbt_model, xgb_model, label_encoders
        
    except Exception as e:
        logger.error(f"\n✗ Erro durante o treinamento: {e}")
        raise

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Treinar modelos
    train_and_save_models("dados_credito_rural.xlsx")