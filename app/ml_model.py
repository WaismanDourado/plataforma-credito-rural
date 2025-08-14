import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "app/credit_approval_model.joblib"

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic data for rural credit approval prediction.

    Features:
        - income: Annual income of the applicant
        - years_farming: Number of years the applicant has been farming
        - area_hectares: Area of land owned by the applicant (in hectares)

    Target Variable:
        - approved: 1 if approved, 0 if not approved
    """

    np.random.seed(42)  # For reproducibility

    # Generate features
    income = np.random.normal(loc=50000, scale=20000, size=num_samples)
    years_farming = np.random.randint(1, 30, size=num_samples)
    area_hectares = np.random.normal(loc=50, scale=30, size=num_samples)

    # Simulate approval logic (target)
    # Simple logic: higher income and more years of farming increases chances of approval
    # Adding some random noise
    approved_probability = (
        0.3 * (income / income.max()) +
        0.4 * (years_farming / years_farming.max()) +
        0.3 * (area_hectares / area_hectares.max()) +
        np.random.uniform(-0.2, 0.2, size=num_samples)  # Noise
    )

    approved = (approved_probability > 0.5).astype(int) # Converts to 0 or 1

    # Create a DataFrame
    df = pd.DataFrame({
        "income": income,
        "years_farming": years_farming,
        "area_hectares": area_hectares,
        "approved": approved
    })

    # Adjust values to be more realistic (e.g. non-negative income)
    df['income'] = df['income'].apply(lambda x: max(1000, x))
    df['area_hectares'] = df['area_hectares'].apply(lambda x: max(1, x))

    return df

def train_and_save_model():
    """
    Generates data, trains a Logistic Regression model, and saves it.
    """

    print("Gerando dados sintéticos...")
    data = generate_synthetic_data()

    X = data[["income", "years_farming", "area_hectares"]]
    y = data["approved"]

    print("Dividindo os dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Treinando o modelo de Regressão Logística...")
    model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
    model.fit(X_train, y_train)

    # Simple Avaliation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácio do modelo: {accuracy:.2f}")

    print(f"Salvando o modelo em {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Modelo salvo com sucesso.")
    return model

def load_model():
    """
    Load the trained model from disk.
    """
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modelo carregado com sucesso")
        return model
    except FileNotFoundError:
        print("Modelo não encontrado. Treinando e salvando um novo modelo...")
        return train_and_save_model()
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}. Treinando e salvando um novo modelo...")
        return train_and_save_model()

if __name__ == "__main__":
    # When this file is run directly, it trains and saves the model
    train_and_save_model()
