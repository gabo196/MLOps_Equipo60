import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================
# Configuración
# ============================================
DATA_PATH = Path("data/processed/obesity_estimation_cleaned.csv")
MODEL_PATH = Path("models/random_forest.joblib")   # Modelo exportado
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "NObesity"   # COLUMNA OBJETIVO CORRECTA

# ============================================
# Función: separar X / y
# ============================================
def split_xy(df):
    if TARGET not in df.columns:
        raise KeyError(f"La columna objetivo '{TARGET}' NO existe.\n"
                       f"Columnas disponibles: {list(df.columns)}")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

# ============================================
# Funciones para simular drift
# ============================================

def add_noise(df, pct=0.10):
    """Agrega ruido gaussiano al 10% de los datos numéricos."""
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include='number').columns

    for col in numeric_cols:
        mask = df2.sample(frac=pct).index
        df2.loc[mask, col] += np.random.normal(0, df2[col].std() * 0.3, size=len(mask))

    return df2

def category_swap(df, pct=0.15):
    """Cambia categorías en columnas categóricas."""
    df2 = df.copy()
    cat_cols = df2.select_dtypes(include='object').columns

    for col in cat_cols:
        mask = df2.sample(frac=pct).index
        df2.loc[mask, col] = df2[col].sample(frac=pct).values

    return df2

def scale_height_weight(df):
    """Simula cambio de escala en Height y Weight (ej: error de sistema)."""
    df2 = df.copy()
    if "Height" in df2.columns:
        df2["Height"] = df2["Height"] * 1.10  # +10%
    if "Weight" in df2.columns:
        df2["Weight"] = df2["Weight"] * 0.92  # -8%
    return df2

# ============================================
# MAIN
# ============================================
print("Cargando dataset base...")
df_base = pd.read_csv(DATA_PATH)
X_base, y_base = split_xy(df_base)

print("Cargando modelo...")
model = joblib.load(MODEL_PATH)

print("Predicción base...")
y_pred_base = model.predict(X_base)
acc_base = accuracy_score(y_base, y_pred_base)
f1_base = f1_score(y_base, y_pred_base, average='weighted')

print(f"✔ Accuracy base: {acc_base:.4f}")
print(f"✔ F1-score base: {f1_base:.4f}\n")


# ============================================
# Generación de escenarios de drift
# ============================================

scenarios = {
    "noise_10pct": add_noise(X_base),
    "category_swap_15pct": category_swap(X_base),
    "scale_height_weight": scale_height_weight(X_base),
    "combo_full": scale_height_weight(add_noise(category_swap(X_base)))
}

results = []

for name, X_drift in scenarios.items():
    print(f"\nSimulando drift: {name}")
    
    y_pred = model.predict(X_drift)
    acc = accuracy_score(y_base, y_pred)
    f1 = f1_score(y_base, y_pred, average='weighted')

    results.append({
        "scenario": name,
        "accuracy": acc,
        "f1_weighted": f1
    })

    # Guardar predicciones
    pred_file = OUTPUT_DIR / f"pred_{name}.csv"
    pd.DataFrame({"y_true": y_base, "y_pred": y_pred}).to_csv(pred_file, index=False)

    print(f"   ➤ Accuracy: {acc:.4f}")
    print(f"   ➤ F1-score: {f1:.4f}")

# Guardar resultados generales
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_DIR / "drift_results.csv", index=False)

print("\nSimulación de drift completada exitosamente.")
print("Resultados guardados en la carpeta: data_drift/")
