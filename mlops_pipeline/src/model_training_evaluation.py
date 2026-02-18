import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score, 
    recall_score, 
    f1_score,
    precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# IMPORTAR DESDE ft_engineering.py
from ft_engineering import preparar_datos, crear_preprocessor

# 1. PREPARAR DATOS
print("PREPARANDO DATOS PARA MODELADO")

X_train, X_test, y_train, y_test,preprocessor = preparar_datos()

# 2. FUNCIONES DE UTILIDAD
def build_model(algorithm, preprocessor):
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', algorithm)
    ])
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def summarize_classification(model, X_test, y_test, model_name="Modelo"):
    y_pred = model.predict(X_test)
    
    print(f"EVALUACIÓN: {model_name}")
    print(classification_report(y_test, y_pred))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.tight_layout()
    plt.show()

def crear_tabla_resumen(modelos, nombres, X_test, y_test):
    resumen = []
    for mod, nombre in zip(modelos, nombres):
        preds = mod.predict(X_test)
        resumen.append({
            'Modelo': nombre,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds)
        })
    return pd.DataFrame(resumen).sort_values('F1-Score', ascending=False)

def graficar_comparacion(tabla_resumen):
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(tabla_resumen))
    width = 0.2
    
    for i, metrica in enumerate(metricas):
        offset = width * (i - 1.5)
        ax.bar(x + offset, tabla_resumen[metrica], width, label=metrica)
    
    ax.set_xlabel('Modelos', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Comparación de Modelos - Todas las Métricas', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tabla_resumen['Modelo'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 3. ENTRENAMIENTO DE MODELOS
print("ENTRENANDO MODELOS")

modelos_config = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

modelos_entrenados = {}

for nombre, algoritmo in modelos_config.items():
    print(f"\nEntrenando {nombre}...")
    modelo = build_model(algoritmo, crear_preprocessor())  # Nuevo preprocessor por modelo
    modelos_entrenados[nombre] = modelo
    print(f"{nombre} entrenado")

# 4. EVALUACIÓN INDIVIDUAL

print("EVALUACIÓN INDIVIDUAL DE MODELOS")

for nombre, modelo in modelos_entrenados.items():
    summarize_classification(modelo, X_test, y_test, nombre)

# 5. COMPARACIÓN DE MODELOS
print("TABLA RESUMEN COMPARATIVA")

nombres_modelos = list(modelos_entrenados.keys())
modelos_lista = list(modelos_entrenados.values())

tabla_evaluacion = crear_tabla_resumen(modelos_lista, nombres_modelos, X_test, y_test)
print("\n")
print(tabla_evaluacion.to_string(index=False))

# Gráfico comparativo
graficar_comparacion(tabla_evaluacion)

# 6. SELECCIÓN DEL MEJOR MODELO
print("SELECCIÓN DEL MEJOR MODELO")

# Criterio: F1-Score (balance entre precision y recall)
mejor_modelo_nombre = tabla_evaluacion.iloc[0]['Modelo']
mejor_modelo = modelos_entrenados[mejor_modelo_nombre]

print(f"\n MODELO SELECCIONADO: {mejor_modelo_nombre}")
print(f"   F1-Score: {tabla_evaluacion.iloc[0]['F1-Score']:.4f}")
print(f"   Accuracy: {tabla_evaluacion.iloc[0]['Accuracy']:.4f}")
print(f"   Recall: {tabla_evaluacion.iloc[0]['Recall']:.4f}")

# 7. GUARDAR MEJOR MODELO
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "../../models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "mejor_modelo.pkl")
joblib.dump(mejor_modelo, model_path)

