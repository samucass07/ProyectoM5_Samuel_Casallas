import pandas as pd
import numpy as np  
import os
import sys
import joblib
import uvicorn
from fastapi import FastAPI
from ft_engineering import cargar_datos,crear_features,crear_preprocessor
from pydantic import BaseModel

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime # Necesario para el timestamp

# --- AJUSTE DE RUTAS: DEBE IR ANTES DE LOS IMPORTS DE TUS SCRIPTS ---
# Buscamos la carpeta 'src' para que encuentre ft_engineering
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Ahora sí podemos importar tus funciones
from ft_engineering import cargar_datos, crear_features

app= FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="API para predecir si un cliente pagará a tiempo su préstamo",
    version="1.0.0"
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mejor_modelo.pkl")

# 2. Cargar usando JOBLIB (Igual a como lo guardaste)
try:
    # IMPORTANTE: Usamos joblib.load, no pickle.load
    model_rf = joblib.load(MODEL_PATH)
    print(f"✅ ¡ÉXITO! Modelo cargado correctamente con Joblib")
    print(f"Ruta: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model_rf = None
    
class ClienteInput(BaseModel):
    """Esquema de datos de entrada para un cliente"""
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tipo_credito: int
    tipo_laboral: str
    tendencia_ingresos: str
    fecha_prestamo: str ="%Y-%m-%d" 

class PredictionOutput(BaseModel):
    """Esquema de salida de predicción"""
    pago_a_tiempo: str
    probabilidad: float
    confianza: str
    timestamp: str    
    
    
#endpint 
@app.get("/saludo")
def saludar():
    return {"mensaje": "¡Hola desde la API de Predicción de Pago a Tiempo!"}

from datetime import datetime
@app.post("/predecir", response_model=PredictionOutput)
def predecir_pago(cliente: ClienteInput):

    if model_rf is None:
        return {"error": "Modelo no disponible"}
    
    try:
        # 1. Convertir entrada a DataFrame
        df_input = pd.DataFrame([cliente.model_dump()])
        df_input['fecha_prestamo'] = pd.to_datetime(df_input['fecha_prestamo'])
        
        if "Pago_atiempo" not in df_input.columns:
            df_input['Pago_atiempo'] = 0 
            
        df_procesado = crear_features(df_input)
        
        if 'Pago_atiempo' in df_procesado.columns:
            df_procesado = df_procesado.drop(columns=['Pago_atiempo'])
            
        prediccion = model_rf.predict(df_procesado)[0]
        probabilidades = model_rf.predict_proba(df_procesado)[0]
        prob_pago = float(probabilidades[1]) 
        
        if prob_pago >= 0.8 or prob_pago <= 0.2:
            confianza = "Alta"
        elif prob_pago >= 0.6 or prob_pago <= 0.4:
            confianza = "Media"
        else:
            confianza = "Baja"
        
        return PredictionOutput(
            pago_a_tiempo="SÍ" if prediccion == 1 else "NO",
            probabilidad=round(prob_pago, 4),
            confianza=confianza,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        return {"error": f"Error al procesar la predicción: {e}"}

#cargar script
if __name__ == "__main__":
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)