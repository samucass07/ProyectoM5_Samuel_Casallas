import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. CARGA DEL DATO LIMPIO
def cargar_datos():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "../df_post_eda.parquet")

    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"Datos cargados desde EDA")
        return df
    else:
        raise FileNotFoundError(f"Archivo no encontrado en {path}")
# 2. INGENIERÍA DE CARACTERÍSTICAS
def crear_features(df_input):
    df_fe = df_input.copy()
    
    df_fe['mes_prestamo'] = df_fe['fecha_prestamo'].dt.month
    df_fe['anio_prestamo'] = df_fe['fecha_prestamo'].dt.year

    df_fe['deuda_total_pendiente'] = df_fe['saldo_principal'] + df_fe['saldo_mora']
    
    df_fe['pct_capital_pendiente'] = (
        df_fe['deuda_total_pendiente'] / (df_fe['capital_prestado'] + 1)
    )
    df_fe['pct_capital_pendiente'] = df_fe['pct_capital_pendiente'].replace(
        [np.inf, -np.inf], 0
    )
    df_fe['Pago_atiempo'] = df_fe['Pago_atiempo'].astype(int)
    print("Features creados")
    return df_fe


# --- PROCESO DE PIPELINES ---
cols_num = [
    'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente', 
    'total_otros_prestamos', 'cuota_pactada', 'puntaje_datacredito', 
    'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total', 
    'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero', 
    'creditos_sectorCooperativo', 'creditos_sectorReal', 
    'promedio_ingresos_datacredito', 
    'deuda_total_pendiente', 
    'pct_capital_pendiente'   
]
cols_cat = ['tipo_credito', 'tipo_laboral', 'tendencia_ingresos']

#puntaje se deja afuera por una decision estadistica de coherencia

def crear_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), cols_num), 
            
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cols_cat)
        ],
        remainder='drop' 
    )
    return preprocessor

def preparar_datos(test_size=0.2, random_state=42):
    
    df = cargar_datos()
    df_fe = crear_features(df)
    preprocessor = crear_preprocessor()

    X = df_fe.drop(columns=['Pago_atiempo'])
    y = df_fe['Pago_atiempo']




    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,stratify=y)
    return X_train, X_test, y_train, y_test,preprocessor

