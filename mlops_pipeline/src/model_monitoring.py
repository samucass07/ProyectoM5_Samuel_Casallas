import pandas as pd
import numpy as np
import os
import joblib
import evidently
from datetime import datetime
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from ft_engineering import cargar_datos, crear_features
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

API_URL = "http://localhost:5000"
DATASET_PATH = "../../Base_de_datos.xlsx"  # Dataset original
MONITORING_LOG = "../df_post_eda.csv"  # Dataset editado/procesado

# ============================================================
# 2. CARGA DE DATOS CON CACHE
# ============================================================

@st.cache_data
def load_data():
    # Cargar y procesar
    df = cargar_datos()
    df = crear_features(df)
    
    # Ordenar por fecha
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
        df = df.sort_values('fecha_prestamo').reset_index(drop=True)
    
    split_point=int(len(df)*0.7)
    df_ref=df.iloc[:split_point]
    df_new=df.iloc[split_point:]
    
    return df_ref, df_new
# ============================================================
# 3. API para predicciones
# ============================================================
def get_predictions(X):
    payload={"batch":X.values.tolist()}
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Error al obtener predicciones: {e}")
        return pd.DataFrame()
# ============================================================
# 4. Guardar logs
# ============================================================
def log_predictions(X,preds):
    log_df = X.copy()
    log_df['prediccion'] = preds
    log_df['timestamp'] = datetime.now()
    
    if os.path.exists(MONITORING_LOG):
        log_df.to_csv(MONITORING_LOG, mode='a', header=False, index=False)
    else:
        log_df.to_csv(MONITORING_LOG, index=False)

# ============================================================
# 5. Reporte Evidently
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
reportesdir = os.path.join(current_dir, "../../reportes")
os.makedirs(reportesdir, exist_ok=True)
def generate_evidently_report(reference, current):
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        myreport=report.run(reference_data=reference, current_data=current)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        reporte_path = os.path.join(reportesdir, f"reporte_{timestamp}.html")
        
        myreport.save_html(reporte_path)
        return reporte_path
    except ModuleNotFoundError:
        st.error("Evidently no está instalado. Instala con: pip install evidently")
        return None
    except Exception as e:
        st.error(f"Error generando reporte: {e}")
        return None
# ============================================================
# 5. FUNCIONES DE DRIFT
# ============================================================

def calculate_psi(reference, current, bins=10):
    """Population Stability Index"""
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 2:
        return 0
    
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)
    
    ref_props = (ref_counts + 1e-6) / (len(reference) + bins * 1e-6)
    curr_props = (curr_counts + 1e-6) / (len(current) + bins * 1e-6)
    
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return psi

def calculate_ks_statistic(reference, current):
    """Kolmogorov-Smirnov"""
    statistic, p_value = stats.ks_2samp(reference, current)
    return statistic, p_value

def calculate_js_divergence(reference, current, bins=10):
    """Jensen-Shannon divergence"""
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)
    
    ref_hist, _ = np.histogram(reference, bins=breakpoints, density=True)
    curr_hist, _ = np.histogram(current, bins=breakpoints, density=True)
    
    ref_hist = (ref_hist + 1e-10) / (ref_hist.sum() + bins * 1e-10)
    curr_hist = (curr_hist + 1e-10) / (curr_hist.sum() + bins * 1e-10)
    
    return jensenshannon(ref_hist, curr_hist)

def calculate_chi_square(reference, current):
    """Chi-cuadrado para categóricas"""
    categories = sorted(set(reference) | set(current))
    
    ref_counts = pd.Series(reference).value_counts().reindex(categories, fill_value=0)
    curr_counts = pd.Series(current).value_counts().reindex(categories, fill_value=0)
    
    contingency_table = np.array([ref_counts, curr_counts])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    return chi2, p_value

def analyze_drift(reference_df, current_df):
    
    # Detectar columnas numéricas y categóricas automáticamente
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = reference_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Quitar columnas de fecha si existen
    date_cols = ['fecha_prestamo', 'timestamp']
    numeric_cols = [col for col in numeric_cols if col not in date_cols]
    categorical_cols = [col for col in categorical_cols if col not in date_cols]
    
    results = []
    
    # Análisis numéricas
    for col in numeric_cols:
        if col not in current_df.columns:
            continue
        
        ref_col = reference_df[col].dropna()
        curr_col = current_df[col].dropna()
        
        if len(ref_col) == 0 or len(curr_col) == 0:
            continue
        
        psi = calculate_psi(ref_col, curr_col)
        ks_stat, ks_p = calculate_ks_statistic(ref_col, curr_col)
        js_div = calculate_js_divergence(ref_col, curr_col)
        
        # Determinar severidad
        if psi >= 0.2 or ks_stat > 0.2 or js_div > 0.2:
            severity = 'HIGH'
            drift = True
        elif psi >= 0.1 or ks_stat > 0.1 or js_div > 0.1:
            severity = 'MEDIUM'
            drift = True
        else:
            severity = 'LOW'
            drift = False
        
        results.append({
            'variable': col,
            'type': 'numeric',
            'PSI': round(psi, 4),
            'KS_statistic': round(ks_stat, 4),
            'KS_p_value': round(ks_p, 4),
            'JS_divergence': round(js_div, 4),
            'drift_detected': drift,
            'severity': severity
        })
    
    # Análisis categóricas
    for col in categorical_cols:
        if col not in current_df.columns:
            continue
        
        ref_col = reference_df[col].dropna()
        curr_col = current_df[col].dropna()
        
        if len(ref_col) == 0 or len(curr_col) == 0:
            continue
        
        chi2, p_value = calculate_chi_square(ref_col, curr_col)
        
        drift = p_value < 0.05
        severity = 'HIGH' if chi2 > 10 else 'MEDIUM' if chi2 > 5 else 'LOW'
        
        results.append({
            'variable': col,
            'type': 'categorical',
            'PSI': None,
            'KS_statistic': None,
            'KS_p_value': None,
            'JS_divergence': None,
            'Chi2': round(chi2, 4),
            'Chi2_p_value': round(p_value, 4),
            'drift_detected': drift,
            'severity': severity
        })
    
    return pd.DataFrame(results)
    
# ============================================================
# 6. STREAMLIT UI (SIMPLE)
# ============================================================

def main():
    st.set_page_config(page_title="Monitoreo ML", page_icon="📊", layout="wide")
    st.title("📊 Sistema de Monitoreo de Data Drift")
    
    # --- INICIALIZACIÓN DE VARIABLES ---
    if "df_reference" not in st.session_state:
        df_ref, df_new = load_data()
        st.session_state.df_reference = df_ref
        st.session_state.df_current = df_new

    # Estas variables son las que usaremos en todo el script
    # Quitamos la columna 'target' o 'fecha' si no quieres que entren en el análisis de drift
    X_ref = st.session_state.df_reference
    X_new = st.session_state.df_current

    # Sidebar
    st.sidebar.header("Estado de los Datos")
    st.sidebar.write(f"📂 **Referencia:** {X_ref.shape[0]} filas")
    st.sidebar.write(f"🆕 **Actual:** {X_new.shape[0]} filas")

    tab1, tab2, tab3 = st.tabs(["🔍 Análisis Drift", "📊 Visualizaciones", "📄 Reporte Evidently"])
    
    # ========== TAB 1: ANÁLISIS DRIFT ==========
    with tab1:
        st.header("Análisis de Data Drift")
        
        if st.button("🔍 Analizar Drift", type="primary"):
            with st.spinner("Analizando..."):
                # AHORA SÍ: X_ref y X_new están definidas arriba
                drift_results = analyze_drift(X_ref, X_new)
                st.session_state['drift_results'] = drift_results
                
                col1, col2, col3 = st.columns(3)
                total = len(drift_results)
                con_drift = drift_results['drift_detected'].sum()
                high = (drift_results['severity'] == 'HIGH').sum()
                
                col1.metric("Total Variables", total)
                col2.metric("Con Drift", con_drift, f"{con_drift/total*100:.1f}%")
                col3.metric("Severidad Alta", high)

        if 'drift_results' in st.session_state:
            df_show = st.session_state['drift_results']
            st.dataframe(df_show, use_container_width=True)

    # ========== TAB 2: VISUALIZACIONES ==========
    with tab2:
        st.header("Visualizaciones")
        if 'drift_results' not in st.session_state:
            st.info("👈 Primero analiza drift en la pestaña anterior")
        else:
            numeric_vars = X_ref.select_dtypes(include=[np.number]).columns.tolist()
            var = st.selectbox("Selecciona variable para comparar:", numeric_vars)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.kdeplot(X_ref[var], label="Referencia", fill=True, ax=ax)
            sns.kdeplot(X_new[var], label="Nuevo", fill=True, ax=ax)
            ax.set_title(f"Distribución de {var}")
            ax.legend()
            st.pyplot(fig)

    # ========== TAB 3: EVIDENTLY ==========
    with tab3:
        st.header("Reporte Evidently")
        if st.button("📄 Generar Reporte Completo"):
            report = generate_evidently_report(X_ref, X_new)
            if report:
                st.success("Reporte generado exitosamente. Revisa el archivo HTML en el directorio.")
                try:
                    with open(report, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html( f"""
                        <div style="width: 200%; overflow-x: auto;">
                            {html_content}
                        </div>
                        """, height=1100, scrolling=True)
                except Exception as e:
                    st.error(f"Error al mostrar reporte: {e}")

if __name__ == "__main__":
    main()