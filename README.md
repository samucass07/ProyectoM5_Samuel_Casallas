# 🏦 Sistema de Predicción de Pagos de Crédito

> Proyecto integral de Machine Learning para predecir si un cliente pagará a tiempo su préstamo, con monitoreo de drift y despliegue en producción.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---


## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Caso de Negocio](#-caso-de-negocio)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Hallazgos Principales](#-hallazgos-principales)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Roadmap](#-roadmap)
- [Contribuciones](#-contribuciones)
- [Autor](#-autor)

---

## 🎯 Descripción del Proyecto

Sistema end-to-end de Machine Learning que predice la probabilidad de que un cliente pague a tiempo su crédito. El proyecto incluye:

- ✅ Análisis Exploratorio de Datos (EDA)
- ✅ Ingeniería de Características
- ✅ Entrenamiento y evaluación de múltiples modelos
- ✅ Sistema de monitoreo de Data Drift
- ✅ API REST para predicciones en producción
- ✅ Despliegue con Docker

---

## 💼 Caso de Negocio

### Problema

Las instituciones financieras enfrentan pérdidas significativas por créditos no pagados. La identificación temprana de clientes con alto riesgo de impago permite:

- 🔍 **Reducir morosidad** 
- 💰 **Optimizar capital** asignando recursos a clientes confiables
- ⚡ **Acelerar decisiones** de aprobación de créditos
- 📊 **Mejorar rentabilidad** del portafolio de préstamos

### Solución

Sistema predictivo basado en ML que:

1. Analiza 22 variables del cliente (demográficas, financieras, comportamiento crediticio)
2. Genera predicción en tiempo real 
3. Proporciona probabilidad de pago y nivel de confianza
4. Monitorea continuamente la calidad del modelo
5. Despliega en producción con Docker

## 🏗️ Arquitectura del Sistema
```
┌─────────────────┐
│  Datos Crudos   │
│  (Base Excel)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   ETAPA 1: ANÁLISIS Y PREPARACIÓN   │
├─────────────────────────────────────┤
│ • Carga de datos                    │
│ • EDA completo                      │
│ • Limpieza y transformaciones       │
│ • Feature Engineering               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   ETAPA 2: MODELADO Y EVALUACIÓN    │
├─────────────────────────────────────┤
│ • Split temporal por fechas         │
│ • Pipelines de preprocesamiento     │
│ • Entrenamiento de modelos:         │
│   - Logistic Regression             │
│   - Random Forest                   │
│   - Gradient Boosting               │
│ • Selección del mejor modelo        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   ETAPA 3: MONITOREO                │
├─────────────────────────────────────┤
│ • Detección de Data Drift           │
│   - PSI, KS, JS, Chi²               │
│ • Dashboard Streamlit               │
│ • Alertas automáticas               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   ETAPA 4: DESPLIEGUE               │
├─────────────────────────────────────┤
│ • API REST (FastAPI)                │
│ • Predicción individual y batch     │
│ • Contenedor (Docker)               │
│ • Logging de predicciones           │
└─────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.11+
- pip o conda
- (Opcional) Docker

### Opción 1: Instalación Local
```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/ProyectoM5_Samuel_Casallas.git
cd ProyectoM5_Samuel_Casallas

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python -c "import sklearn, pandas, fastapi; print('✅ Todo instalado')"
```

### Opción 2: Con Docker
```bash
# 1. Construir imagen
docker build -t credit-api:v1.0 .

# 2. Ejecutar contenedor
docker run -d -p 8000:8000 --name credit-api credit-api:v1.0

# 3. Verificar
curl http://localhost:8000/health
```

---

## 💻 Uso

### 1️⃣ Entrenar Modelo
```bash
# Ejecutar notebooks en orden:
jupyter notebook notebooks/1_Cargar_datos.ipynb
jupyter notebook notebooks/2_Comprension_eda.ipynb

# Entrenar modelos
python src/ft_engineering.py
python src/model_training_evaluation.py
```

### 2️⃣ Iniciar API
```bash
# Desarrollo
python src/model_deploy.py

# Producción con Docker
docker-compose up -d
```

### 3️⃣ Hacer Predicciones

**Ejemplo con curl:**
```bash
  -H "Content-Type: application/json" \
  -d '{
    "capital_prestado": 5000000,
    "plazo_meses": 24,
    "edad_cliente": 35,
    "salario_cliente": 3000000,
    "total_otros_prestamos": 0,
    "cuota_pactada": 250000,
    "puntaje_datacredito": 700,
    "cant_creditosvigentes": 1,
    "huella_consulta": 2,
    "saldo_mora": 0,
    "saldo_total": 500000,
    "saldo_principal": 500000,
    "saldo_mora_codeudor": 0,
    "creditos_sectorFinanciero": 1,
    "creditos_sectorCooperativo": 0,
    "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 3000000,
    "tipo_credito": 1,
    "tipo_laboral": "Empleado",
    "tendencia_ingresos": "Estable",
    "fecha_prestamo": "2022-01-01"
  }'
```

**Respuesta:**
```json
{
  "pago_a_tiempo": "SÍ",
  "probabilidad": 0.8542,
  "confianza": "Alta",
  "timestamp": "2026-02-18T14:30:00"
}
```

### 4️⃣ Monitoreo de Drift
```bash
# Iniciar dashboard de monitoreo
streamlit run src/model_monitoring.py
```

---

### Desempeño de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.952 | 0.952 | 1.00 | 0.975 |
| Random Forest | 0.952 | 0.952 | 1.00 | 0.975 |
| Gradient Boosting | 0.9535 | 0.9535 | 1.00 | 0.976 |

**🏆 Modelo Seleccionado:** Gradient Boosting

**Justificación:**
- ✅ Mejor F1-Score (balance precision-recall)
- ✅ Robusto ante outliers
- ✅ Interpretabilidad mediante feature importance
- ✅ Buen desempeño en validación cruzada

### Variables Derivadas Creadas
```python
# Feature Engineering aplicado:
1. deuda_total_pendiente = saldo_principal + saldo_mora
2. pct_capital_pendiente = deuda_total / capital_prestado
```

Estas variables mejoraron el F1-Score en el modelo.

---

## 🛠️ Tecnologías Utilizadas

### Core ML Stack

- **Python 3.11** - Lenguaje base
- **scikit-learn 1.5** - Modelado ML
- **pandas 2.2** - Manipulación de datos
- **numpy 1.26** - Operaciones numéricas

### Análisis y Visualización

- **Jupyter** - Notebooks interactivos
- **matplotlib / seaborn** - Visualizaciones
- **Streamlit** - Dashboard de monitoreo

### Monitoreo

- **Evidently** - Reportes de drift
- **scipy** - Tests estadísticos (KS, Chi²)

### Deploy

- **FastAPI** - API REST
- **Pydantic** - Validación de datos
- **Uvicorn** - Servidor ASGI
- **Docker** - Contenedorización

### DevOps

- **Git** - Control de versiones
- **joblib** - Serialización de modelos

---

<div align="center">

</div>