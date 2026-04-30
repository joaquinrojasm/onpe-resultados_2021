import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# CONFIGURACIÓN
st.set_page_config(page_title="Análisis Electoral 2021", layout="wide")

st.title("📊 Sistema de Análisis Electoral - ONPE 2021")

# CARGA DE DATOS
df = pd.read_csv("data/resultados.csv", sep=";", encoding="latin-1")

df = df.fillna(0)

df["VOTOS_P1"] = pd.to_numeric(df["VOTOS_P1"], errors="coerce")
df["VOTOS_P2"] = pd.to_numeric(df["VOTOS_P2"], errors="coerce")
df["VOTOS_VB"] = pd.to_numeric(df["VOTOS_VB"], errors="coerce")
df["VOTOS_VN"] = pd.to_numeric(df["VOTOS_VN"], errors="coerce")

df["DEPARTAMENTO"] = df["DEPARTAMENTO"].str.strip()

# FILTRO SUPERIOR
st.subheader("🔎 Filtro de datos")

col_f1, col_f2, col_f3 = st.columns([1,2,1])

with col_f2:
    departamentos = sorted(df["DEPARTAMENTO"].dropna().unique())
    departamento_sel = st.selectbox(
        "Selecciona un departamento",
        ["Todos"] + departamentos
    )

# Aplicar filtro
if departamento_sel == "Todos":
    df_filtrado = df
else:
    df_filtrado = df[df["DEPARTAMENTO"] == departamento_sel]

# MÉTRICAS PRINCIPALES
st.subheader("📌 Indicadores Generales")

col1, col2, col3 = st.columns(3)

col1.metric("Mesas Totales", len(df))
col2.metric("Departamentos", df["DEPARTAMENTO"].nunique())
col3.metric("Región seleccionada", departamento_sel)

# RESULTADOS GENERALES
total_p1 = df["VOTOS_P1"].sum()
total_p2 = df["VOTOS_P2"].sum()

st.subheader("📈 Resultados Nacionales")

col1, col2 = st.columns(2)

col1.metric("Votos Candidato 1", int(total_p1))
col2.metric("Votos Candidato 2", int(total_p2))

# RESULTADOS FILTRADOS
st.subheader(f"📍 Resultados en {departamento_sel}")

votos_p1_f = df_filtrado["VOTOS_P1"].sum()
votos_p2_f = df_filtrado["VOTOS_P2"].sum()

col1, col2 = st.columns(2)

col1.metric("Candidato 1", int(votos_p1_f))
col2.metric("Candidato 2", int(votos_p2_f))

# GRÁFICOS
st.subheader("📊 Visualización de votos")

col1, col2 = st.columns(2)

with col1:
    st.write("Votos por candidato")
    st.bar_chart({
        "Candidato 1": votos_p1_f,
        "Candidato 2": votos_p2_f
    })

st.subheader("🌎 Distribución de votos por región")

votos_region = df.groupby("DEPARTAMENTO")[["VOTOS_P1", "VOTOS_P2"]].sum()

st.bar_chart(votos_region)

with col2:
    st.write("Tipos de votos")
    st.bar_chart({
        "Candidato 1": votos_p1_f,
        "Candidato 2": votos_p2_f,
        "Blanco": df_filtrado["VOTOS_VB"].sum(),
        "Nulos": df_filtrado["VOTOS_VN"].sum()
    })

# MACHINE LEARNING
st.subheader("🤖 Modelo de Machine Learning")

df["GANADOR_MESA"] = (df["VOTOS_P1"] > df["VOTOS_P2"]).astype(int)

X = df[["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN"]]
y = df["GANADOR_MESA"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.metric("Precisión del modelo", accuracy)

# PREDICCIONES
st.subheader("🔮 Predicciones del modelo")

resultados_pred = X_test.copy()
resultados_pred["REAL"] = y_test.values
resultados_pred["PREDICCION"] = y_pred

st.dataframe(resultados_pred.head(15))

# CLUSTERING
st.subheader("🧠 Agrupamiento de mesas")

kmeans = KMeans(n_clusters=3, random_state=42)
df["CLUSTER"] = kmeans.fit_predict(df[["VOTOS_P1", "VOTOS_P2"]])

col1, col2 = st.columns(2)

with col1:
    st.write("Distribución de clusters")
    st.bar_chart(df["CLUSTER"].value_counts())

with col2:
    st.write("Ejemplo de agrupación")
    st.dataframe(df[["VOTOS_P1", "VOTOS_P2", "CLUSTER"]].head(10))

# EVALUACIÓN DEL MODELO
st.subheader("📉 Evaluación del modelo")

st.write("Datos de entrenamiento:", len(X_train))
st.write("Datos de prueba:", len(X_test))