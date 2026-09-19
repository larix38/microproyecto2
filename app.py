import streamlit as st
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import nltk


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Clasificador ODS",
    layout="centered"
)


# =========================================================
# PREPROCESAMIENTO
# =========================================================

try:
    stopwords_es = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    stopwords_es = set(stopwords.words("spanish"))

tokenizer = RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("spanish")


def text_preprocess(text):
    tokens = tokenizer.tokenize(text.lower())

    tokens = [
        stemmer.stem(w)
        for w in tokens
        if w not in stopwords_es and len(w) > 2
    ]

    return " ".join(tokens)


# =========================================================
# CARGAR MODELO
# =========================================================

modelo = joblib.load("modelo_ods.pkl")


# =========================================================
# NOMBRES DE LOS ODS
# =========================================================

ods_names = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos"
}


# =========================================================
# INTERFAZ
# =========================================================

st.title("Clasificador de Objetivos de Desarrollo Sostenible")

st.write(
    "Ingrese un texto y el modelo determinará "
    "a qué Objetivo de Desarrollo Sostenible (ODS) se relaciona."
)

texto = st.text_area(
    "Escribe el texto que deseas clasificar:",
    height=200,
    placeholder="Ejemplo: Se desarrollarán programas para garantizar el acceso al agua potable..."
)


# =========================================================
# CLASIFICACIÓN
# =========================================================

if st.button("Clasificar texto"):

    if texto.strip() == "":
        st.warning("Por favor, escribe un texto antes de clasificar.")

    else:

        # Predicción
        prediccion = modelo.predict([texto])[0]

        # Probabilidades
        probabilidades = modelo.predict_proba([texto])[0]

        # Probabilidad máxima
        prob_max = probabilidades.max()

        # Mostrar resultado
        st.subheader(f"ODS predicho: {prediccion}")

        try:
            ods_numero = int(prediccion)

            st.write(
                f"**{ods_names.get(ods_numero, 'ODS')}**"
            )

        except:
            st.write(f"**ODS {prediccion}**")

        st.metric(
            "Probabilidad de la predicción",
            f"{prob_max:.2%}"
        )

        # =================================================
        # PROBABILIDADES
        # =================================================

        clases = modelo.classes_

        df_probabilidades = pd.DataFrame({
            "ODS": [f"ODS {c}" for c in clases],
            "Probabilidad": probabilidades
        })

        df_probabilidades["Probabilidad (%)"] = (
            df_probabilidades["Probabilidad"] * 100
        )

        df_probabilidades = df_probabilidades.sort_values(
            "Probabilidad (%)",
            ascending=False
        )

        st.subheader("Probabilidades por ODS")

        st.dataframe(
            df_probabilidades[
                ["ODS", "Probabilidad (%)"]
            ].style.format({
                "Probabilidad (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

        st.bar_chart(
            df_probabilidades.set_index("ODS")["Probabilidad (%)"]
        )
