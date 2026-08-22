import pickle
import pandas as pd
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="Clasificador de Pingüinos", page_icon="🐧", layout="centered"
)

st.title("Clasificador de Especies de Pingüinos 🐧")
st.write(
    "Ingrese las características morfológicas para predecir la especie correspondiente."
)


# Cargar artefactos con caché para optimizar el rendimiento
@st.cache_resource()
def cargar_modelo():
    with open("random_forest_penguin.pickle", "rb") as rf_file:
        pipeline = pickle.load(rf_file)
    with open("output_penguin.pickle", "rb") as out_file:
        target_names = pickle.load(out_file)
    return pipeline, target_names


rf_pipeline, especies_unicas = cargar_modelo()

# Formulario de entrada
col1, col2 = st.columns(2)

with col1:
    isla = st.selectbox("Isla de origen", ["Biscoe", "Dream", "Torgerson"])
    sexo = st.selectbox("Sexo", ["Male", "Female"])
    longitud_pico = st.number_input(
        "Longitud del pico (mm)", value=40.0, step=0.1
    )

with col2:
    profundidad_pico = st.number_input(
        "Profundidad del pico (mm)", value=18.0, step=0.1
    )
    longitud_aleta = st.number_input(
        "Longitud de la aleta (mm)", value=200.0, step=1.0
    )
    masa_corporal = st.number_input(
        "Masa corporal (g)", value=4000.0, step=50.0
    )

# Botón para ejecutar la predicción
if st.button("Predecir Especie", type="primary"):
    # Preservar estructura DataFrame para evitar advertencias de Scikit-Learn
    datos_entrada_df = pd.DataFrame(
        [
            {
                "island": isla,
                "bill_length_mm": longitud_pico,
                "bill_depth_mm": profundidad_pico,
                "flipper_length_mm": longitud_aleta,
                "body_mass_g": masa_corporal,
                "sex": sexo,
            }
        ]
    )

    prediccion_idx = rf_pipeline.predict(datos_entrada_df)[0]
    especie_predicha = especies_unicas[prediccion_idx]

    st.success(f"**Especie predicha:** {especie_predicha}")
