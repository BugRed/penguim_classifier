import streamlit as st
import requests
from PIL import Image
import numpy as np
import os

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Penguim Classifier", layout="centered")

st.title("--Penguim Species Classifier--")
st.write("Insira as medidas e obtenha a predição da espécie e a certeza (probabilidades).")

# Show image
img_path = "app_frontend/assets/penguins.png"
if os.path.exists(img_path):
    st.image(Image.open(img_path), caption="Penguins")
else:
    st.info("Coloque uma imagem do penguim em 'app_frontend/assets/penguins.png' para visualizar aqui.")


# --------- Função para setar valores ---------
def set_penguin_params(flipper, body_mass, culmen_len, culmen_dep):
    st.session_state.flipper_length_mm = flipper
    st.session_state.body_mass_g = body_mass
    st.session_state.culmen_length_mm = culmen_len
    st.session_state.culmen_depth_mm = culmen_dep

# --------- Sidebar ---------
st.sidebar.header("Parâmetros dos penguins:")

# Botões de autopreenchimento
if st.sidebar.button("Opção 1: Pinguim Adelie"):
    set_penguin_params(39.1, 3750.0, 38.0, 18.0)
if st.sidebar.button("Opção 2: Pinguim Chinstrap"):
    set_penguin_params(197.0, 4150.0, 52.0, 19.0)
if st.sidebar.button("Opção 3: Pinguim Gentoo"):
    set_penguin_params(48.2, 5000.0, 50.0, 20.0)

# Campos de input com session_state para atualizar dinamicamente
flipper_length_mm = st.sidebar.number_input(
    "Comprimento da nadadeira em mm (flipper_length_mm)",
    min_value=0.0, value=float(st.session_state.get("flipper_length_mm", 39.1)),
    step=0.1, format="%.2f"
)
body_mass_g = st.sidebar.number_input(
    "Massa corporal em g (body_mass_g)",
    min_value=0.0, value=float(st.session_state.get("body_mass_g", 3750.0)),
    step=0.1, format="%.2f"
)
culmen_length_mm = st.sidebar.number_input(
    "Comprimento da crista superior do bico em mm (culmen_length_mm)",
    min_value=0.0, value=float(st.session_state.get("culmen_length_mm", 38.0)),
    step=0.1, format="%.2f"
)
culmen_depth_mm = st.sidebar.number_input(
    "Profundidade da crista superior do bico em mm (culmen_depth_mm)",
    min_value=0.0, value=float(st.session_state.get("culmen_depth_mm", 18.0)),
    step=0.1, format="%.2f"
)


if st.button("Prever"):
    payload = {
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "culmen_length_mm": culmen_length_mm,
        "culmen_depth_mm": culmen_depth_mm
    }
    try:
        with st.spinner("Consultando o modelo..."):
            res = requests.post(API_URL, json=payload, timeout=10)
        if res.status_code != 200:
            st.error(f"Erro na API: {res.status_code} - {res.text}")
        else:
            data = res.json()
            pred = data["predicted_class"]
            confidence = float(data["confidence"])
            probs = data["probabilities"]  # [Adelie, Chinstrap, Gentoo]

            st.success(f"Predição: **{pred}**")
            st.write(f"Grau de certeza: **{confidence*100:.2f}%**")

            # Mostrar barras com as probabilidades
            st.subheader("Probabilidades por classe")
            labels = ["Adelie", "Chinstrap", "Gentoo"]
            # Implementa um DataFrame para exibir
            import pandas as pd
            df = pd.DataFrame({"classe": labels, "probabilidade": probs})
            df = df.set_index("classe")
            st.bar_chart(df)

            # Opcional: mostrar probabilidades de todas as classes
            st.write("Probabilidades (raw):", {labels[i]: f"{probs[i]*100:.2f}%" for i in range(len(labels))})
    except requests.exceptions.RequestException as e:
        st.error(f"Falha ao conectar com a API: {e}")
