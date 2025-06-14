# app.py (Versión 2.0 - Bilingüe)

import streamlit as st
import pandas as pd
import joblib

# --- 1. DICCIONARIO DE TEXTOS ---
TEXTS = {
    'es': {
        'page_title': "Predicción de Precios de Autos", 'page_icon': "🚗",
        'title': "Predicción de Precios de Automóviles Usados 🚗",
        'description': "Esta app utiliza un modelo de Machine Learning para estimar el precio de un auto usado. Mueve los controles y selecciona las opciones para ver la predicción.",
        'sidebar_header': "Características del Auto", 'car_age_label': "Antigüedad del Auto (años)",
        'km_label': "Kilometraje", 'power_label': "Potencia Máxima (bhp)", 'engine_label': "Cilindrada (cc)",
        'fuel_label': "Tipo de Combustible", 'transmission_label': "Transmisión",
        'seller_label': "Tipo de Vendedor", 'owner_label': "Dueños Anteriores",
        'button_predict': "Predecir Precio", 'subheader_prediction': "Predicción de Precio",
        'success_price': "El precio estimado del auto es:",
        'note': "**Nota:** Esta es una estimación basada en un modelo entrenado con datos de la India."
    },
    'en': {
        'page_title': "Car Price Prediction", 'page_icon': "🚗",
        'title': "Used Car Price Prediction 🚗",
        'description': "This app uses a Machine Learning model to estimate the price of a used car. Move the sliders and select the options to see the prediction.",
        'sidebar_header': "Car Features", 'car_age_label': "Car Age (years)",
        'km_label': "Kilometers", 'power_label': "Max Power (bhp)", 'engine_label': "Engine (cc)",
        'fuel_label': "Fuel Type", 'transmission_label': "Transmission",
        'seller_label': "Seller Type", 'owner_label': "Previous Owners",
        'button_predict': "Predict Price", 'subheader_prediction': "Price Prediction",
        'success_price': "The estimated price for the car is:",
        'note': "**Note:** This is an estimate based on a model trained with data from India."
    }
}
# --- DICCIONARIOS DE MAPEO (ES -> EN) ---
MAPPINGS = {
    'es': {
        'Fuel Type': {'Diésel': 'Diesel', 'Gasolina': 'Petrol', 'GLP': 'LPG', 'GNC': 'CNG'},
        'Transmission': {'Manual': 'Manual', 'Automática': 'Automatic'},
        'Seller Type': {'Individual': 'Individual', 'Concesionario': 'Dealer'},
        'Owner': {'Primero': 'First', 'Segundo': 'Second', 'Tercero': 'Third'}
    },
    'en': {
        'Fuel Type': {'Diesel': 'Diesel', 'Petrol': 'Petrol', 'LPG': 'LPG', 'CNG': 'CNG'},
        'Transmission': {'Manual': 'Manual', 'Automatic': 'Automatic'},
        'Seller Type': {'Individual': 'Individual', 'Dealer': 'Dealer'},
        'Owner': {'First': 'First', 'Second': 'Second', 'Third': 'Third'}
    }
}

# --- MANEJO DEL ESTADO DEL IDIOMA ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
def toggle_language():
    st.session_state.lang = 'es' if st.session_state.lang == 'en' else 'en'
texts = TEXTS[st.session_state.lang]
maps = MAPPINGS[st.session_state.lang]

# --- Carga del modelo ---
@st.cache_resource
def load_model():
    model = joblib.load('modelo_autos_rf.pkl')
    model_columns = joblib.load('columnas_modelo.pkl')
    return model, model_columns
model, model_columns = load_model()

# --- INTERFAZ ---
st.set_page_config(page_title=texts['page_title'], page_icon=texts['page_icon'], layout="wide")
st.button('Español / English', on_click=toggle_language)
st.title(texts['title'])
st.markdown(texts['description'])

st.sidebar.header(texts['sidebar_header'])

def user_input_features():
    car_age = st.sidebar.slider(texts['car_age_label'], 1, 30, 8)
    kilometer = st.sidebar.slider(texts['km_label'], 1000, 500000, 60000)
    max_power = st.sidebar.slider(texts['power_label'], 50.0, 300.0, 85.0)
    engine = st.sidebar.slider(texts['engine_label'], 600, 5000, 1200)

    # Usamos las keys del diccionario de mapeo para las opciones visibles
    fuel_type_disp = st.sidebar.selectbox(texts['fuel_label'], list(maps['Fuel Type'].keys()))
    transmission_disp = st.sidebar.selectbox(texts['transmission_label'], list(maps['Transmission'].keys()))
    seller_type_disp = st.sidebar.selectbox(texts['seller_label'], list(maps['Seller Type'].keys()))
    owner_disp = st.sidebar.selectbox(texts['owner_label'], list(maps['Owner'].keys()))

    # Mapeamos la selección del usuario de vuelta al valor en inglés que el modelo entiende
    fuel_type = maps['Fuel Type'][fuel_type_disp]
    transmission = maps['Transmission'][transmission_disp]
    seller_type = maps['Seller Type'][seller_type_disp]
    owner = maps['Owner'][owner_disp]

    data = {
        'Kilometer': kilometer, 'Engine': engine, 'Max Power': max_power, 'Car_Age': car_age,
        'Length': 4280, 'Width': 1767, 'Height': 1591, 'Seating Capacity': 5.0, 'Fuel Tank Capacity': 52.0,
        'Fuel Type_Diesel': 1 if fuel_type == 'Diesel' else 0, 'Fuel Type_LPG': 1 if fuel_type == 'LPG' else 0,
        'Fuel Type_Petrol': 1 if fuel_type == 'Petrol' else 0, 'Transmission_Manual': 1 if transmission == 'Manual' else 0,
        'Owner_Second': 1 if owner == 'Second' else 0, 'Owner_Third': 1 if owner == 'Third' else 0,
        'Owner_Unregistered Car': 0, 'Seller Type_Individual': 1 if seller_type == 'Individual' else 0, 'Seller Type_Corporate': 0
    }
    features = pd.DataFrame(data, index=[0]).reindex(columns=model_columns, fill_value=0)
    return features

input_df = user_input_features()

st.subheader(texts['subheader_prediction'])
prediction = model.predict(input_df)
st.success(f"{texts['success_price']} **${prediction[0]:,.2f}**")
st.write('---')
st.write(texts['note'])