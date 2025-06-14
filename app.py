# app.py (Versión 3.0 - Bilingüe y con Conversor de Moneda)

import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import date

# --- 1. DICCIONARIO DE TEXTOS ---
TEXTS = {
    'es': {
        'page_title': "Predictor de Precios de Autos (India)", 'page_icon': "🚗",
        'title': "Predictor de Precios de Autos Usados (Mercado de India) 🚗",
        'description': "Esta app estima el precio de un auto usado utilizando un modelo de Machine Learning. Los resultados se muestran en Rupias Indias (₹) y su equivalente en Dólares Americanos ($).",
        'sidebar_header': "Características del Auto", 'car_age_label': "Antigüedad (años)",
        'km_label': "Kilometraje", 'power_label': "Potencia (bhp)", 'engine_label': "Cilindrada (cc)",
        'fuel_label': "Combustible", 'transmission_label': "Transmisión",
        'seller_label': "Vendedor", 'owner_label': "Dueños Previos",
        'button_predict': "Predecir Precio", 'subheader_prediction': "Estimación de Precio",
        'inr_label': "Precio en Rupias Indias", 'usd_label': "Precio en Dólares Americanos",
        'rate_info': "Tipo de cambio actual"
    },
    'en': {
        'page_title': "Car Price Predictor (India)", 'page_icon': "🚗",
        'title': "Used Car Price Predictor (Indian Market) 🚗",
        'description': "This app estimates a used car's price using a Machine Learning model. Results are shown in Indian Rupees (₹) and the equivalent in US Dollars ($).",
        'sidebar_header': "Car Features", 'car_age_label': "Car Age (years)",
        'km_label': "Kilometers", 'power_label': "Max Power (bhp)", 'engine_label': "Engine (cc)",
        'fuel_label': "Fuel Type", 'transmission_label': "Transmission",
        'seller_label': "Seller Type", 'owner_label': "Previous Owners",
        'button_predict': "Predict Price", 'subheader_prediction': "Price Estimation",
        'inr_label': "Price in Indian Rupees", 'usd_label': "Price in US Dollars",
        'rate_info': "Current exchange rate"
    }
}
# (El diccionario MAPPINGS no cambia)
MAPPINGS = { 'es': { 'Fuel Type': {'Diésel': 'Diesel', 'Gasolina': 'Petrol'}, 'Transmission': {'Manual': 'Manual', 'Automática': 'Automatic'}, 'Seller Type': {'Individual': 'Individual', 'Concesionario': 'Dealer'}, 'Owner': {'Primero': 'First', 'Segundo': 'Second', 'Tercero': 'Third'} }, 'en': { 'Fuel Type': {'Diesel': 'Diesel', 'Petrol': 'Petrol'}, 'Transmission': {'Manual': 'Manual', 'Automatic': 'Automatic'}, 'Seller Type': {'Individual': 'Individual', 'Dealer': 'Dealer'}, 'Owner': {'First': 'First', 'Second': 'Second', 'Third': 'Third'} } }

# --- 2. LÓGICA DE LA APP ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
def toggle_language():
    st.session_state.lang = 'es' if st.session_state.lang == 'en' else 'en'
texts = TEXTS[st.session_state.lang]
maps = MAPPINGS[st.session_state.lang]

@st.cache_resource
def load_model():
    model = joblib.load('modelo_autos_rf.pkl')
    model_columns = joblib.load('columnas_modelo.pkl')
    return model, model_columns

# Usamos st.cache_data para no llamar a la API en cada re-ejecución
@st.cache_data(ttl=3600) # Cache por 1 hora
def get_inr_to_usd_rate():
    try:
        # Usamos el ticker de yfinance para la conversión INR a USD
        rate = yf.Ticker("INRUSD=X").info['previousClose']
        return rate
    except Exception:
        # Valor de respaldo por si la API falla
        return 0.012 

model, model_columns = load_model()
inr_to_usd_rate = get_inr_to_usd_rate()

# --- INTERFAZ ---
st.set_page_config(page_title=texts['page_title'], page_icon=texts['page_icon'], layout="wide")
st.button('Español / English', on_click=toggle_language)
st.title(texts['title'])
st.markdown(texts['description'])

st.sidebar.header(texts['sidebar_header'])

def user_input_features():
    # (El código de los widgets de la sidebar no cambia)
    car_age = st.sidebar.slider(texts['car_age_label'], 1, 30, 8); kilometer = st.sidebar.slider(texts['km_label'], 1000, 500000, 60000); max_power = st.sidebar.slider(texts['power_label'], 50.0, 300.0, 85.0); engine = st.sidebar.slider(texts['engine_label'], 600, 5000, 1200); fuel_type_disp = st.sidebar.selectbox(texts['fuel_label'], list(maps['Fuel Type'].keys())); transmission_disp = st.sidebar.selectbox(texts['transmission_label'], list(maps['Transmission'].keys())); seller_type_disp = st.sidebar.selectbox(texts['seller_label'], list(maps['Seller Type'].keys())); owner_disp = st.sidebar.selectbox(texts['owner_label'], list(maps['Owner'].keys()))
    fuel_type = maps['Fuel Type'][fuel_type_disp]; transmission = maps['Transmission'][transmission_disp]; seller_type = maps['Seller Type'][seller_type_disp]; owner = maps['Owner'][owner_disp]
    data = { 'Kilometer': kilometer, 'Engine': engine, 'Max Power': max_power, 'Car_Age': car_age, 'Length': 4280, 'Width': 1767, 'Height': 1591, 'Seating Capacity': 5.0, 'Fuel Tank Capacity': 52.0, 'Fuel Type_Diesel': 1 if fuel_type == 'Diesel' else 0, 'Fuel Type_LPG': 0, 'Fuel Type_Petrol': 1 if fuel_type == 'Petrol' else 0, 'Transmission_Manual': 1 if transmission == 'Manual' else 0, 'Owner_Second': 1 if owner == 'Second' else 0, 'Owner_Third': 1 if owner == 'Third' else 0, 'Owner_Unregistered Car': 0, 'Seller Type_Individual': 1 if seller_type == 'Individual' else 0, 'Seller Type_Corporate': 0 }
    return pd.DataFrame(data, index=[0]).reindex(columns=model_columns, fill_value=0)

input_df = user_input_features()
prediction_inr = model.predict(input_df)[0]
prediction_usd = prediction_inr * inr_to_usd_rate

st.subheader(texts['subheader_prediction'])
col1, col2 = st.columns(2)
col1.metric(label=texts['inr_label'], value=f"₹ {prediction_inr:,.2f}")
col2.metric(label=texts['usd_label'], value=f"$ {prediction_usd:,.2f}")
st.caption(f"{texts['rate_info']}: 1 INR ≈ ${inr_to_usd_rate:.4f} USD")