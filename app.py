# app.py

import streamlit as st
import pandas as pd
import joblib

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicción de Precios de Autos",
    page_icon="🚗",
    layout="wide"
)


# --- FUNCIONES ---
@st.cache_resource
def load_model():
    """Carga el modelo y las columnas desde los archivos .pkl"""
    model = joblib.load('modelo_autos_rf.pkl')
    model_columns = joblib.load('columnas_modelo.pkl')
    return model, model_columns

# --- CARGA DE DATOS ---
try:
    model, model_columns = load_model()
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo ('modelo_autos_rf.pkl', 'columnas_modelo.pkl'). Asegúrate de que estén en el mismo directorio que app.py.")
    st.stop()


# --- INTERFAZ DE USUARIO ---
st.title('Predicción de Precios de Automóviles Usados 🚗')
st.write('Esta app utiliza un modelo de Machine Learning (Random Forest) para estimar el precio de un auto usado basado en sus características. ¡Mueve los controles para ver la predicción!')

# Usamos la barra lateral para los controles de entrada
st.sidebar.header('Introduce las Características del Auto:')

# --- WIDGETS DE ENTRADA ---
def user_input_features():
    car_age = st.sidebar.slider('Antigüedad del Auto (años)', 1, 30, 8)
    kilometer = st.sidebar.slider('Kilometraje', 1000, 500000, 60000)
    max_power = st.sidebar.slider('Potencia Máxima (bhp)', 50.0, 300.0, 85.0)
    engine = st.sidebar.slider('Cilindrada del Motor (cc)', 600, 5000, 1200)

    fuel_type = st.sidebar.selectbox('Tipo de Combustible', ('Petrol', 'Diesel', 'LPG', 'CNG'))
    transmission = st.sidebar.selectbox('Transmisión', ('Manual', 'Automatic'))
    seller_type = st.sidebar.selectbox('Tipo de Vendedor', ('Individual', 'Dealer')) # Simplificado
    owner = st.sidebar.selectbox('Dueños Anteriores', ('First', 'Second', 'Third'))

    # Convertimos la entrada a un DataFrame de pandas
    data = {
        'Kilometer': kilometer,
        'Engine': engine,
        'Max Power': max_power,
        'Length': 4280, # Usamos valores promedio/comunes para las características que no pedimos
        'Width': 1767,
        'Height': 1591,
        'Seating Capacity': 5.0,
        'Fuel Tank Capacity': 52.0,
        'Car_Age': car_age,
        'Fuel Type_Diesel': 1 if fuel_type == 'Diesel' else 0,
        'Fuel Type_LPG': 1 if fuel_type == 'LPG' else 0,
        'Fuel Type_Petrol': 1 if fuel_type == 'Petrol' else 0,
        'Transmission_Manual': 1 if transmission == 'Manual' else 0,
        'Owner_Second': 1 if owner == 'Second' else 0,
        'Owner_Third': 1 if owner == 'Third' else 0,
        'Owner_Unregistered Car': 0, # Simplificamos
        'Seller Type_Individual': 1 if seller_type == 'Individual' else 0,
        'Seller Type_Corporate': 0 # Simplificamos
    }
    features = pd.DataFrame(data, index=[0])
    # Nos aseguramos de que el dataframe de entrada tenga las mismas columnas que el modelo espera
    final_features = features.reindex(columns=model_columns, fill_value=0)
    return final_features

input_df = user_input_features()

# --- MOSTRAR PREDICCIÓN ---
st.subheader('Predicción de Precio')

# Usamos el modelo para predecir sobre los datos de entrada
prediction = model.predict(input_df)

# Mostramos el precio predicho en un formato llamativo
st.success(f'El precio estimado del auto es: **${prediction[0]:,.2f}**')

st.write('---')
st.write('**Nota:** Esta es una estimación basada en un modelo entrenado con datos de la India. Los precios están en Rupias Indias (₹), pero se muestran con el símbolo de `$` por convención.')