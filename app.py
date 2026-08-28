import streamlit as st
import pandas as pd
import joblib
import os

# Título de la aplicación
st.title('Predicción del Precio de Vivienda en California')
st.write('Esta aplicación predice el valor promedio de una vivienda (en cientos de miles de dólares) en California usando un modelo de Regresión Lineal.')

# Cargar el modelo y el escalador
@st.cache_resource
def load_model_and_scaler():
    # Asegúrate de que los archivos están en el directorio 'processed_data'
    model_path = os.path.join('processed_data', 'linear_regression_model.joblib')
    scaler_path = os.path.join('processed_data', 'minmax_scaler.pkl')
    
    try:
        linear_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return linear_model, scaler
    except FileNotFoundError:
        st.error(f"Error: Los archivos del modelo o del escalador no se encontraron en '{os.path.abspath('processed_data')}'. Asegúrate de que los has guardado previamente y están en la ubicación correcta.")
        st.stop()

linear_model, scaler = load_model_and_scaler()

# Definir las características de entrada
st.sidebar.header('Características de la Vivienda')

# Las características deben coincidir con las usadas en el entrenamiento
# MedInc: median income in block group
# AveRooms: average number of rooms per household
# AveBedrms: average number of bedrooms per household
# AveOccup: average number of household members
# Latitude: block group latitude

def user_input_features():
    MedInc = st.sidebar.slider('Ingreso Mediano (MedInc)', 0.0, 15.0, 3.87)
    AveRooms = st.sidebar.slider('Promedio de Habitaciones (AveRooms)', 0.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider('Promedio de Dormitorios (AveBedrms)', 0.0, 5.0, 1.0)
    AveOccup = st.sidebar.slider('Promedio de Ocupantes (AveOccup)', 0.0, 10.0, 3.0)
    Latitude = st.sidebar.slider('Latitud (Latitude)', 32.0, 42.0, 34.0)
    
    data = {'MedInc': MedInc,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'AveOccup': AveOccup,
            'Latitude': Latitude}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Valores de entrada especificados:')
st.write(df_input)

# Escalar las características de entrada
df_input_scaled = scaler.transform(df_input)
df_input_scaled_df = pd.DataFrame(df_input_scaled, columns=df_input.columns, index=df_input.index)

st.subheader('Valores de entrada escalados:')
st.write(df_input_scaled_df)

# Realizar la predicción
if st.button('Predecir'):
    prediction = linear_model.predict(df_input_scaled_df)
    st.subheader('Predicción del Valor Mediano de Vivienda:')
    st.metric(label="Valor Predicho (en cientos de miles de $)", value=f"${prediction[0]:.2f}")
    st.write(f"Esto significa que el precio promedio predicho es de aproximadamente **${prediction[0]*100000:,.2f}** dólares.")
