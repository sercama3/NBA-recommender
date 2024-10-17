import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Cargar los datos de los jugadores
@st.cache_data
def cargar_datos():
    jugadores2023 = pd.read_csv("data/jugadoresNBA2023.csv", sep=";")
    jugadores2022 = pd.read_csv("data/jugadoresNBA2022.csv", sep=";")
    
    # Crear estadísticas por partido
    jugadores2023['PPG'] = jugadores2023['PTS'] / jugadores2023['GP']
    jugadores2023['APG'] = jugadores2023['AST'] / jugadores2023['GP']
    jugadores2023['RPG'] = jugadores2023['REB'] / jugadores2023['GP']
    jugadores2023['STLPG'] = jugadores2023['STL'] / jugadores2023['GP']
    jugadores2023['BLKPG'] = jugadores2023['BLK'] / jugadores2023['GP']

    # Seleccionar las columnas importantes
    columnas = ['PLAYER','AGE','PPG','3PM','APG','RPG','STLPG','BLKPG','FG%','FT%','3P%']
    jugadores = jugadores2023[columnas]

    # Normalizar las estadísticas
    stats = jugadores.drop('PLAYER', axis=1)
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats)

    # Crear DataFrame de similitud de coseno
    similarity_matrix = cosine_similarity(scaled_stats)
    similarity_df = pd.DataFrame(similarity_matrix, index=jugadores['PLAYER'], columns=jugadores['PLAYER'])

    return jugadores, similarity_df, scaler

# Cargar datos (se cachea para mejorar rendimiento)
jugadores, similarity_df, scaler = cargar_datos()

# Función para recomendar jugadores basados en un nombre
def recomendar_jugadores(jugador, n_recomendaciones=10):
    # Verificar si el jugador está en la base de datos
    if jugador not in similarity_df.index:
        st.write(f"El jugador {jugador} no está en la base de datos.")
        return
    
    # Obtener similitudes del jugador
    similitudes = similarity_df[jugador]
    
    # Ordenar jugadores por similitud y excluir al jugador mismo
    similitudes = similitudes.drop(jugador)
    jugadores_similares = similitudes.sort_values(ascending=False).head(n_recomendaciones)
    
    st.write(f"Jugadores similares en 2023 a {jugador}:\n")
    for similar_jugador, score in jugadores_similares.items():
        st.write(f"{similar_jugador} con una similitud de {score:.2f}")

# Función para recomendar jugadores en base a estadísticas personalizadas
def recomendar_jugadores_por_estadisticas(estadisticas_objetivo, scaled_df, jugadores, n=5):
    # Normalizar las estadísticas objetivo
    estadisticas_objetivo_normalizadas = scaler.transform([estadisticas_objetivo])
    
    # Calcular la similitud entre las estadísticas objetivo y todos los jugadores
    similitudes = cosine_similarity(estadisticas_objetivo_normalizadas, scaled_df)
    
    # Obtener los jugadores más similares
    mejores_indices = np.argsort(similitudes[0])[::-1][:n]  # Ordenar y seleccionar los top n
    
    # Mostrar los jugadores recomendados
    st.write("Jugadores recomendados basados en tus estadísticas objetivo:")
    for i in mejores_indices:
        jugador = jugadores.iloc[i]['PLAYER']
        st.write(f"{jugador}")

# Interfaz de usuario con Streamlit
st.title("Recomendador de Jugadores NBA 2023")

# Descripción de la aplicación
st.write("Encuentra jugadores similares o basados en estadísticas personalizadas.")

# **Primera opción**: Selecciona un jugador de la lista desplegable
jugador = st.selectbox("Selecciona un jugador:", jugadores['PLAYER'].unique())

# Botón para ejecutar la recomendación
if st.button("Recomendar jugadores similares"):
    recomendar_jugadores(jugador)

# **Segunda opción**: Ingresar estadísticas personalizadas
st.write("Recomendar jugadores basados en estadísticas personalizadas")

# Pedir al usuario estadísticas objetivo
ppg = st.number_input("Puntos por partido (PPG)", min_value=0.0, max_value=50.0, value=10.0)
rpg = st.number_input("Rebotes por partido (RPG)", min_value=0.0, max_value=20.0, value=5.0)
apg = st.number_input("Asistencias por partido (APG)", min_value=0.0, max_value=15.0, value=3.0)
blkpg = st.number_input("Bloqueos por partido (BLKPG)", min_value=0.0, max_value=5.0, value=1.0)
stlpg = st.number_input("Robos por partido (STLPG)", min_value=0.0, max_value=5.0, value=1.0)

# Botón para ejecutar la recomendación basada en estadísticas
if st.button("Recomendar jugadores basados en estadísticas personalizadas"):
    estadisticas_objetivo = [ppg, rpg, apg, blkpg, stlpg]
    recomendar_jugadores_por_estadisticas(estadisticas_objetivo, scaler.transform(jugadores.drop('PLAYER', axis=1)), jugadores)
