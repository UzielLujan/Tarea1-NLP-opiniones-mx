# Modulo de Preprocesamiento para el corpus de texto


import pandas as pd
import re 
import os

def limpiar_truncamientos(texto):
    return re.sub(r'\.{2,}\s*Más\b.*', '', texto, flags=re.IGNORECASE)

def normalizar_espacios(texto):
    texto = texto.replace('\n', ' ')
    texto = re.sub(r'\s+', ' ', texto)
    # Eliminar puntos, comas y signos de puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

def limpiar_corpus(ruta, archivo):
    data_path = os.path.join(ruta, "data", "raw", archivo)
    print('\n',f'Cargando corpus desde: {data_path}...')
    corpus = pd.read_csv(data_path, encoding="utf-8")
    corpus['Review'] = corpus['Review'].apply(limpiar_truncamientos)
    corpus['Review'] = corpus['Review'].apply(normalizar_espacios)
    return corpus