# src/analysis/zipf.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import Counter
import os

nltk.download('punkt', quiet=True)

def analizar_zipf(df: pd.DataFrame, columna_texto="Review", eliminar_stopwords=True, stopwords_set=None):
    print("\n📈 Analizando la Ley de Zipf...")

    # Tokenización con NLTK
    tokens = df[columna_texto].apply(lambda x: nltk.word_tokenize(str(x), language='spanish')).explode()
    
    if eliminar_stopwords and stopwords_set:
        tokens = tokens[~tokens.str.lower().isin(stopwords_set)]

    # Conteo de frecuencias
    frecuencia = Counter(tokens)
    palabras_ordenadas = frecuencia.most_common()

    rangos = np.arange(1, len(palabras_ordenadas) + 1)
    frecuencias = np.array([f for _, f in palabras_ordenadas])

    log_rangos = np.log10(rangos)
    log_frecuencias = np.log10(frecuencias)

    # Ajuste de regresión lineal
    coef = np.polyfit(log_rangos, log_frecuencias, deg=1)
    pendiente, interseccion = coef

    print(f"- Exponente de Zipf (s): {-pendiente:.4f}")
    print(f"- Constante C (aprox frecuencia palabra más común): {10**interseccion:.2f}")

    # Graficar
    plt.figure(figsize=(8, 5))
    plt.plot(log_rangos, log_frecuencias, label='Datos reales', alpha=0.7)
    plt.plot(log_rangos, np.polyval(coef, log_rangos), '--', label='Ajuste lineal')
    plt.title("Ley de Zipf - log(Rango) vs log(Frecuencia)")
    plt.xlabel("log10(Rango)")
    plt.ylabel("log10(Frecuencia)")
    plt.legend()
    plt.grid(True)

    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "zipf_plot.png")
    plt.savefig(output_path)
    print(f"- Gráfica guardada en {output_path}")
    plt.close()
