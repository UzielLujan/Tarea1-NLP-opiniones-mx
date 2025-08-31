# src/analysis/zipf.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import Counter
import os

nltk.download('punkt', quiet=True)

def analizar_zipf(df: pd.DataFrame, columna_texto="Review", eliminar_stopwords=True, stopwords_set=None, top_n=None):
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

    # Recorte opcional para ajuste
    if top_n is not None:
        log_rangos_fit = log_rangos[:top_n]
        log_frecuencias_fit = log_frecuencias[:top_n]
    else:
        log_rangos_fit = log_rangos
        log_frecuencias_fit = log_frecuencias

    # Ajuste de regresión lineal
    coef = np.polyfit(log_rangos_fit, log_frecuencias_fit, deg=1)
    pendiente, interseccion = coef
    # Coeficiente de determinación
    r2 = 1 - (np.sum((log_frecuencias_fit - np.polyval(coef, log_rangos_fit))**2) / np.sum((log_frecuencias_fit - np.mean(log_frecuencias_fit))**2))
    print(f"- Exponente de Zipf (s): {-pendiente:.4f}")
    print(f"- Constante C (modelo): {10**interseccion:.2f}, Frecuencia palabra más común: {frecuencias[0]:.2f}")
    print(f"- Δ C vs f(1): {abs(10**interseccion - frecuencias[0]):.2f}")
    print(f"- Coeficiente de determinación (R²): {r2:.4f}")

    # Graficar
    plt.figure(figsize=(8, 5))
    plt.plot(log_rangos_fit, log_frecuencias_fit, label='Datos reales', alpha=0.7)
    plt.plot(log_rangos_fit, np.polyval(coef, log_rangos_fit), '--', label='Ajuste lineal')
    plt.title("Ley de Zipf - log(Rango) vs log(Frecuencia)")
    plt.xlabel("log10(Rango)")
    plt.ylabel("log10(Frecuencia)")
    plt.legend()
    plt.grid(True)

    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "no_stopwords" if eliminar_stopwords else "con_stopwords"
    if top_n:
        suffix += f"_top{top_n}"
    filename = f"zipf_plot_{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f"- Gráfica guardada en {output_path}")
    plt.close()
