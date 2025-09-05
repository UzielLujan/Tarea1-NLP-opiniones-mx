# src/analysis/zipf.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import Counter
import os
import scipy.stats
nltk.download('punkt', quiet=True)

def analizar_zipf(df: pd.DataFrame, columna_texto="Review", top_n=None):
    print("\n📈 Analizando la Ley de Zipf...")

    # Tokenización con NLTK
    tokens = df[columna_texto].apply(lambda x: nltk.word_tokenize(str(x), language='spanish')).explode()

    # Conteo de frecuencias
    frecuencia = Counter(tokens)
    palabras_ordenadas = frecuencia.most_common()
    print('Top 10 palabras:')
    for i, (palabra, freq) in enumerate(palabras_ordenadas[:10]):
        print(f" {i + 1}. {palabra}: {freq}")
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
    slope, intercept, r_value, _, _ = scipy.stats.linregress(log_rangos_fit, log_frecuencias_fit)
    pendiente = slope
    interseccion = intercept
    r2 = r_value ** 2

    print(f"- Exponente de Zipf (s): {-pendiente:.4f}")
    print(f"- Constante C (modelo): {10**interseccion:.2f}, Frecuencia palabra más común: {frecuencias[0]:.2f}")
    print(f"- Δ C vs f(1): {abs(10**interseccion - frecuencias[0]):.2f}")
    print(f"- Coeficiente de determinación (R²): {r2:.4f}")

    # Graficar
    plt.figure(figsize=(8, 5))
    plt.scatter(log_rangos_fit, log_frecuencias_fit, label='Datos reales', alpha=0.7)
    plt.plot(log_rangos_fit, pendiente * log_rangos_fit + interseccion, color='red', linestyle='--', label='Ajuste lineal')
    plt.title("Ley de Zipf - log(Rango) vs log(Frecuencia)")
    plt.xlabel("log10(Rango)")
    plt.ylabel("log10(Frecuencia)")
    plt.legend()
    plt.grid(True)

    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "sin_stopwords" if top_n else "con_stopwords"
    if top_n:
        suffix += f"_top{top_n}"
    filename = f"zipf_plot_{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"- Gráfica guardada en {output_path}")
    plt.close()
