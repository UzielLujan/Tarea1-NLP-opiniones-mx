# src/visualization/estadisticas.py

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # type: ignore
import os
import pandas as pd

sns.set(style="whitegrid")

def plot_distribucion_clases(df, columna_clase="Polarity", output_path="reports/figures/distribucion_"):
    paleta = "Reds"
    conteo = df[columna_clase].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=conteo.index.astype(str),
        y=conteo.values,
        hue=conteo.index.astype(str),
        palette=paleta,
        legend=False,
        ax=ax
    )
    ax.set_title(f"Distribución de la Variable '{columna_clase}'", fontsize=16)
    ax.set_ylabel("Número de Reseñas", fontsize=12)
    ax.set_xlabel(columna_clase, fontsize=12)
    for i, v in enumerate(conteo.values):
        ax.text(i, v + max(conteo.values)*0.01, str(v), color='black', ha='center', fontsize=9)

    os.makedirs(os.path.dirname(output_path + columna_clase +".png"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path + columna_clase +".png")
    plt.close()
    print(f"- Distribución de clases guardada en {output_path}")


def plot_palabras_frecuentes_por_clase(diccionario_palabras, output_dir="reports/figures/palabras_frecuentes"):
    os.makedirs(output_dir, exist_ok=True)
    for clase, lista in diccionario_palabras.items():
        palabras, freqs = zip(*lista)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=list(freqs),
            y=list(palabras),
            palette="Reds",
            ax=ax
        )
        ax.set_title(f"Top palabras frecuentes - Clase {clase}", fontsize=16)
        ax.set_xlabel("Frecuencia", fontsize=12)
        ax.set_ylabel("Palabra", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_palabras_clase_{clase}.png"), dpi=300)
        print(f"- Gráfica de palabras frecuentes clase {clase} guardada")
        plt.close()

def generar_nube_palabras(lista_tokens, output_path="reports/figures/nube_palabras.png", titulo="Nube de palabras"):
    texto = " ".join(lista_tokens)
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(texto)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(titulo, fontsize=14)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"- Nube de palabras guardada en {output_path}")
