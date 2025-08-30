# src/analysis/estadisticas.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os

# Descargar stopwords si no están disponibles
nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))

def generar_estadisticas(df: pd.DataFrame, columna_texto="Review", columna_clase="Polarity"):
    print("\n📊 Estadísticas del corpus:")

    # Total de documentos
    num_docs = df.shape[0]
    print(f"- Número de documentos: {num_docs}")

    # Tokenización básica por espacios
    tokens = df[columna_texto].str.split().explode()
    total_tokens = tokens.shape[0]
    print(f"- Número total de tokens: {total_tokens}")

    # Vocabulario único
    vocabulario = set(tokens)
    print(f"- Tamaño del vocabulario: {len(vocabulario)}")

    # Hapax legomena
    conteo = Counter(tokens)
    hapax = [palabra for palabra, freq in conteo.items() if freq == 1]
    print(f"- Hapax legomena: {len(hapax)} ({len(hapax)/len(vocabulario)*100:.2f}%)")

    # Stopwords
    num_stopwords = sum(1 for palabra in tokens if palabra.lower() in stopwords_es)
    porcentaje_stopwords = (num_stopwords / total_tokens) * 100
    print(f"- Stopwords: {num_stopwords} tokens ({porcentaje_stopwords:.2f}%)")

    # Estadísticas por clase
    print("\n📂 Estadísticas por clase:")
    clases = df[columna_clase].unique()
    for clase in sorted(clases):
        subset = df[df[columna_clase] == clase]
        tokens_clase = subset[columna_texto].str.split().explode()
        vocab_clase = set(tokens_clase)
        print(f"Clase {clase}: {subset.shape[0]} docs, {tokens_clase.shape[0]} tokens, {len(vocab_clase)} vocab")
