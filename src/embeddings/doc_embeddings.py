# src/embeddings/doc_embeddings.py

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
import pickle

def calcular_doc_embeddings(df: pd.DataFrame, modelo: Word2Vec, columna_texto="Review"):
    print("\n📐 Calculando embeddings de documento...")

    def vector_promedio(texto):
        palabras = texto.split()
        vectores = [modelo.wv[word] for word in palabras if word in modelo.wv]
        if len(vectores) == 0:
            return np.zeros(modelo.vector_size)
        return np.mean(vectores, axis=0)

    embeddings = df[columna_texto].apply(vector_promedio)
    matriz = np.vstack(embeddings.values)

    output_dir = os.path.join("data", "interim")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "doc_embeddings.npy"), matriz)

    print(f"- Embeddings de documentos guardados en {output_dir}/doc_embeddings.npy")
    return matriz
