# src/embeddings/word2vec.py

import pandas as pd
import nltk
from gensim.models import Word2Vec
import os
import pickle

nltk.download('punkt', quiet=True)

def entrenar_word2vec(df: pd.DataFrame, columna_texto="Review", vector_size=100, window=5, min_count=5, workers=4, sg=1, guardar=True):
    print("\n🧠 Entrenando modelo Word2Vec...")

    # Tokenización por documento
    corpus_tokenizado = df[columna_texto].apply(lambda x: nltk.word_tokenize(str(x), language='spanish')).tolist()

    # Entrenamiento del modelo
    modelo = Word2Vec(
        sentences=corpus_tokenizado,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg  # sg=1 -> skip-gram, sg=0 -> CBOW
    )

    if guardar:
        output_dir = os.path.join("data", "interim")
        os.makedirs(output_dir, exist_ok=True)
        modelo.save(os.path.join(output_dir, "word2vec.model"))
        print(f"- Modelo Word2Vec guardado en {output_dir}/word2vec.model")

    return modelo

def analogias(modelo: Word2Vec, ejemplos=None):
    print("\n🔁 Ejemplos de analogías semánticas:")

    if ejemplos is None:
        ejemplos = [
            ("comida", "restaurante", "hotel"),
            ("playa", "sol", "noche"),
            ("turismo", "viaje", "trabajo"),
            ("bueno", "mejor", "malo"),
            ("servicio", "atención", "precio")
        ]

    for a, b, c in ejemplos:
        try:
            resultado = modelo.wv.most_similar(positive=[b, c], negative=[a], topn=1)
            print(f"- {a} : {b} :: {c} : {resultado[0][0]} (score={resultado[0][1]:.2f})")
        except KeyError as e:
            print(f"  [!] Palabra no encontrada en vocabulario: {e}")



