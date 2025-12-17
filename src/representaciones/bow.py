# src/features/bow.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import pickle

def construir_bow_tfidf(df: pd.DataFrame, columna_texto="Review", ngram_range=(1,1), min_df=5, max_df=1.0, guardar=True):
    print("\n🧠 Construyendo representaciones BoW y TF-IDF...")

    textos = df[columna_texto].astype(str).tolist()

    # BoW (bolsa de palabras con n-gramas)
    vectorizer_bow = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    bow_matrix = vectorizer_bow.fit_transform(textos)
    print(f"- BoW: {bow_matrix.shape[0]} documentos, {bow_matrix.shape[1]} características")

    # TF-IDF
    vectorizer_tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    tfidf_matrix = vectorizer_tfidf.fit_transform(textos)
    print(f"- TF-IDF: {tfidf_matrix.shape[0]} documentos, {tfidf_matrix.shape[1]} características")

    if guardar:
        output_dir = os.path.join("data", "interim")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "bow_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer_bow, f)

        with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer_tfidf, f)

        print(f"- Vectorizadores guardados en {output_dir}")

    return bow_matrix, tfidf_matrix, vectorizer_bow, vectorizer_tfidf
