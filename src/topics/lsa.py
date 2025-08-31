# src/topics/lsa.py

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os

def ejecutar_lsa(tfidf_matrix, vectorizer_tfidf, n_topics=50, top_n=10):
    print("\n🧠 Ejecutando LSA (Latent Semantic Analysis)...")

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    términos = np.array(vectorizer_tfidf.get_feature_names_out())
    topicos = {}

    for i, comp in enumerate(svd.components_):
        top_terms = términos[np.argsort(comp)[-top_n:][::-1]]
        topicos[f"Tópico {i+1}"] = top_terms
        print(f"\nTópico {i+1}:")
        print(", ".join(top_terms))

    # Guardar resultados
    output_dir = os.path.join("reports", "topics")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame.from_dict(topicos, orient="index").to_csv(os.path.join(output_dir, "topicos_lsa.csv"))
    print(f"- Tópicos guardados en {output_dir}/topicos_lsa.csv")

    return lsa_matrix, topicos
