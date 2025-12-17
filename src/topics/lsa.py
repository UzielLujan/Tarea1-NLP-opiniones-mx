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
    importancia = svd.explained_variance_ratio_

    print(f"\n🔎 Varianza explicada total: {importancia.sum():.4f}\n")

    for i, (comp, var_exp) in enumerate(zip(svd.components_, importancia)):
        top_terms = términos[np.argsort(comp)[-top_n:][::-1]]
        topicos[f"Tópico {i+1}"] = top_terms
        print(f"Tópico {i+1} (Varianza explicada = {var_exp:.4f}):")
        print(", ".join(top_terms) + "\n")

    # Crear DataFrame extendido con importancia
    df_topicos = pd.DataFrame.from_dict(topicos, orient="index")
    df_topicos["Varianza_Explicada"] = importancia

    # Ordenar por varianza explicada
    df_topicos = df_topicos.sort_values("Varianza_Explicada", ascending=False)

    # Guardar resultados
    output_dir = os.path.join("reports", "topics")
    os.makedirs(output_dir, exist_ok=True)
    df_topicos.to_csv(os.path.join(output_dir, "topicos_lsa.csv"))
    print(f"- Tópicos guardados en {output_dir}/topicos_lsa.csv")

    return lsa_matrix, df_topicos

