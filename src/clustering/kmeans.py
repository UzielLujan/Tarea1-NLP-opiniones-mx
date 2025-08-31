# src/clustering/kmeans.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

def clusterizar_documentos(embeddings: np.ndarray, df_original: pd.DataFrame, columna_texto="Review", columna_clase="Polarity", k=5):
    print("\n📊 Aplicando K-means sobre embeddings de documentos...")

    modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    etiquetas_cluster = modelo_kmeans.fit_predict(embeddings)

    df_clusters = df_original.copy()
    df_clusters["cluster"] = etiquetas_cluster

    # Extraer top 5 documentos más cercanos al centroide por cluster
    resultados = {}
    for i in range(k):
        idx_cluster = np.where(etiquetas_cluster == i)[0]
        centroid = modelo_kmeans.cluster_centers_[i]
        distancias = np.linalg.norm(embeddings[idx_cluster] - centroid, axis=1)
        top_indices = idx_cluster[np.argsort(distancias)[:5]]
        textos_top = df_original.iloc[top_indices][columna_texto].tolist()
        resultados[i] = textos_top

        print(f"\n🌀 Cluster {i} (centroide):")
        for texto in textos_top:
            print(f"- {texto[:100]}...")

    # Guardar asignación de clusters
    output_dir = os.path.join("reports", "clustering")
    os.makedirs(output_dir, exist_ok=True)
    df_clusters.to_csv(os.path.join(output_dir, "documentos_clusterizados.csv"), index=False)

    print(f"- Asignación de clusters guardada en {output_dir}/documentos_clusterizados.csv")

    return df_clusters, resultados
