# src/features/feature_selection.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif
import os
def seleccionar_caracteristicas(bow_matrix, tfidf_matrix, labels, vectorizer_bow, vectorizer_tfidf, top_k=20):
    print("\n🔍 Selección de características...")
    labels = pd.Series(labels).astype(int).values  # Asegura que labels es un array numpy de enteros
    nombres_bow = np.array(vectorizer_bow.get_feature_names_out())
    nombres_tfidf = np.array(vectorizer_tfidf.get_feature_names_out())

    # Chi-cuadrado sobre BoW
    chi2_scores_bow, _ = chi2(bow_matrix, labels)
    top_chi2_idx_bow = np.argsort(chi2_scores_bow)[-top_k:][::-1]
    top_chi2_terms_bow = nombres_bow[top_chi2_idx_bow]

    # Chi-cuadrado sobre TF-IDF
    chi2_scores_tfidf, _ = chi2(tfidf_matrix, labels)
    top_chi2_idx_tfidf = np.argsort(chi2_scores_tfidf)[-top_k:][::-1]
    top_chi2_terms_tfidf = nombres_tfidf[top_chi2_idx_tfidf]

    # Información mutua sobre BoW
    mi_scores_bow = mutual_info_classif(bow_matrix, labels, discrete_features=True)
    top_mi_idx_bow = np.argsort(mi_scores_bow)[-top_k:][::-1]
    top_mi_terms_bow = nombres_bow[top_mi_idx_bow]

    # Información mutua sobre TF-IDF
    mi_scores_tfidf = mutual_info_classif(tfidf_matrix, labels, discrete_features=True)
    top_mi_idx_tfidf = np.argsort(mi_scores_tfidf)[-top_k:][::-1]
    top_mi_terms_tfidf = nombres_tfidf[top_mi_idx_tfidf]

    print(f"- Top {top_k} términos por Chi² (BoW):")
    print(top_chi2_terms_bow)
    print(f"- Top {top_k} términos por Chi² (TF-IDF):")
    print(top_chi2_terms_tfidf)
    print(f"- Top {top_k} términos por Información Mutua (BoW):")
    print(top_mi_terms_bow)
    print(f"- Top {top_k} términos por Información Mutua (TF-IDF):")
    print(top_mi_terms_tfidf)

    # Guardar resultados
    output_dir = os.path.join("reports", "features")
    os.makedirs(output_dir, exist_ok=True)
    pd.Series(top_chi2_terms_bow).to_csv(os.path.join(output_dir, "top_chi2_bow.csv"), index=False)
    pd.Series(top_chi2_terms_tfidf).to_csv(os.path.join(output_dir, "top_chi2_tfidf.csv"), index=False)
    pd.Series(top_mi_terms_bow).to_csv(os.path.join(output_dir, "top_mi_bow.csv"), index=False)
    pd.Series(top_mi_terms_tfidf).to_csv(os.path.join(output_dir, "top_mi_tfidf.csv"), index=False)

    print(f"- Resultados guardados en {output_dir}")

    return {
        "chi2_bow": top_chi2_terms_bow,
        "chi2_tfidf": top_chi2_terms_tfidf,
        "mi_bow": top_mi_terms_bow,
        "mi_tfidf": top_mi_terms_tfidf,
    }

