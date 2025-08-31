# src/features/feature_selection.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif
import os


def seleccionar_caracteristicas(bow_matrix, tfidf_matrix, labels, vectorizer_bow, vectorizer_tfidf, top_k=20):
    print("\n🔍 Selección de características...")

    nombres_bow = np.array(vectorizer_bow.get_feature_names_out())
    nombres_tfidf = np.array(vectorizer_tfidf.get_feature_names_out())

    # Chi-cuadrado sobre BoW
    chi2_scores, _ = chi2(bow_matrix, labels)
    top_chi2_idx = np.argsort(chi2_scores)[-top_k:][::-1]
    top_chi2_terms = nombres_bow[top_chi2_idx]

    # Información mutua sobre TF-IDF
    mi_scores = mutual_info_classif(tfidf_matrix, labels, discrete_features=True)
    top_mi_idx = np.argsort(mi_scores)[-top_k:][::-1]
    top_mi_terms = nombres_tfidf[top_mi_idx]

    print(f"- Top {top_k} términos por Chi²:")
    print(top_chi2_terms)

    print(f"\n- Top {top_k} términos por Información Mutua:")
    print(top_mi_terms)

    # Guardar resultados
    output_dir = os.path.join("reports", "features")
    os.makedirs(output_dir, exist_ok=True)

    pd.Series(top_chi2_terms).to_csv(os.path.join(output_dir, "top_chi2.csv"), index=False)
    pd.Series(top_mi_terms).to_csv(os.path.join(output_dir, "top_mi.csv"), index=False)

    print(f"- Resultados guardados en {output_dir}")

    return top_chi2_terms, top_mi_terms
