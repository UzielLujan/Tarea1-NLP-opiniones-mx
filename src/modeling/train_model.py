# src/modeling/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def ejecutar_experimentos_clasificacion(
    experimentos,
    columna_texto="Review",
    columna_clase="Polarity"
):
    print("\n🤖 Ejecutando clasificación con distintas variantes de preprocesamiento...")

    output_dir = os.path.join("reports", "modeling")
    os.makedirs(output_dir, exist_ok=True)

    metricas_resumen = []

    for i, exp in enumerate(experimentos):
        print(f"\n🔄 Experimento {i+1}: {exp['descripcion']}")

        # Si el experimento trae X_override, úsalo directamente (para embeddings)
        if "X_override" in exp:
            X = exp["X_override"]
            y = exp["y_override"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo = exp.get('modelo', LogisticRegression(
                            multi_class="auto",
                            max_iter=2000,
                            random_state=42
                        ))
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
        else:
            df = exp["corpus"]
            X = df[columna_texto].astype(str)
            y = df[columna_clase].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            vectorizer = exp['vectorizer']
            modelo = exp.get('modelo', LogisticRegression(
                            multi_class="auto",
                            max_iter=2000,
                            random_state=42
                        ))
            clf = Pipeline([
                ("vect", vectorizer),
                ("clf", modelo)
            ])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        matriz = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Mostrar métricas en consola
        print(f"- Accuracy: {acc:.4f}")
        print(f"- F1 macro: {f1:.4f}")

        # Guardar métricas para resumen
        metricas_resumen.append({
            "Experimento": exp['descripcion'],
            "Accuracy": acc,
            "F1_macro": f1
        })

        # Guardar matriz como imagen
        plt.figure(figsize=(6, 5))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - {exp['descripcion']}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()

        img_path = os.path.join(output_dir, f"confusion_{i+1}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"- Matriz de confusión guardada en {img_path}")

    # Guardar resumen de métricas en CSV
    df_metricas = pd.DataFrame(metricas_resumen)
    resumen_path = os.path.join(output_dir, "resumen_metricas.csv")
    df_metricas.to_csv(resumen_path, index=False)
    print(f"\nResumen de métricas guardado en {resumen_path}")