# src/modeling/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def ejecutar_experimentos_clasificacion(df: pd.DataFrame, columna_texto="Review", columna_clase="Polarity", preprocesamientos=[]):
    print("\n🤖 Ejecutando clasificación con distintas variantes de preprocesamiento...")

    X = df[columna_texto].astype(str)
    y = df[columna_clase].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    resultados = []
    output_dir = os.path.join("reports", "modeling")
    os.makedirs(output_dir, exist_ok=True)

    for i, preproc in enumerate(preprocesamientos):
        print(f"\n🔄 Experimento {i+1}: {preproc['descripcion']}")

        vectorizer = preproc['vectorizer']
        modelo = preproc.get('modelo', LogisticRegression(max_iter=1000))

        clf = Pipeline([
            ("vect", vectorizer),
            ("clf", modelo)
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        matriz = confusion_matrix(y_test, y_pred)

        resultados.append((preproc['descripcion'], report, matriz))

        # Guardar matriz como imagen
        plt.figure(figsize=(6, 5))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - {preproc['descripcion']}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()

        img_path = os.path.join(output_dir, f"confusion_{i+1}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"- Matriz de confusión guardada en {img_path}")

    return resultados
