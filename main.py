import argparse
from email import parser
import os
from nltk.corpus import stopwords
from gensim.models import Word2Vec # type: ignore
import numpy as np
# Importar módulos personalizados
from src.preprocessing.limpieza import leer_corpus, procesar_corpus
from src.analysis.estadisticas import generar_estadisticas
from src.analysis.zipf import analizar_zipf
from src.analysis.palabras_y_pos import palabras_frecuentes_por_clase, pos_4gramas_global, pos_4gramas_por_clase
from src.representaciones.bow import construir_bow_tfidf
from src.representaciones.feature_selection import seleccionar_caracteristicas
from src.embeddings.word2vec import entrenar_word2vec, analogias
from src.embeddings.doc_embeddings import calcular_doc_embeddings
from src.clustering.kmeans import clusterizar_documentos
from src.modeling.train_model import ejecutar_experimentos_clasificacion
from src.topics.lsa import ejecutar_lsa
from src.visualization.visualizations import (
    plot_distribucion_clases,
    plot_palabras_frecuentes_por_clase,
    generar_nube_palabras,
    cargar_vectores,
    visualizar_pca
)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def main():
    parser = argparse.ArgumentParser()
    # Argumentos para seleccionar qué partes del pipeline ejecutar
    parser.add_argument("--pipeline_completo", action="store_true")
    parser.add_argument("--estadisticas", action="store_true")
    parser.add_argument("--zipf", action="store_true")
    parser.add_argument("--frecuentes", action="store_true")
    parser.add_argument("--pos_global", action="store_true")
    parser.add_argument("--pos_clase", action="store_true")
    parser.add_argument("--bow", action="store_true")
    parser.add_argument("--seleccion", action="store_true")
    parser.add_argument("--word2vec", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--clasificacion", action="store_true")
    parser.add_argument("--lsa", action="store_true")
    # Argumentos adicionales para análisis específicos
    parser.add_argument("--sin_stopwords", action="store_true")
    parser.add_argument("--sin_hapax", action="store_true")
    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--max_df", type=float, default=1.0)
    parser.add_argument("--ngram_max", type=int, default=1)
    parser.add_argument("--custom_words", action="store_true", help="Filtrar palabras personalizadas")
    # Argumentos para visualizaciones
    parser.add_argument("--plot_distribucion", action="store_true")
    parser.add_argument("--plot_palabras_frecuentes", action="store_true")
    parser.add_argument("--nube_palabras", action="store_true")
    parser.add_argument("--pca", action="store_true", help="Visualizar PCA de representaciones vectoriales")
    parser.add_argument("--pca_tipo", type=str, choices=["npy", "pkl", "model"], help="Tipo de vector: npy, pkl, model")
    parser.add_argument("--pca_path", type=str, help="Ruta al archivo de vectores (.npy, .pkl, .model)")
    parser.add_argument("--pca_title", type=str, default="PCA de representaciones vectoriales")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    archivo = "MeIA_2025_train.csv"
    stopwords_es = set(stopwords.words("spanish"))
    # Palabras detectadas manualmente que no aportan valor semántico a la Polaridad
    custom_words = {'hotel', 'restaurante', 'lugar', 'playa', 'comida', 'servicio', 'si', 'habitación', 'habitaciones'}

    # 0. Leer corpus una sola vez (limpieza básica)
    corpus = leer_corpus(base_dir, archivo, metodo='ftfy')

    if args.pipeline_completo:
        args.estadisticas = True
        args.zipf = True
        args.frecuentes = True
        args.pos_global = True
        args.pos_clase = True
        args.bow = True
        args.seleccion = True
        args.word2vec = True
        args.cluster = True
        args.clasificacion = True
        args.lsa = True

    # 1. Estadísticas y Descripcion del corpus: solo limpieza básica
    if args.estadisticas:
        generar_estadisticas(corpus)

    # Distribución de clases
    if args.plot_distribucion:
        plot_distribucion_clases(corpus, columna_clase="Polarity")

    # 2. Zipf: permite opción con o sin stopwords y con o sin hapax, siempre normalizando
    if args.zipf:
        corpus_zipf = procesar_corpus(
            corpus,
            eliminar_stop=args.sin_stopwords,
            stopwords_set=stopwords_es if args.sin_stopwords else None,
            normalizar_texto=True
        )
        analizar_zipf(
            corpus_zipf,
            stopws=args.sin_stopwords,
            top_n=args.top_n,
            quitar_hapax=args.sin_hapax
        )

    # 3. Palabras frecuentes por clase: normalización y sin stopwords
    if args.frecuentes:
        corpus_frec = procesar_corpus(
            corpus,
            eliminar_stop=True,
            stopwords_set=stopwords_es,
            normalizar_texto=True
        )
        palabras_frecuentes_por_clase(
            corpus_frec,
            top_n=args.top_n if args.top_n else 15,
            custom_words=custom_words  # Aquí se usan las palabras personalizadas definidas arriba
        )

    # Gráficas de palabras frecuentes por clase
    if args.plot_palabras_frecuentes and args.frecuentes:
        # Asegúrate de guardar el resultado de palabras_frecuentes_por_clase
        resultados_palabras = palabras_frecuentes_por_clase(
            corpus_frec,
            top_n=args.top_n if args.top_n else 15,
            custom_words=custom_words
        )
        plot_palabras_frecuentes_por_clase(resultados_palabras)
    # Nube de palabras general y por clase
    if args.nube_palabras and args.frecuentes:
        resultados_palabras = palabras_frecuentes_por_clase(
            corpus_frec,
            top_n=args.top_n if args.top_n else 15,
            custom_words=custom_words
        )
        for clase in resultados_palabras:
            if "filtrado" in str(clase):
                palabras_filtradas = [palabra for palabra, _ in resultados_palabras[clase]]
                generar_nube_palabras(
                    palabras_filtradas,
                    output_path=f"reports/figures/nube_palabras_{clase}.png",
                    titulo=f"Nube de palabras Clase {clase.replace('_filtrado', '')}"
                )

    # 4. POS global: normalización y sin stopwords
    if args.pos_global:
        corpus_pos = procesar_corpus(
            corpus,
            eliminar_stop=True,
            stopwords_set=stopwords_es,
            normalizar_texto=True
        )
        pos_4gramas_global(corpus_pos)

    # 4. POS por clase: normalización y sin stopwords y filtrado por globales
    if args.pos_clase:
        corpus_pos = procesar_corpus(
            corpus,
            eliminar_stop=True,
            stopwords_set=stopwords_es,
            normalizar_texto=True
        )
        top_global_4gramas = pos_4gramas_global(corpus, top_n=5)
        global_4gramas_set = set([ng for ng, _ in top_global_4gramas])
        pos_4gramas_por_clase(
            corpus,
            global_4gramas=global_4gramas_set
        )

    # 5. BoW y TF-IDF: normalización y sin stopwords
    if args.bow:
        global bow, tfidf, vec_bow, vec_tfidf
        corpus_bow = procesar_corpus(
            corpus,
            eliminar_stop=True,
            stopwords_set=stopwords_es,
            normalizar_texto=True,
            custom_words=custom_words if args.custom_words else None
        )
        bow, tfidf, vec_bow, vec_tfidf = construir_bow_tfidf(
            corpus_bow,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df
        )

    # Selección de características: usa el corpus y matrices generadas en BoW
    if args.seleccion:
        labels = corpus["Polarity"].astype(int).values
        seleccionar_caracteristicas(
            bow_matrix=bow,
            tfidf_matrix=tfidf,
            labels=labels,
            vectorizer_bow=vec_bow,
            vectorizer_tfidf=vec_tfidf
        )

    # Word2Vec: normalización y sin stopwords
    if args.word2vec:
        global model_w2v
        corpus_w2v = procesar_corpus(
            corpus,
            eliminar_stop=True,
            stopwords_set=stopwords_es,
            normalizar_texto=True,
            custom_words=custom_words if args.custom_words else None,
        )
        model_w2v = entrenar_word2vec(corpus_w2v)
        analogias(model_w2v)

    # Clustering: requiere embeddings y modelo Word2Vec entrenado
    if args.cluster:
        if 'model_w2v' not in globals():
            print("⚠️ Debes entrenar Word2Vec antes de clusterizar documentos.")
        else:
            embeddings = calcular_doc_embeddings(corpus_w2v, model_w2v)
            clusterizar_documentos(embeddings, corpus_w2v)

    # Visualización PCA de representaciones vectoriales
    if args.pca:
        if args.pca_tipo == "npy":
            vectores = cargar_vectores(args.pca_path, tipo="npy")
        elif args.pca_tipo == "pkl":
            vectores = cargar_vectores(args.pca_path, tipo="pkl", df=corpus)
        elif args.pca_tipo == "model":
            from gensim.models import Word2Vec # type: ignore
            modelo = Word2Vec.load(args.pca_path)
            vectores = cargar_vectores(None, tipo="model", modelo_word2vec=modelo, df=corpus)
        else:
            raise ValueError("Tipo de vector no soportado para PCA.")
        visualizar_pca(
            vectores,
            etiquetas=corpus["Polarity"],
            titulo=args.pca_title,
            output_path="reports/figures/" + args.pca_title + ".png"
        )

    # Clasificación: experimentos acumulativos con diferentes niveles de procesamiento

    if args.clasificacion:
        experimentos = [
            {
                "descripcion": "Sin preprocesamiento",
                "corpus": corpus,
                "vectorizer": CountVectorizer()
            },
            {
                "descripcion": "Con minúsculas",
                "corpus": procesar_corpus(corpus, normalizar_texto=True),
                "vectorizer": CountVectorizer(lowercase=True)
            },
            {
                "descripcion": "Con minúsculas y lematización",
                "corpus": procesar_corpus(corpus, normalizar_texto=True, lematizar_stem=True, metodo_lematizar="lematizar"),
                "vectorizer": CountVectorizer(lowercase=True)
            },
            {
                "descripcion": "Con minúsculas, lematización y min_df=10",
                "corpus": procesar_corpus(corpus, normalizar_texto=True, lematizar_stem=True, metodo_lematizar="lematizar"),
                "vectorizer": CountVectorizer(lowercase=True, min_df=10)
            },
            {
                "descripcion": "Embeddings de documentos (Word2Vec)",
                "X_override": np.load("data/interim/doc_embeddings.npy"),
                "y_override": corpus["Polarity"].astype(int).values
            }
        ]

        ejecutar_experimentos_clasificacion(experimentos)

    # LSA: usa la matriz tfidf generada en BoW
    if args.lsa:
        ejecutar_lsa(tfidf_matrix=tfidf, vectorizer_tfidf=vec_tfidf)

if __name__ == "__main__":
    main()
