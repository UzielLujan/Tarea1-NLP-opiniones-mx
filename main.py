import argparse
import os
from nltk.corpus import stopwords

from src.preprocessing.limpieza import limpiar_corpus
from src.analysis.estadisticas import generar_estadisticas
from src.analysis.zipf import analizar_zipf
from src.analysis.palabras_y_pos import palabras_frecuentes_por_clase, pos_4gramas_por_clase
from src.features.bow import construir_bow_tfidf
from src.features.feature_selection import seleccionar_caracteristicas
from src.embeddings.word2vec import entrenar_word2vec, analogias
from src.embeddings.doc_embeddings import calcular_doc_embeddings
from src.clustering.kmeans import clusterizar_documentos
from src.modeling.train_model import ejecutar_experimentos_clasificacion
from src.topics.lsa import ejecutar_lsa

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pipeline_completo", action="store_true")
    parser.add_argument("--zipf", action="store_true")
    parser.add_argument("--frecuentes", action="store_true")
    parser.add_argument("--pos", action="store_true")
    parser.add_argument("--bow", action="store_true")
    parser.add_argument("--seleccion", action="store_true")
    parser.add_argument("--word2vec", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--clasificacion", action="store_true")
    parser.add_argument("--lsa", action="store_true")

    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--sin_stopwords", action="store_true")
    parser.add_argument("--min_df", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=1)

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    archivo = "MeIA_2025_train.csv"
    corpus = limpiar_corpus(base_dir, archivo)
    stopwords_es = set(stopwords.words("spanish"))

    generar_estadisticas(corpus)

    if args.pipeline_completo:
        args.zipf = True
        args.frecuentes = True
        args.pos = True
        args.bow = True
        args.seleccion = True
        args.word2vec = True
        args.cluster = True
        args.clasificacion = True
        args.lsa = True

    if args.zipf:
        analizar_zipf(
            corpus,
            eliminar_stopwords=args.sin_stopwords,
            stopwords_set=stopwords_es,
            top_n=args.top_n
        )

    if args.frecuentes:
        palabras_frecuentes_por_clase(corpus)

    if args.pos:
        pos_4gramas_por_clase(corpus)

    if args.bow:
        global bow, tfidf, vec_bow, vec_tfidf
        bow, tfidf, vec_bow, vec_tfidf = construir_bow_tfidf(
            corpus,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df
        )

    if args.seleccion:
        seleccionar_caracteristicas(
            bow_matrix=bow,
            tfidf_matrix=tfidf,
            labels=corpus["Polarity"],
            vectorizer_bow=vec_bow,
            vectorizer_tfidf=vec_tfidf
        )

    if args.word2vec:
        global model_w2v
        model_w2v = entrenar_word2vec(corpus)
        analogias(model_w2v)

    if args.cluster:
        embeddings = calcular_doc_embeddings(corpus, model_w2v)
        clusterizar_documentos(embeddings, corpus)

    if args.clasificacion:
        preprocesamientos = [
            {"descripcion": "BoW sin preprocesamiento", "vectorizer": CountVectorizer()},
            {"descripcion": "TF-IDF min_df=10", "vectorizer": TfidfVectorizer(min_df=10)},
        ]
        ejecutar_experimentos_clasificacion(corpus, preprocesamientos=preprocesamientos)

    if args.lsa:
        ejecutar_lsa(tfidf_matrix=tfidf, vectorizer_tfidf=vec_tfidf)


if __name__ == "__main__":
    main()
