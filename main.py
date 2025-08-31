# Script principal para orquestar el pipeline

from src.preprocessing.limpieza import limpiar_corpus

from src.analysis.estadisticas import generar_estadisticas

from src.analysis.zipf import analizar_zipf

#from src.analysis.palabras_y_pos import palabras_frecuentes_por_clase, pos_4gramas_por_clase

from src.features.bow import construir_bow_tfidf

from src.features.feature_selection import seleccionar_caracteristicas

#from src.embeddings.word2vec import entrenar_word2vec, analogias

#from src.embeddings.doc_embeddings import calcular_doc_embeddings
#from src.clustering.kmeans import clusterizar_documentos
'''
from src.modeling.train_model import ejecutar_experimentos_clasificacion
from src.topics.lsa import ejecutar_lsa
from src.visualization.utils import configurar_estilo_visual
'''
import os
from nltk.corpus import stopwords
stopwords_es = set(stopwords.words('spanish'))

def main():
    print("🔧 Iniciando pipeline de Tarea 1 NLP...")
    
    archivo = "MeIA_2025_train.csv"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Paso 1: Preprocesamiento
    corpus_limpio = limpiar_corpus(base_dir, archivo)

    # Paso 2: Análisis descriptivo
    generar_estadisticas(corpus_limpio)
    analizar_zipf(corpus_limpio, columna_texto='Review', eliminar_stopwords=False, stopwords_set=stopwords_es, top_n=100)

    # Paso 3: Representaciones BoW / TF-IDF
    #bow, tfidf, vec_bow, vec_tfidf = construir_bow_tfidf(corpus_limpio)
    
    
    # Paso 4: Selección de características
    '''
    top_chi2, top_mi = seleccionar_caracteristicas(
        bow_matrix=bow,
        tfidf_matrix=tfidf,
        labels=corpus_limpio["Polarity"],
        vectorizer_bow=vec_bow,
        vectorizer_tfidf=vec_tfidf
    )
    '''
    # Paso 5: Embeddings Word2Vec
    #model_w2v = entrenar_word2vec(corpus_limpio)
    #analogias(model_w2v)
    
    # Paso 6: Embeddings de documento y clusterización
    #embeddings = calcular_doc_embeddings(corpus_limpio, model_w2v)
    #df_clusterizado, textos_por_cluster = clusterizar_documentos(embeddings, corpus_limpio)
    '''
    # Paso 7: Clasificación con distintas variantes de preprocesamiento
    ejecutar_experimentos_clasificacion("data/processed/dataset.csv")

    # Paso 8: LSA
    ejecutar_lsa(tfidf)

    print("✅ Pipeline completado. Revisa el reporte en /reports")
    '''
if __name__ == "__main__":
    main()
