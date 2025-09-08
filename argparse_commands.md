1. Analizar Ley de Zipf sin stopwords y con top_n=100:
python main.py --zipf --sin_stopwords --top_n 100

2. Generar palabras frecuentes y 4-gramas POS:
python main.py --frecuentes --pos

3. Entrenar Word2Vec y correr analogías:
python main.py --word2vec

3.1 Entrenar Word2Vec y clusterizar
python main.py --word2vec --cluster
python main.py --word2vec --cluster --custom_words

4. Crear BoW y TF-IDF con bigramas y min_df=10:
python main.py --bow --min_df 10 --ngram_max 2

5. Ejecutar solo LSA:
python main.py --lsa

6. Visualizar representacion vectoriales con PCA
python main.py --pca --pca_tipo pkl --pca_path data/interim/bow_vectorizer.pkl --pca_title "PCA BoW"
python main.py --pca --pca_tipo pkl --pca_path data/interim/tdidf_vectorizer.pkl --pca_title "PCA TDiDF"
python main.py --pca --pca_tipo model --pca_path data/interim/word2vec.model --pca_title "PCA Word2Vec"
python main.py --pca --pca_tipo npy --pca_path data/interim/doc_embeddings.npy --pca_title "PCA Doc Embeddings"