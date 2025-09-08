## Comandos para ejecutar el pipeline por módulos

### 0. Estadísticas generales del corpus
```bash
python main.py --estadisticas
```

### 1. Distribución de clases (gráfica)
```bash
python main.py --plot_distribucion
```

### 2. Analizar Ley de Zipf
- Sin stopwords y top_n=100:
```bash
python main.py --zipf --sin_stopwords --top_n 100
```
- Con hapax removidos:
```bash
python main.py --zipf --sin_stopwords --sin_hapax --top_n 100
```

### 3. Palabras frecuentes por clase
- Top 15 palabras frecuentes (sin stopwords, filtrando palabras personalizadas):
```bash
python main.py --frecuentes --top_n 15 --custom_words
```
- Gráfica de palabras frecuentes:
```bash
python main.py --frecuentes --plot_palabras_frecuentes --top_n 15 --custom_words
```
- Nube de palabras por clase:
```bash
python main.py --frecuentes --nube_palabras --top_n 15 --custom_words
```

### 4. 4-gramas POS
- Global:
```bash
python main.py --pos_global
```
- Por clase:
```bash
python main.py --pos_clase
```

### 5. BoW y TF-IDF
- Con bigramas y min_df=10:
```bash
python main.py --bow --min_df 10 --ngram_max 2
```
- Filtrando palabras personalizadas:
```bash
python main.py --bow --custom_words
```

### 6. Selección de características
```bash
python main.py --seleccion
```

### 7. Entrenar Word2Vec y correr analogías
```bash
python main.py --word2vec
```
- Filtrando palabras personalizadas:
```bash
python main.py --word2vec --custom_words
```

### 8. Clustering de documentos (requiere Word2Vec entrenado)
```bash
python main.py --word2vec --cluster
```
- Filtrando palabras personalizadas:
```bash
python main.py --word2vec --cluster --custom_words
```

### 9. Visualización PCA de representaciones vectoriales
- BoW:
```bash
python main.py --pca --pca_tipo pkl --pca_path data/interim/bow_vectorizer.pkl --pca_title "PCA BoW"
```
- TF-IDF:
```bash
python main.py --pca --pca_tipo pkl --pca_path data/interim/tdidf_vectorizer.pkl --pca_title "PCA TDiDF"
```
- Word2Vec:
```bash
python main.py --pca --pca_tipo model --pca_path data/interim/word2vec.model --pca_title "PCA Word2Vec"
```
- Doc Embeddings:
```bash
python main.py --pca --pca_tipo npy --pca_path data/interim/doc_embeddings.npy --pca_title "PCA Doc Embeddings"
```

### 10. Clasificación supervisada (experimentos acumulativos)
```bash
python main.py --clasificacion
```

### 11. LSA (Latent Semantic Analysis)
```bash
python main.py --lsa
```

### 12. Ejecutar el pipeline completo
```bash
python main.py --pipeline_completo
```
