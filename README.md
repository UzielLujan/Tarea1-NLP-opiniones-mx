# Text and Image Processing with Deep Learning

# Task 1 — Comprehensive Analysis of Spanish Corpus

This repository contains the implementation of a **complete Natural Language Processing (NLP) pipeline** applied to a Spanish corpus with 5 classes.  
The project is part of the **Master's in Statistical Computing** and covers everything from descriptive analysis to vector representations and supervised classification.

---

## Objective
Apply a comprehensive NLP pipeline to:
- Analyze and describe Spanish corpora.
- Evaluate empirical language laws (e.g., Zipf's Law).
- Explore lexical and grammatical structures.
- Build representations (BoW, TF-IDF, Word2Vec).
- Perform clustering, classification, and topic analysis.
- Explain results with quantitative and visual evidence.

---

## Project Structure
```bash
Tarea1_NLP-opiniones-mx/
│── data/                 # Corpus and processed data
│   ├── raw/              # Original corpus
│   ├── interim/          # Intermediate data (tokens, embeddings)
│   └── processed/        # Data ready for modeling
│
│── src/                  # Source code in modules
│   ├── preprocessing/    # Cleaning, stopwords, tokenization, stemming
│   ├── analysis/         # Descriptive statistics, Zipf's Law, hapax
│   ├── representaciones/ # BoW, TF-IDF, bigrams, feature selection
│   ├── embeddings/       # Word2Vec, doc embeddings
│   ├── modeling/         # Classification (SVM, Logistic Regression)
│   ├── clustering/       # K-means, cluster analysis
│   ├── topics/           # LSA with 50 topics
│   └── visualization/    # Graphs and utilities
│
│── notebooks/            # Experiments and quick prototypes
│
│── reports/
│   ├── figures/          # Generated images
│   └── Tarea1_Reporte.pdf
│
│── tests/                # Unit tests for key functions
│
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── main.py               # Main script that runs the complete pipeline
```

## Reproducible Environment

To run this project locally, use the `conda` environment defined in `environment.yml`.

### Create environment from scratch

```bash
conda env create -f environment.yml
conda activate tarea1-nlp
```
## Run the complete pipeline
To execute the complete pipeline, run:
```bash
python main.py --pipeline_completo
```

## Some commands to run the pipeline in parts

0. General corpus statistics
```bash
python main.py --estadisticas
```

1. Analyze Zipf's Law without stopwords and with top_n=100:
```bash
python main.py --zipf --sin_stopwords --top_n 100
```

2. Generate frequent words and 4-gram POS:
```bash
python main.py --frecuentes --pos
```
3. Train Word2Vec and run analogies:
```bash
python main.py --word2vec
```
4. Create BoW and TF-IDF with bigrams and min_df=10:
```bash
python main.py --bow --min_df 10 --ngram_max 2
```

5. Visualize vector representations with PCA
```bash
python main.py --pca --pca_tipo pkl --pca_path data/interim/bow_vectorizer.pkl --pca_title "PCA BoW"
python main.py --pca --pca_tipo pkl --pca_path data/interim/tdidf_vectorizer.pkl --pca_title "PCA TDiDF"
python main.py --pca --pca_tipo model --pca_path data/interim/word2vec.model --pca_title "PCA Word2Vec"
python main.py --pca --pca_tipo npy --pca_path data/interim/doc_embeddings.npy --pca_title "PCA Doc Embeddings"
```
6. Run classification (Logistic Regression) with different preprocessing levels:
```bash
python main.py --clasificacion
``` 

7. Run only LSA:
```bash
python main.py --lsa
```
> **Note:** To check the complete list of commands and usage examples for the pipeline by modules, review the file [`argparse_commands.md`](./argparse_commands.md).