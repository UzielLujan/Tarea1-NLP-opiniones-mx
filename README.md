# Procesamiento de Texto e Imagenes con Deep Learning

# 📚 Tarea 1 — Análisis integral de corpus en español

Este repositorio contiene la implementación de un **pipeline completo de Procesamiento de Lenguaje Natural (NLP)** aplicado a un corpus en español con 5 clases.  
El proyecto forma parte de la **Maestría en Cómputo Estadístico** y cubre desde análisis descriptivo hasta representaciones vectoriales y clasificación supervisada.

---

## 🎯 Objetivo
Aplicar un pipeline integral de NLP para:
- Analizar y describir corpus en español.
- Evaluar leyes empíricas del lenguaje (ej. Ley de Zipf).
- Explorar estructuras léxicas y gramaticales.
- Construir representaciones (BoW, TF-IDF, Word2Vec).
- Realizar clustering, clasificación y análisis de tópicos.
- Explicar resultados con evidencia cuantitativa y visual.

---

## 📂 Estructura del proyecto
```bash
Tarea1_NLP-opiniones-mx/
│── data/                 # Corpus y datos procesados
│   ├── raw/              # Corpus original
│   ├── interim/          # Datos intermedios (tokens, embeddings)
│   └── processed/        # Datos listos para modelado
│
│── src/                  # Código fuente en módulos
│   ├── preprocessing/    # Limpieza, stopwords, tokenización, stemming
│   ├── analysis/         # Estadísticas descriptivas, Ley de Zipf, hapax
│   ├── features/         # BoW, TF-IDF, bigramas, selección de características
│   ├── embeddings/       # Word2Vec, doc embeddings
│   ├── modeling/         # Clasificación (SVM, Regresión Logística)
│   ├── clustering/       # K-means, análisis de clústeres
│   ├── topics/           # LSA con 50 tópicos
│   └── visualization/    # Gráficas y utilidades
│
│── notebooks/            # Experimentos y prototipos rápidos
│
│── reports/
│   ├── figures/          # Imágenes generadas
│   └── Tarea1_Reporte.pdf
│
│── tests/                # Unit tests para funciones clave
│
│── requirements.txt      # Dependencias
│── README.md             # Documentación del proyecto
│── main.py               # Script principal que ejecuta el pipeline completo

```
## 🛠️ Entorno reproducible

Para correr este proyecto localmente, usar el entorno `conda` definido en `environment.yml`.

### 🔁 Crear entorno desde cero

```bash
conda env create -f environment.yml
conda activate tarea1-nlp
```


## 🧪 Ejecutar el pipeline por partes

1. Analizar Ley de Zipf sin stopwords y con top_n=100:
```bash
python main.py --zipf --sin_stopwords --top_n 100
```

2. Generar palabras frecuentes y 4-gramas POS:
```bash
python main.py --frecuentes --pos
```
3. Entrenar Word2Vec y correr analogías:
```bash
python main.py --word2vec
```
4. Crear BoW y TF-IDF con bigramas y min_df=10:
```bash
python main.py --bow --min_df 10 --ngram_max 2
```

5. Ejecutar solo LSA:
```bash
python main.py --lsa
```