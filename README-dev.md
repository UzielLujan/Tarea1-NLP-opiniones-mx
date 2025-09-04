# 🧠 Contexto: Tarea 1 - NLP (Maestría en Cómputo Científico)

Este documento resume el estado actual del proyecto para la **Tarea 1 del curso Análisis de Texto e Imágenes con Deep Learning**.

---

## 📦 Estructura modular

- `src/` organizado por módulos:
  - `preprocessing/limpieza.py`: limpieza configurable por parámetros
  - `analysis/`: Zipf, estadísticas, palabras frecuentes, POS
  - `representaciones/`: BoW, TF-IDF, selección de características
  - `embeddings/`: Word2Vec, embeddings promedio
  - `clustering/`: KMeans con análisis de centroides
  - `modeling/`: clasificación supervisada
  - `topics/`: LSA y términos por tópico
  - `visualization/`: distribución de clases, nubes de palabras, gráficas de barras
- `main.py` usa `argparse` para ejecutar cada bloque por separado
- `data/` y `reports/` organizados para guardar salidas (.csv, .png, .npy, etc.)
- `environment.yml` especifica dependencias exactas (Python 3.10, etc)

---

## 🧪 Entorno

- Librerías clave:
  - `nltk`, `spacy`, `scikit-learn`, `gensim`, `matplotlib`, `seaborn`, `wordcloud`
- Modelo `es_core_news_sm` instalado correctamente
- Conflictos resueltos:
  - OpenMP (`libiomp5md.dll`) usando `KMP_DUPLICATE_LIB_OK=TRUE`
  - `dtype` conflict entre `numpy` y `scikit-learn`

---

## 🎨 Visualización

- Estilo consistente con el notebook de EDA:
  - Paleta `"Reds"`
  - Barras con etiquetas de conteo
  - Nubes de palabras con `WordCloud`
  - Gráficas por clase (`Polarity`, `Type`, `Town`)

---

## 🛠️ Comandos útiles con argparse

```bash
python main.py --zipf --sin_stopwords --top_n 100
python main.py --frecuentes --pos
python main.py --bow --ngram_max 2 --min_df 10
python main.py --word2vec --cluster
python main.py --clasificacion
python main.py --lsa
```

---

## 🔧 Posibles mejoras y pendientes

- Crear `tests/` con `pytest` para validación modular
- Agregar visualizaciones como `plot_top_n_categorica()`
- Integrar todo en reporte LaTeX

---

Este documento sirve como resumen técnico para continuar, compartir o migrar el proyecto sin perder el contexto original.