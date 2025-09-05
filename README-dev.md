# 🧠 Contexto: Tarea 1 - NLP (Maestría en Cómputo Científico)

Este documento describe el pipeline actualizado y las decisiones de diseño para la **Tarea 1 del curso Análisis de Texto e Imágenes con Deep Learning**.

---

## 📦 Estructura modular y flujo de procesamiento

- **Preprocesamiento flexible:**  
  - `src/preprocessing/limpieza.py` ahora separa la lectura básica (`leer_corpus`) del procesamiento avanzado (`procesar_corpus`).
  - El corpus se lee una sola vez y se procesa solo con el nivel de limpieza requerido por cada módulo.
  - Los pasos de limpieza avanzados (normalización, stopwords, lematización/stemming) se controlan por parámetros.

- **Módulos independientes y eficientes:**  
  - Cada módulo (`analysis/`, `representaciones/`, `embeddings/`, `clustering/`, `modeling/`, `topics/`) recibe el corpus ya procesado según sus necesidades.
  - No hay doble procesamiento ni redundancias.
  - El pipeline es fácil de mantener y permite experimentar con variantes de limpieza sin modificar los módulos internos.

- **main.py como orquestador:**  
  - Usa `argparse` para ejecutar cada bloque por separado o el pipeline completo.
  - Solo procesa el corpus con el nivel de limpieza necesario para cada experimento.
  - Ejemplo de flujo:
    - Estadísticas: corpus solo con limpieza básica.
    - Zipf: corpus con o sin stopwords según argumento.
    - Palabras frecuentes/POS: corpus normalizado y sin stopwords.
    - BoW/TF-IDF/Word2Vec: corpus normalizado y sin stopwords.
    - Clasificación: variantes acumulativas de procesamiento (sin preprocesamiento, minúsculas, lematización, etc.).
    - LSA: usa la matriz TF-IDF generada.

- **Persistencia y salidas organizadas:**  
  - `data/` y `reports/` organizados para guardar salidas (.csv, .png, .npy, etc.).
  - Vectorizadores, embeddings y resultados de clustering/modelado se guardan para análisis posterior.

---

## 🧪 Entorno y dependencias

- **Librerías clave:**  
  - `nltk`, `spacy`, `scikit-learn`, `gensim`, `matplotlib`, `seaborn`, `wordcloud`
- **Modelo spaCy:**  
  - `es_core_news_sm` instalado y cargado solo una vez.
- **Compatibilidad:**  
  - Python 3.10, conflictos de dependencias resueltos.
  - OpenMP (`libiomp5md.dll`) y conflictos `dtype` entre `numpy` y `scikit-learn` documentados y resueltos.

---

## 🛠️ Ejecución y ejemplos de uso

```bash
python main.py --zipf --sin_stopwords --top_n 100
python main.py --frecuentes
python main.py --bow --ngram_max 2 --min_df 10
python main.py --word2vec --cluster
python main.py --clasificacion
python main.py --lsa
```

- Puedes combinar flags para ejecutar varios módulos en una sola corrida.
- El pipeline completo se ejecuta con `--pipeline_completo`.

---

## 🎨 Visualización

- Estilo consistente con el notebook de EDA:
  - Paleta `"Reds"`
  - Barras con etiquetas de conteo
  - Nubes de palabras con `WordCloud`
  - Gráficas por clase (`Polarity`, `Type`, `Town`)

---

## 🔍 Decisiones de diseño y mejores prácticas

- **Separación de lectura y procesamiento:**  
  Permite máxima flexibilidad y eficiencia, evitando recargas y reprocesamientos innecesarios.
- **Procesamiento parametrizable:**  
  Los pasos de limpieza se aplican solo cuando y como se necesitan.
- **Módulos desacoplados:**  
  Cada módulo asume que el corpus ya está en el estado requerido.
- **Facilidad para experimentación:**  
  Cambiar el nivel de limpieza para cualquier experimento es trivial y no requiere modificar los módulos internos.
- **Soporte para análisis avanzados:**  
  Filtrado de palabras custom en análisis de palabras frecuentes, experimentos acumulativos en clasificación, etc.

---

## 🧪 Pendientes y mejoras sugeridas

- Crear `tests/` con `pytest` para validación modular.
- Agregar visualizaciones adicionales (`plot_top_n_categorica`, etc.).
- Integrar todo en reporte LaTeX.
- Mejorar documentación de cada módulo y agregar ejemplos de uso.

---

Este documento sirve como referencia técnica para continuar, compartir o migrar el proyecto sin perder el contexto y las decisiones clave del pipeline.