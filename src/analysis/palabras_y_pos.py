# src/analysis/palabras_y_pos.py

import pandas as pd
import nltk
from collections import Counter
import spacy
from nltk.corpus import stopwords
from itertools import islice
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

nlp = spacy.load("es_core_news_sm")
stopwords_es = set(stopwords.words('spanish'))

def palabras_frecuentes_por_clase(df: pd.DataFrame, columna_texto="Review", columna_clase="Polarity", top_n=15):
    print("\n📝 Palabras más frecuentes por clase (sin stopwords):")

    clases = df[columna_clase].unique()
    resultados = {}
    output_dir = os.path.join("reports", "features")
    os.makedirs(output_dir, exist_ok=True)

    for clase in sorted(clases):
        subset = df[df[columna_clase] == clase]
        tokens = subset[columna_texto].apply(lambda x: nltk.word_tokenize(str(x), language='spanish')).explode()
        tokens_filtrados = tokens[~tokens.str.lower().isin(stopwords_es)]
        mas_comunes = Counter(tokens_filtrados).most_common(top_n)
        resultados[clase] = mas_comunes
        print(f"\nClase {clase}:")
        for palabra, freq in mas_comunes:
            print(f"- {palabra}: {freq}")

        # Guardar CSV por clase
        df_palabras = pd.DataFrame(mas_comunes, columns=["Palabra", "Frecuencia"])
        df_palabras.to_csv(os.path.join(output_dir, f"palabras_frecuentes_clase_{clase}.csv"), index=False)

    return resultados

def pos_4gramas_por_clase(df: pd.DataFrame, columna_texto="Review", columna_clase="Polarity", top_n=10):
    print("\n🔤 4-gramas gramaticales por clase:")

    def obtener_4gramas_pos(texto):
        doc = nlp(str(texto))
        tags = [token.pos_ for token in doc]
        return zip(*(islice(tags, i, None) for i in range(4)))

    clases = df[columna_clase].unique()
    resultados = {}
    output_dir = os.path.join("reports", "features")
    os.makedirs(output_dir, exist_ok=True)

    for clase in sorted(clases):
        subset = df[df[columna_clase] == clase]
        ngramas = subset[columna_texto].apply(obtener_4gramas_pos).explode()
        conteo = Counter(ngramas)
        mas_comunes = conteo.most_common(top_n)
        resultados[clase] = mas_comunes

        print(f"\nClase {clase}:")
        for ngrama, freq in mas_comunes:
            tag_str = "-".join(ngrama)
            print(f"- {tag_str}: {freq}")

        # Guardar CSV por clase
        df_ngrams = pd.DataFrame([("-".join(ng), freq) for ng, freq in mas_comunes], columns=["4gram_POS", "Frecuencia"])
        df_ngrams.to_csv(os.path.join(output_dir, f"4gramas_POS_clase_{clase}.csv"), index=False)

    return resultados
