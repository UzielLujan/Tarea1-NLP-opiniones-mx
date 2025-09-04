# Modulo de Preprocesamiento para el corpus de texto

import pandas as pd
import re 
import os

def reemplazar_mojibake_comun(texto):
    reemplazos = {
        'Ã²': 'ó',
        'Ã³': 'ó',
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã­': 'í',
        'Ãº': 'ú',
        'Ã±': 'ñ',
        'Ã': 'Á',
        'Ã‰': 'É',
        'Ã': 'Í',
        'Ã“': 'Ó',
        'Ãš': 'Ú',
        'Ã‘': 'Ñ',
        'maiorÃ': 'mayor',
        'Ãl': 'Él',
        'mayorÃ': 'mayoría',
        'Alimentosâ¦': 'Alimentos...',
        '»': '',
        '': '',
        'Â': ''
        # Agrega más según lo que observes
    }
    for malo, bueno in reemplazos.items():
        texto = texto.replace(malo, bueno)
    return texto

def fix_mojibake(texto):
    # Intenta decodificar con latin1->utf-8, si falla regresa el texto original
    try:
        return texto.encode('latin1').decode('utf-8')
    except:
        return texto

def limpiar_mojibake_ftfy(texto):
    try:
        import ftfy
        return ftfy.fix_text(texto)
    except ImportError:
        print("ftfy no está instalado. Usando limpieza manual.")
        return limpiar_mojibake_manual(texto)


def normalizar(texto):
    # Eliminar puntos, comas y signos de puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    # Eliminar digitos
    texto = re.sub(r'\d+', '', texto)
    # Convertir a minúsculas
    texto = texto.lower()
    return texto.strip()

def limpiar_corpus(texto, metodo='ftfy'):
    # Reemplazar saltos de línea por espacios
    texto = texto.replace('\n', ' ')
    # Eliminar múltiples espacios
    texto = re.sub(r'\s+', ' ', texto)
    # Elimina patrones de reseñas truncadas tipo "...Más"
    texto = re.sub(r'\.{2,}\s*Más\b.*', '', texto, flags=re.IGNORECASE)
    # Limpiar mojibake usando el método seleccionado
    if metodo == 'ftfy':
        texto = limpiar_mojibake_ftfy(texto)
    else:
        texto = reemplazar_mojibake_comun(texto)
        texto = fix_mojibake(texto)

    return texto

def leer_corpus(ruta, archivo, metodo='ftfy'):
    data_path = os.path.join(ruta, "data", "raw", archivo)
    print('\n',f'Cargando corpus desde: {data_path}...')
    corpus = pd.read_csv(data_path, encoding="utf-8")
    print('Corpus cargado. Realizando limpieza...')
    # Limpiar el texto en la columna 'Review'
    corpus['Review'] = corpus['Review'].apply(limpiar_corpus, metodo=metodo)
    corpus['Polarity'] = corpus['Polarity'].astype(int)
    return corpus