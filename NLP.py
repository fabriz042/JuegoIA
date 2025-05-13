import numpy as np
import random
from unidecode import unidecode

#Diccionario 
dictionary = {
    "hola": [-0.2, -0.3, -0.5, 0.1, -0.7],
    "adios": [0.4, -0.6, 0.2, -0.3, 0.1],
    "gracias": [-0.1, 0.4, -0.3, 0.5, -0.2],
    "por": [0.3, -0.5, 0.1, 0.2, 0.4],
    "favor": [0.1, 0.2, 0.6, -0.4, -0.1],
    "como": [-0.3, 0.2, 0.4, -0.2, -0.5],
    "estas": [0.5, -0.1, 0.3, -0.6, 0.2],
    "bien": [0.6, 0.3, -0.1, -0.5, 0.4],
    "mal": [-0.2, -0.4, 0.2, 0.5, -0.3],
    "nombre": [-0.1, 0.5, -0.4, 0.3, 0.2],
    "me": [0.4, -0.3, 0.2, -0.5, 0.6],
    "llamo": [-0.2, 0.6, -0.5, 0.3, 0.1],
    "que": [0.3, 0.1, -0.2, 0.4, -0.6],
    "haces": [-0.5, 0.2, 0.3, -0.4, 0.1],
    "trabajo": [0.5, 0.4, -0.3, 0.2, -0.5],
    "estudio": [-0.3, -0.5, 0.1, 0.3, 0.6],
    "donde": [0.4, -0.1, -0.5, 0.2, 0.1],
    "vives": [-0.2, 0.3, 0.5, -0.6, 0.4],
    "mucho": [0.6, -0.3, 0.4, 0.1, -0.2],
    "gusta": [-0.4, 0.1, -0.3, 0.5, 0.3]
}

# Longitud de embedding y tamaño de la capa de salida
LonEmb = 5  
SalidaDim = 5  

# Pesos: capa densa 5×5
pesos = np.array([
    [ 0.1, -0.2,  0.3, -0.4,  0.5],
    [-0.7,  0.8, -0.9,  1.0, -1.1],
    [ 0.2, -0.1,  0.4, -0.3,  0.6],
    [-0.6,  0.7, -0.8,  0.9, -1.0],
    [ 0.5, -0.4,  0.3, -0.2,  0.1],
])  # cada fila = pesos de una de las 5 salidas

bias = np.array([0.1, 0.2, -0.1, 0.05, -0.05])


#Nomrmalizacion
def normalizar(texto):
    normalizado = unidecode(texto.lower())
    return normalizado

# Tokenización
def tokenizacion(texto):
    return texto.split()

# Embedding diccionario
def embeding(texto):
    texto_norm = normalizar(texto)
    tokens = tokenizacion(texto_norm)
    emb = []
    for token in tokens:
        if token in dictionary:
            emb.append(dictionary[token])
    return np.array(emb)  # (n_tokens, 5)

# Generación: 3×5 → 1×5 → aplicación de pesos/bias → 1×5
def generar(entrada_emb):
    # 1) Combina los n_tokens para obtener un solo vector 1×5
    vector = np.mean(entrada_emb, axis=0)  # forma (5,)

    # 2) Aplica la capa densa: (5×5) · (5,) + (5,) → (5,)
    salida = np.dot(pesos, vector) + bias
    return np.round(salida, 3)

# Ejecución de prueba
entrada_emb = embeding("Hola como estás")
print("Embedding de entrada (3×5):")
print(entrada_emb, "\n")

salida = generar(entrada_emb)
print("Salida generada (1×5):")
print(salida)
