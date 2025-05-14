import numpy as np
import random
from unidecode import unidecode

#Diccionario 
dictionary = {
    "hola": [-0.2, -0.3, -0.5, 0.1, -0.7],
    "adios": [0.4, -0.6, 0.2, -0.3, 0.1],
    "gracias": [-0.1, 0.4, -0.3, 0.5, -0.2],
    "como": [-0.3, 0.2, 0.4, -0.2, -0.5],
    "estas": [0.5, -0.1, 0.3, -0.6, 0.2],
    "bien": [0.6, 0.3, -0.1, -0.5, 0.4],
    "mal": [-0.2, -0.4, 0.2, 0.5, -0.3],
    "que": [0.3, 0.1, -0.2, 0.4, -0.6],
    "haces": [-0.5, 0.2, 0.3, -0.4, 0.1],
    "trabajo": [0.5, 0.4, -0.3, 0.2, -0.5],
    "estudio": [-0.3, -0.5, 0.1, 0.3, 0.6],
    "donde": [0.4, -0.1, -0.5, 0.2, 0.1],
    "mucho": [0.6, -0.3, 0.4, 0.1, -0.2],
    "bye": [-0.4, 0.1, -0.3, 0.5, 0.3]
}

entrenamiento = [
    ("hola como estas", "bien"),
    ("que haces",       "trabajo"),
    ("adios",           "bye"),
    ("hola",            "hola")
]

# Longitud de embedding y tamaño de la capa de salida
LonEmb = 5  
SalidaDim = 5  

lr = 0.1
epocas = 1000

# Pesos: capa densa 5×5
pesos = np.random.uniform(low=-1.0, high=1.0, size=(5, 5))
bias = np.random.uniform(low=-1.0, high=1.0, size=5)

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

def predecir_palabra(salida_vector, diccionario):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # Calcular scores (producto punto) entre salida y cada palabra
    palabras = list(diccionario.keys())
    vectores = np.array([diccionario[p] for p in palabras])
    scores = vectores @ salida_vector  # producto punto

    # Aplicar softmax para convertir a probabilidades
    probabilidades = softmax(scores)

    # Obtener la palabra con mayor probabilidad
    indice_max = np.argmax(probabilidades)
    palabra_max = palabras[indice_max]
    porcentaje = probabilidades[indice_max] * 100

    return palabra_max, round(porcentaje, 2)

#Entrenar
for _ in range(epocas):
    # Se selecciona un par de entrenamiento aleatorio
    entrada_texto, salida_deseada = random.choice(entrenamiento)
    # Se calcula el embedding de la entrada
    entrada_emb = embeding(entrada_texto)
    # Se combina el embedding para obtener un único vector 1x5
    vector = np.mean(entrada_emb, axis=0)
    # Se calcula la salida generada
    salida = np.dot(pesos, vector) + bias
    error = salida - dictionary[salida_deseada]
    # Gradientes
    grad_pesos = np.outer(error, vector)
    grad_bias = error
    # Actualización de pesos y bias
    pesos -= lr * grad_pesos
    bias -= lr * grad_bias

# Generación: 3×5 → 1×5 → aplicación de pesos/bias → 1×5
def generar(entrada_emb):
    # 1) Combina los n_tokens para obtener un solo vector 1×5
    vector = np.mean(entrada_emb, axis=0)
    salida = np.dot(pesos, vector) + bias
    return np.round(salida, 3)

# Modo interactivo para el usuario
def main():
    while True:
        user_in = input("Ingrese un texto (o 'salir' para terminar): ")
        if user_in.strip().lower() == 'salir':
            print("Chat finalizado. ¡Hasta luego!")
            break
        emb_user = embeding(user_in)
        if emb_user.size == 0:
            print("No se encontraron tokens conocidos en el diccionario. Intente con otras palabras.")
            continue
        salida_vec = generar(emb_user)
        palabra, pct = predecir_palabra(salida_vec, dictionary)
        print(f"Respuesta: '{palabra}' con probabilidad {pct}%\n")

if __name__ == "__main__":
    main()
