import random

# Parámetros de las 2 neuronas
peso1 = 0.5
sesgo1 = 0.5
peso2 = -0.3
sesgo2 = 0.2

tasa_aprendizaje = 0.01

# Predicción combinada de las 2 neuronas
def predecir(edad):
    salida1 = (peso1 * edad + sesgo1)
    salida2 = (peso2 * edad + sesgo2)
    # Votación por mayoría (0, 1 o 2 votos “mayor”)
    votos_mayor = salida1 + salida2
    if votos_mayor >= 1:
        return 1  # mayor
    else:
        return 0  # menor

# Ajusta un peso y un sesgo según el error
def ajustar(peso_actual, sesgo_actual, edad, prediccion):
    # valor correcto
    if edad >= 18:
        correcto = 1
    else:
        correcto = 0
    error = correcto - prediccion
    nuevo_peso = peso_actual + (tasa_aprendizaje * error * edad)
    nuevo_sesgo = sesgo_actual + (tasa_aprendizaje * error)
    return nuevo_peso, nuevo_sesgo

# Entrenamiento
for _ in range(2000):
    edad = random.randint(2, 70)
    pred = predecir(edad)
    # Actualizamos ambas neuronas con la misma señal de error
    peso1, sesgo1 = ajustar(peso1, sesgo1, edad, pred)
    peso2, sesgo2 = ajustar(peso2, sesgo2, edad, pred)

# Evaluación
print("Evaluación con edades no vistas:")
for edad in [2, 9, 15, 17, 18, 19, 25, 50, 77, 99]:
    sal = predecir(edad)
    raw1 = peso1 * edad + sesgo1
    raw2 = peso2 * edad + sesgo2
    print(f"Edad {edad}: {'mayor' if sal else 'menor'} "
          f"(n1={round(raw1,2)}, n2={round(raw2,2)})")

print(f"\nPeso1: {peso1:.2f}, Sesgo1: {sesgo1:.2f}")
print(f"Peso2: {peso2:.2f}, Sesgo2: {sesgo2:.2f}")
