Redes Neuronales:

-Unidad más simple: Un perceptrón.
Un perceptrón es un tipo de neurona artificial que imita el funcionamiento de una neurona biológica. Es la unidad más simple de una red neuronal.
Estructura de un Perceptrón:

    Entradas (x1,x2,…,xnx1​,x2​,…,xn​): Son las características o atributos del conjunto de datos.
    Pesos (w1,w2,…,wnw1​,w2​,…,wn​): Cada entrada tiene un peso asociado que se ajusta durante el proceso de entrenamiento para minimizar el error de predicción.
    Sesgo (bb): Es un valor constante que permite ajustar la función de decisión.
    Función de activación: Combina las entradas ponderadas y produce la salida. En el perceptrón clásico, se utiliza una función escalón, donde la salida es 1 si la combinación lineal de entradas y pesos es mayor que un umbral, y 0 en caso contrario.

La salida del perceptrón se calcula como:
Salida=Funcioˊn de activacioˊn(∑i=1nwixi+b)
Salida=Funcioˊn de activacioˊn(i=1∑n​wi​xi​+b)

Limitaciones del Perceptrón:

    Un perceptrón solo puede resolver problemas linealmente separables. No puede manejar problemas más complejos donde las clases no se pueden separar con una línea recta.
    Para superar estas limitaciones, se introdujeron redes neuronales multicapa (MLP), que son la base del Deep Learning.

-Perceptrón Multicapa (MLP) y Deep Learning

El Perceptrón Multicapa (MLP) es una extensión del perceptrón simple y forma la base de las redes neuronales profundas utilizadas en Deep Learning:

    Capas ocultas: Un MLP tiene múltiples capas de neuronas entre la capa de entrada y la capa de salida, llamadas capas ocultas. Cada capa oculta permite al modelo aprender características más complejas y no lineales de los datos.
    Funciones de activación no lineales: Se utilizan funciones como ReLU, sigmoid, o tanh en lugar de la función escalón para permitir que el modelo aprenda representaciones no lineales.
    Deep Learning: Se refiere al uso de redes neuronales con muchas capas ocultas (redes profundas) para resolver problemas complejos como el reconocimiento de imágenes, el procesamiento del lenguaje natural, y mucho más.