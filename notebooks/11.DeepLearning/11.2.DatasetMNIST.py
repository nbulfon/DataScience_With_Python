# El dataset del MNIST
## es el dataset que siempre se usa como ejemplo para enseñar el
## DeepLearning.
import tensorflow as _tensorFlow
import numpy as np
from skimage import io
from IPython.display import display, Math

# Cargar el dataset MNIST usando la API de TensorFlow 2.x
mnist = _tensorFlow.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

# Mostrar una imagen de entrenamiento
im_temp = train_images[0]
io.imshow(im_temp)

# Mostrar la primera etiqueta
print("Etiqueta de la primera imagen:", train_labels[0])

# Una red neuronal con TensorFlow - v2
# 28 x 28 = 784 --> es el array aplanado de la matriz con los pixeles de la img.
# Las imágenes de entrenamiento de MNIST viven en un espacio vectorial de dimensión 784.
# El dataset se puede pensar como 55000 filas y 784 columnas.
# Cada dato del dataset es un número real entre 0 y 1 (indica el grado de opacidad de la imagen).

dim_input = 784  # 28x28 imágenes de entrada aplanadas
n_categories = 10  # 10 dígitos (0-9)

# Construir el modelo utilizando `tf.keras`
# Utilizo la regresión SoftMax.
model = _tensorFlow.keras.models.Sequential([
    _tensorFlow.keras.layers.Flatten(input_shape=(28, 28)),  # Aplana las imágenes de 28x28 a un vector de 784 elementos
    _tensorFlow.keras.layers.Dense(n_categories, activation=_tensorFlow.nn.softmax)  # Capa densa con softmax para clasificación
])

# Compilar el modelo con la función de pérdida y el optimizador
model.compile(optimizer=_tensorFlow.keras.optimizers.SGD(learning_rate=0.5),  # Optimizador SGD con tasa de aprendizaje 0.5
              loss='categorical_crossentropy',  # Función de pérdida categorical crossentropy
              metrics=['accuracy'])  # Métrica de precisión

# Mostrar la fórmula de la entropía cruzada
display(Math(r"H_{y}(\hat{y}) = -\sum_{i} y_i log(\hat{y_i})"))

# Entrenar el modelo
model.fit(train_images, _tensorFlow.keras.utils.to_categorical(train_labels), epochs=10, batch_size=150)

# Evaluar la red neuronal en el conjunto de test
test_loss, test_acc = model.evaluate(
    test_images,
    _tensorFlow.keras.utils.to_categorical(test_labels))

print("Eficacia de la red neuronal: {:.2f}%".format(test_acc * 100))





