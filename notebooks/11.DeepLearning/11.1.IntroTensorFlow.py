# Introduccion a Tensor Flow

import tensorflow as _tensorFlow

x1 = _tensorFlow.constant([1,2,3,4,5])
x2 = _tensorFlow.constant([6,7,8,9,10])

res = _tensorFlow.multiply(x1, x2)
print(res)

# Suponiendo que 'res' es un tensor o una operación de TensorFlow:
print(res.numpy())  # Esto imprime el resultado directamente

# O simplemente:
# Si 'res' es un tensor, esto imprimirá su valor en modo Eager Execution
print(res)

# Asumiendo que 'res' es un tensor o una operación de TensorFlow:
output = res.numpy()  # Si 'res' es un tensor, esto obtiene su valor como un array de NumPy
print(output)

# Para las configuraciones, puedes usar `tf.config` en lugar de `tf.ConfigProto`:

# Configurar `log_device_placement`
_tensorFlow.debugging.set_log_device_placement(True)

# `allow_soft_placement` se maneja automáticamente en TensorFlow 2.x,
# así que generalmente no necesitas configurarlo manualmente.


## Aprendizaje neuronal de las señales de transito --

import os
import skimage.io
import numpy as _numpy

## Análisis exploratorio de los datos
# (Data Cleaning y Data Wrangling)

# funcion para cargar los datos.
def load_ml_data(data_directory):
    dirs =[d for d in os.listdir(data_directory)
           if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]
        
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
            
    return images,labels

directory = "C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\belgian"
directory_train = "\\Training\\"
directory_test = "\\Testing\\"

images, labels = load_ml_data(directory + directory_train)
len(images)
images[0]

images = _numpy.array(images)
labels = _numpy.array(labels)
#images.ndim
images.size

len(set(labels))
images.flags

import matplotlib.pyplot as _plt
_plt.hist(labels, len(set(labels)))
_plt.show()

import random

# creo imagenes de las señales de transito ->
rand_signs = random.sample(population=range(0,len(labels)), k=6)
rand_signs
for i in range(len(rand_signs)):
    # defino una imagen temporal ->
    temp_im = images[rand_signs[i]]
    
    _plt.subplot(1,6,i+1)
    # hago un off para evitar los ejes de coordenadas.
    _plt.axis("off")
    _plt.imshow(temp_im)
    _plt.subplots_adjust(wspace=0.5)
    _plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(
        temp_im.shape,
        temp_im.min(),
        temp_im.max()))

# primera complicación - las imagenes tienen distinto tamaño.
# (esto puede joder, despues a la hora de compilar con la Red Neuronal)
unique_labels = set(labels)
# defino un tamaño predeterminado ->
_plt.figure(figsize=(16,9))
i = 1

# armo un bucle para imprimir las imagenes todas juntas,
# y que me diga, cuantas hay, de cada Clase. (hago un groupby Clase...).
for label in unique_labels:
    # labels.index --> devuelve la posición que ocupa la etiqueta.
    temp_im = images[labels.index(label)]
    _plt.subplot(8,8,i)
    # hago un off para evitar los ejes de coordenadas.
    _plt.axis("off")
    _plt.title("Clase {0} ({1})".format(
        label, list(labels).count(label))
        )
    i +=1
    _plt.imshow(temp_im)
_plt.show()

type(labels)

## FIN Análisis exploratorio de los datos

## Pre-procesado de las imagenes con las que voy a trabajar.
## Cuestiones a considerar:
#  Las imágenes no todas son del mismo tamaño
#  Hay 62 clases de imágenes (desde la 0 hasta la 61)
#  La distribución de señales de tráfico no es uniforme (algunas salen más veces que otras)

# importante aclarar. Tener una regla de decisión para clasificar imagenes, basada en el
# color, es poco practico, porque te influye mucho la iluminación ,la calidad de la imagen, etc.
from skimage import transform

# me intento quedar con la anchura y la altura mas pequeñas ->
width = 9999
height = 9999

for image in images:
    if (image.shape[0] < height):
        height = image.shape[0]
    if (image.shape[1] < width):
        width = image.shape[1]
print("Tamaño mínimo:{0}x{1}".format(height,width))

# hago que todas las imagenes tengan el mismo tamaño ->
images30 = [transform.resize(image, (30,30)) for image in images]
images30[0]

# creo imagenes de las señales de transito ->
rand_signs = random.sample(population=range(0,len(labels)), k=6)
rand_signs
for i in range(len(rand_signs)):
    # defino una imagen temporal ->
    temp_im = images30[rand_signs[i]]
    
    _plt.subplot(1,6,i+1)
    # hago un off para evitar los ejes de coordenadas.
    _plt.axis("off")
    _plt.imshow(temp_im)
    _plt.subplots_adjust(wspace=0.5)
    _plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(
        temp_im.shape,
        temp_im.min(),
        temp_im.max()))

# seteo todas las imagenes en escala de grises ->
from skimage.color import rgb2gray
images30 = _numpy.array(images30)
images30 = rgb2gray(images30)

rand_signs = random.sample(population=range(0,len(labels)), k=6)
rand_signs
for i in range(len(rand_signs)):
    # defino una imagen temporal ->
    temp_im = images30[rand_signs[i]]
    
    _plt.subplot(1,6,i+1)
    # hago un off para evitar los ejes de coordenadas.
    _plt.axis("off")
    _plt.imshow(temp_im, cmap="gray")
    _plt.subplots_adjust(wspace=0.5)
    _plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(
        temp_im.shape,
        temp_im.min(),
        temp_im.max()))

## Modelo de Red Neuronal con TensorFlow

# Definición de las entradas usando tf.keras.Input (común en modelos Keras)
x = _tensorFlow.keras.Input(shape=[30, 30], dtype=_tensorFlow.float32)
y = _tensorFlow.keras.Input(shape=(1,), dtype=_tensorFlow.int32)

# Aplanar las imágenes usando tf.keras.layers.Flatten
images_flat = _tensorFlow.keras.layers.Flatten()(x)

# Crear una capa totalmente conectada con 62 neuronas y activación ReLU.
# (Hago uso de la Red Neuronal, para realizar una Regresión Logística)
regLogistica = _tensorFlow.keras.layers.Dense(
    62,
    activation=_tensorFlow.nn.relu)(images_flat)

# Crear el modelo de Keras
model = _tensorFlow.keras.Model(inputs=x, outputs=regLogistica)

optimizadorAdam = _tensorFlow.keras.optimizers.Adam(learning_rate=0.001)
funcionPerdida = _tensorFlow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Compilar el modelo con la función de pérdida y el optimizador Adam
# SparseCategoricalCrossentropy es equivalente a sparse_softmax_cross_entropy_with_logits
model.compile(optimizer=optimizadorAdam,
              loss=funcionPerdida,
              metrics=['accuracy'])

# Entrenamiento del modelo 
# Establecer la semilla aleatoria
_tensorFlow.random.set_seed(1234)

# Suponiendo que ya tienes definido el modelo
# Aquí deberías haber definido tu modelo usando Keras (como en la respuesta anterior)

# Compila el modelo de nuevo para asegurarte de que todo está configurado
model.compile(optimizer=_tensorFlow.keras.optimizers.Adam(learning_rate=0.001),
              loss=_tensorFlow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenamiento manual en lugar de utilizar sess.run()
# el numero 3000 es dicrecional. A medida que le de más, más se minimizará la pérdida,
# es decir, más preciso será el modelo, dicho de otro modo: Mejor entrenado estará.
for i in range(3000):
    # Entrenar el modelo manualmente en un lote (batch) con tf.GradientTape
    with _tensorFlow.GradientTape() as tape:
        logits = model(images30, training=True)  # `images30` es tu conjunto de datos de entrada
        
        loss_value = _tensorFlow.reduce_mean(
            _tensorFlow.nn.sparse_softmax_cross_entropy_with_logits(
                labels=list(labels), logits=logits))
        
        accuracy_val = _tensorFlow.reduce_mean(
            _tensorFlow.metrics.sparse_categorical_accuracy(
                list(labels), logits))

    # Calcular y aplicar los gradientes
    gradients = tape.gradient(loss_value, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Imprimir los resultados en intervalos de 50 épocas
    if i % 50 == 0:
        print("EPOCH", i)
        print("Eficacia: ", accuracy_val.numpy())
        print("Pérdidas:", loss_value.numpy())

### Evaluación de la red neuronal

# Seleccionar una muestra aleatoria de 40 imágenes
sample_idx = random.sample(range(len(images30)), 40)
sample_images = [images30[i] for i in sample_idx]
sample_labels = [labels[i] for i in sample_idx]

# Realizar predicciones directamente usando el modelo
# Si `final_pred` es la predicción final, en TensorFlow 2.x simplemente llama al modelo
prediction = model.predict(_tensorFlow.convert_to_tensor(sample_images))

# Convertir las predicciones a las clases finales
final_pred = _tensorFlow.argmax(prediction, axis=1)

# Visualizar los resultados
_plt.figure(figsize=(16,20))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    predi = final_pred[i].numpy()  # Obtener el valor del tensor
    _plt.subplot(10, 4, i+1)
    _plt.axis("off")
    color = "green" if truth == predi else "red"
    _plt.text(32, 15, "Real:         {0}\nPredicción: {1}".format(truth, predi),
             fontsize=14, color=color)
    _plt.imshow(sample_images[i], cmap="gray")
_plt.show()

# Cargar los datos de prueba
test_images, test_labels = load_ml_data(directory + directory_test)

# Redimensionar las imágenes a 30x30
test_images30 = [transform.resize(im, (30, 30)) for im in test_images]

# Convertir las imágenes a escala de grises
test_images30 = rgb2gray(_numpy.array(test_images30))

# Realizar predicciones directamente usando el modelo
prediction = model.predict(_tensorFlow.convert_to_tensor(test_images30))

# Convertir las predicciones a las clases finales
final_pred = _tensorFlow.argmax(prediction, axis=1).numpy()

# Contar las coincidencias entre predicciones y etiquetas reales
match_count = sum([int(l0 == lp) for l0, lp in zip(test_labels, final_pred)])

# Calcular la precisión
acc = match_count / len(test_labels) * 100
print("Eficacia de la red neuronal: {:.2f}%".format(acc))
### FIN Evaluación de la red neuronal

## FIN Modelo de Red Neuronal con TensorFlow







