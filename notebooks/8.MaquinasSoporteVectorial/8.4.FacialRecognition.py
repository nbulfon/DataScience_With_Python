# Reconocimiento facial

# imports
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as _plt

# obtengo las caras ->
faces = fetch_lfw_people(min_faces_per_person=60)

print(faces.target_names)

print(faces.images.shape)

fig, ax = _plt.subplots(5,5, figsize=(16,9))
for i, ax_i in enumerate(ax.flat):
    # dibujo la imagen en blanco y negro
    ax_i.imshow(faces.images[i], cmap="bone")
    # incorporo el tag con el nombre que corresponda
    ax_i.set(xticks=[], yticks=[],xlabel=faces.target_names[faces.target[i]])

# ANÁLISIS DE COMPONENTES PRINCIPALES
# (para reducir el espacio vectorial, para que nuestro SVM clasifique las imagenes
# más rapido...(lo hago porque es un curso...))
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

pca = PCA(n_components=150, whiten=True, random_state=42)
# defino el support vector clasifier, con un kernel de Radial Basic Function (rbf)
# tb le paso un class_weight = balanced, para que el clasificador
# pondere las var. "más importantes" (VER ESO)
svc = SVC(kernel="rbf", class_weight="balanced")
model = make_pipeline(pca, svc)


from sklearn.model_selection import train_test_split


# defino los conjuntos de entrenamiento y conjuntos de testeo/prueba ->
Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target, random_state = 42)

from sklearn.model_selection import GridSearchCV
# defino los parámetros de la Radial Basic Function, para construir el pasillo/parrilla ->
param_grid = {
    "svc__C":[0.1,1,5,10,50],
    "svc__gamma":[0.0001, 0.0005, 0.001, 0.005, 0.01]
}
grid = GridSearchCV(model, param_grid)

#print(grid.best_params_)
# Ajustar GridSearchCV a los datos
grid.fit(Xtrain, Ytrain)

classifier = grid.best_estimator_
yfit = classifier.predict(Xtest)

fig, ax = _plt.subplots(8,6,figsize=(16,9))

for i, ax_i in enumerate(ax.flat):
    ax_i.imshow(Xtest[i].reshape(62,47), cmap="bone")
    ax_i.set(xticks=[], yticks=[])
    ax_i.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color = "black" if yfit[i]==Ytest[i] else "red")

fig.suptitle("Predicciones de las imágenes (incorrectas en rojo)", size = 15)

# REPORTES
# Hago un reporte de mi modelo. Para ver los estadísticos y valores más importantes.
# Precison, promedio de aciertos, errores, soporte, etc.
from sklearn.metrics import classification_report

print(classification_report(Ytest, yfit, target_names = faces.target_names))

# Represento las métricas que arroja el reporte en una matriz de confusión.
# Mediante un mapa de calor.
from sklearn.metrics import confusion_matrix
# armo la matriz de confusion
mat = confusion_matrix(Ytest, yfit)

import seaborn as sns; sns.set()
# imprimo el mapa de calor
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True, 
            xticklabels=faces.target_names, yticklabels=faces.target_names )
# FIN REPORTES























