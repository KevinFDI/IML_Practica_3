
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from AA_utils import print_classification_report, visualizar_confusion_matrix
from AA_utils import elegir_imagen_al_azar, visualizar_parametros_para_imagenes


# CARGA DE DATOS

X = np.load('datasets/MNIST.npy')
y = np.load('datasets/MNIST_labels.npy')

print(f'''
      ====================================================================================================
      El problema de clasificación es de {X.shape[1]} features ---(prediciendo)---> {len(set(y))} clases
      ====================================================================================================
      ''')


# PREPROCESADO

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

NORMALIZAR = True

if (NORMALIZAR):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


# MODELADO

modelo = LogisticRegression(solver='saga', tol=0.1)  # Saga es más rápido para dataset muy grandes
modelo.fit(X_train, y_train)


# EVALUACION

y_pred = modelo.predict(X_test)

print_classification_report(y_train, modelo.predict(X_train))
print_classification_report(y_test, modelo.predict(X_test))

visualizar_confusion_matrix(y_test, y_pred)

imagen, indice = elegir_imagen_al_azar(X_test, pixeles_ancho=28, pixeles_alto=28)
print(f"Elemento aleatorio del Test Set:\nEl modelo predice: {modelo.predict(imagen)[0]}")
print(f"El valor real es: {y_pred[indice]}")

visualizar_parametros_para_imagenes(modelo, cantidad_de_clases=len(set(y)), pixeles_ancho=28, pixeles_alto=28)
