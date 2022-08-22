import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from AA_utils import visualizar_confusion_matrix
from AA_utils import visualizar_frontera_de_desicion_2D, visualizar_funcion_transferencia_2D
from AA_utils import visualizar_parametros, print_classification_report

# CARGA DE DATOS

data = pd.read_csv('datasets/2D_6_clases_hard.csv')

# PREPROCESADO

X = data.drop(columns=['target'])
y = data['target']

print(f'''
      ====================================================================================================
      El problema de clasificación es de {X.shape[1]} features ---(prediciendo)---> {len(set(y))} clases
      ====================================================================================================
      ''')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NORMALIZAR = False

if NORMALIZAR:
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

# MODELADO

modelo = LogisticRegression()
modelo.fit(X_train, y_train)


# EVALUACION

print_classification_report(y_train, modelo.predict(X_train))
print_classification_report(y_test, modelo.predict(X_test))

visualizar_confusion_matrix(y_test, modelo.predict(X_test))

visualizar_parametros(modelo, mostrar_bias=True, feature_names=X.columns, target_name=y.name)
visualizar_frontera_de_desicion_2D(modelo, X, X_train, X_test, y_train, y_test)
visualizar_funcion_transferencia_2D(modelo, X, X_train, X_test, y_train, y_test)
