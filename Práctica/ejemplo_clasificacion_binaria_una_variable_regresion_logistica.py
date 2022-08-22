import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from AA_utils import visualizar_confusion_matrix, visualizar_curva_ROC, visualizar_curva_logistica
from AA_utils import visualizar_parametros, print_classification_report

# CARGA DE DATOS

data = pd.read_csv('datasets/breast_cancer_radius.csv')


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

visualizar_curva_ROC(modelo, X, y)

visualizar_parametros(modelo, mostrar_bias=True, feature_names=X.columns, target_name=y.name)
visualizar_curva_logistica(modelo, X_train, y_train, feature_name=X.columns[0])
