import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from AA_utils import visualizar_ajuste_por_regresion, if_dataframe_to_numpy, visualizar_curva_error_entrenamiento


# CARGA DE DATOS

data = pd.read_csv('datasets/consumo_nafta.csv')

# PREPROCESADO

X = data.drop(columns=['km/Litro'])
y = data['km/Litro']

print(f'''
      ====================================================================================================
      El problema de regresión es de {X.shape[1]} features ---(prediciendo)---> {y.shape[1] if len(y.shape) > 1 else 1} feature
      ====================================================================================================
      ''')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NORMALIZAR = False

if NORMALIZAR:
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = if_dataframe_to_numpy(y_train).reshape(-1, 1)
    y_test = if_dataframe_to_numpy(y_test).reshape(-1, 1)

    scaler = StandardScaler().fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)


# MODELADO

modelo = MLPRegressor(hidden_layer_sizes=(16, 5),
                      max_iter=2000,
                      tol=1e-5,
                      verbose=True,
                      activation='tanh',
                      learning_rate_init=0.005,
                      solver='adam')
modelo.fit(X_train, y_train)


# EVALUACION

y_train_predict = modelo.predict(X_train)
y_test_predict = modelo.predict(X_test)

visualizar_ajuste_por_regresion(modelo, X_train, X_test, y_train, y_test,
                                feature_name=X.columns[0], target_name=y.name)

visualizar_curva_error_entrenamiento(modelo)

print(f'''
      train Error cuadrático medio: {mean_squared_error(y_train_predict, y_train)}
      test Error cuadrático medio: {mean_squared_error(y_test_predict, y_test)}

      train Error abs. medio: {mean_absolute_error(y_train_predict, y_train)}
      test Error abs. medio: {mean_absolute_error(y_test_predict, y_test)}

      train r2_score: {r2_score(y_train_predict, y_train)}
      test r2_score: {r2_score(y_test_predict, y_test)}
      ''')
