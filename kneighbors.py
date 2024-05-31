import numpy as np
import pandas as pd

train_data_path = 'data/vino_entrenamiento.csv'
test_data_path = 'data/vino_prueba.csv'


# Biblio
# https://steemit.com/spanish/@waster/explicacion-alternativa-para-accuracy-precision-recall-y-f1-score
# https://blogdatlas.wordpress.com/2023/07/16/cuales-son-los-tipos-de-distancias-en-analitica-de-datos-euclediana-manhattan-isocronas-y-mucho-mas-columna-de-investigacion-datlas/

def mostrar_resultados(resultados):
    """
    Función para mostrar los resultados de la evaluación de los algoritmos
    :param resultados:
    :return:
    """
    # Mostramos los resultados en una tabla
    mostrar_tabla(resultados)

    # Encontramos el modelo con el mejor valor de f1 score
    mejor_distancia, mejor_k = max(resultados.items(), key=lambda x: x[1][3])
    print(f"La mejor distancia/k es {mejor_distancia} para maximizar el f1 score, con un f1 score de {resu}")

    # Encontramos el modelo con el mejor valor de accuracy
    mejor_distancia, mejor_k = max(resultados.items(), key=lambda x: x[1][0])
    print(f"La mejor distancia/k es {mejor_distancia} para maximizar la accuracy")

    # Encontramos el modelo con el mejor valor de precision
    mejor_distancia, mejor_k = max(resultados.items(), key=lambda x: x[1][1])
    print(f"La mejor distancia/k es {mejor_distancia} para maximizar la precision")


def mostrar_tabla(resultados):
    # Imprimir el encabezado
    print("{:<15} {:<5} {:<10} {:<10} {:<10} {:<10}".format('Distancia', 'k', 'Accuracy', 'Precision', 'Recall',
                                                            'F1 Score'))
    for key, value in resultados.items():
        distance, k = key
        accuracy, precision, recall, f1 = value
        print(
            "{:<15} {:<5} {:<10} {:<10} {:<10} {:<10}".format(distance, k, accuracy, precision, recall, f1))


# Función para estandarizar datos
def estandarizar_datos(X):
    """
    La estandarización es un método de preprocesamiento que se utiliza para escalar las características (variables)
    de los datos de modo que tengan una media (promedio) de cero y una desviación estándar de uno.
    Esto es importante por:
    - Igualdad de Escala:
    - Mejora del Rendimiento del Algoritmo
    - Distancias más Significativas:
    :param X:
    :return:
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def calcular_accuracy(valores_reales, valores_prediccion):
    """
    Función para calcular la accuracy del algoritmo
    La cantidad de veces que acertaste una afirmación, sobre el total de datos de entrada
    :param valores_reales:
    :param valores_prediccion:
    :return:
    """
    return np.sum(valores_reales == valores_prediccion) / len(valores_reales)


def calcular_precision(valores_reales, valores_prediccion):
    """
    Función para calcular la precision del algoritmo
    Se compara la cantidad de casos clasificados como verdaderos positivos sobre todo lo que realmente era positivo
    :param valores_reales:
    :param valores_prediccion:
    :return:
    """
    true_positives = np.sum((valores_reales == valores_prediccion) & (valores_reales == 1))
    predicted_positives = np.sum(valores_prediccion == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0


def calcular_recall(valores_reales, valores_prediccion):
    """
    Función para calcular el recall del algoritmo
    Cantidad de casos clasificados como verdaderos positivos sobre todo lo que realmente era positivo
    :param valores_reales:
    :param valores_prediccion:
    :return:
    """
    true_positives = np.sum((valores_reales == valores_prediccion) & (valores_reales == 1))
    actual_positives = np.sum(valores_reales == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0


def calcular_f1_score(valores_reales, valores_prediccion):
    """
    Función para calcular el F1 Score del algoritmo
    El F1 Score es una medida de la precisión y el recall de un algoritmo.
    Es util si necesitamos mantenernos lejos de falsos positivos y falsos negativos
    :param valores_reales:
    :param valores_prediccion:
    :return:
    """
    precision = calcular_precision(valores_reales, valores_prediccion)
    recall = calcular_recall(valores_reales, valores_prediccion)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0


def calcular_metricas(valores_reales, valores_prediccion):
    """
    Función para calcular los valores de accuracy, precision, recall y f1 score
    :param valores_reales:
    :param valores_prediccion:
    :return:
    """
    accuracy = calcular_accuracy(valores_reales, valores_prediccion)
    precision = calcular_precision(valores_reales, valores_prediccion)
    recall = calcular_recall(valores_reales, valores_prediccion)
    f1 = calcular_f1_score(valores_reales, valores_prediccion)
    # Redondear a 2 decimales y convertir a porcentaje
    accuracy = f"{round(accuracy * 100, 2)}%"
    precision = f"{round(precision * 100, 2)}%"
    recall = f"{round(recall * 100, 2)}%"
    f1 = f"{round(f1 * 100, 2)}%"
    return accuracy, precision, recall, f1


def manual_label_encoder(labels):
    """
    Función para convertir los valores de clase manual a un formato de vectores
    Asignamos un valor a cada clase empezando por 0 e incrementando
    :param labels:
    :return:
    """
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])
    return int_labels, label_to_int


def k_nearest_neighbors_no_sklearn(datos, datos_nuevos, variables, target, k):
    """
    Función para calcular los k vecinos más cercanos de un vector de datos sin utilizar scikit-learn
    :param datos:
    :param datos_nuevos:
    :param variables:
    :param target:
    :param k:
    :return:
    """
    # Convertimos los valores de clase manual a un formato de vectores
    y_train, label_dict = manual_label_encoder(datos[target].values)

    # Extraer las características para los datos de entrenamiento y prueba
    X_train = datos[variables].values
    X_test = datos_nuevos[variables].values

    # Preparar la lista para almacenar las predicciones
    predictions = []

    # Calcular la distancia euclidiana entre cada punto de prueba y todos los puntos de entrenamiento
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))

        # Obtener los índices de los k vecinos más cercanos
        k_nearest_indices = np.argsort(distances)[:k]

        # Extraer las clases de los k vecinos más cercanos
        k_nearest_classes = y_train[k_nearest_indices]

        # Determinar la clase más frecuente entre los vecinos
        prediction = np.bincount(k_nearest_classes).argmax()
        predictions.append(prediction)

    # Decodificar las predicciones a etiquetas originales
    int_to_label = {v: k for k, v in label_dict.items()}
    predictions_labels = np.array([int_to_label[pred] for pred in predictions])

    return predictions_labels


def k_nearest_neighbors_benchmark(datos, datos_nuevos, variables, target, k, distance_type='euclidean',
                                  estandarizar=False):
    """
    Función para calcular los k vecinos más cercanos de un vector de datos sin utilizar scikit-learn
    Y con distancias de diferentes tipos (euclidiana, manhattan, chebyshev) y con diferentes valores de k
    :param datos:
    :param datos_nuevos:
    :param variables:
    :param target:
    :param k:
    :param distance_type:
    :return:
    """

    # Codificar la variable objetivo manualmente
    y_train, label_dict = manual_label_encoder(datos[target].values)

    # Estandarizar las características para los datos de entrenamiento y prueba

    if estandarizar:
        X_train = estandarizar_datos(datos[variables].values)
        X_test = estandarizar_datos(datos_nuevos[variables].values)
    else:
        X_train = datos[variables].values
        X_test = datos_nuevos[variables].values

    # Preparar la lista para almacenar las predicciones
    predictions = []

    # Calcular la distancia entre cada punto de prueba y todos los puntos de entrenamiento
    for test_point in X_test:
        if distance_type == 'euclidean':
            distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        elif distance_type == 'manhattan':
            distances = np.sum(np.abs(X_train - test_point), axis=1)
        elif distance_type == 'chebyshev':
            distances = np.max(np.abs(X_train - test_point), axis=1)
        else:
            print("Error: Distancia no valida")
            return

        # Obtener los índices de los k vecinos más cercanos
        k_nearest_indices = np.argsort(distances)[:k]

        # Extraer las clases de los k vecinos más cercanos
        k_nearest_classes = y_train[k_nearest_indices]

        # Determinar la clase más frecuente entre los vecinos
        prediction = np.bincount(k_nearest_classes).argmax()
        predictions.append(prediction)

    # Decodificar las predicciones a etiquetas originales
    int_to_label = {v: k for k, v in label_dict.items()}
    predictions_labels = np.array([int_to_label[pred] for pred in predictions])

    return predictions_labels


# Inicio del programa

# Cargar los datos de entrenamiento y de prueba
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Extraer las variables, excluyendo la columna de clase
variables = [col for col in train_data.columns if col != 'class']

# Ejecutar el algoritmo con k=5
k = 5
Y_pred_no_sklearn = k_nearest_neighbors_no_sklearn(train_data, test_data, variables, 'class', k)

# Extraer las etiquetas verdaderas de los datos de prueba
true_labels = test_data['class'].values

# Codificar las etiquetas verdaderas y predichas
_, true_label_dict = manual_label_encoder(train_data['class'].values)
encoded_true_labels = np.array([true_label_dict[label] for label in true_labels])
encoded_predictions = np.array([true_label_dict[label] for label in Y_pred_no_sklearn])

# Calcular los valores de accuracy, precision, recall y f1 score
accuracy, precision, recall, f1 = calcular_metricas(encoded_true_labels, encoded_predictions)

# Calcular la cantidad de predicciones incorrectas
incorrect_predictions = np.where(encoded_predictions != encoded_true_labels)[0]
num_incorrect_predictions = len(incorrect_predictions)

# Imprimir los valores de accuracy, precision, recall y f1 score, asi como el numero de predicciones incorrectas

print("Ejemplo de partida con k=5 y distancia euclidiana")

print(f"Number of incorrect predictions: {num_incorrect_predictions}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Comparacion usando diferentes valores de k, con datos estandarizados y diferentes tipos de distancia

print("Comparacion usando diferentes valores de k, con datos estandarizados y diferentes tipos de distancia")

print("#################################################################################")
print("############################# Datos sin estandarizar ############################")
print("#################################################################################")

# Probar diferentes valores de k con datos estandarizados y diferentes tipos de distancia
ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
distance_types = ['euclidean', 'manhattan', 'chebyshev']
resultados = {}

# Iteramos sobre los valores de k y distancia
for distance_type in distance_types:
    for k in ks:
        # Lanzamos la prediccion con k vecinos más cercanos
        predictions_k = k_nearest_neighbors_benchmark(train_data, test_data, variables, 'class', k, distance_type)
        encoded_predictions_k = np.array([true_label_dict[label] for label in predictions_k])
        # Calculamos los valores de accuracy, precision, recall y f1 score
        accuracy_k, precision_k, recall_k, f1_k = calcular_metricas(encoded_true_labels,
                                                                    encoded_predictions_k)

        # Guardamos los valores de accuracy, precision, recall y f1 score en un diccionario
        resultados[(distance_type, k)] = (accuracy_k, precision_k, recall_k, f1_k)

mostrar_resultados(resultados)

# Volvemos a hacer las pruebas estandarizando los datos

print("#################################################################################")
print("############################# Datos estandarizados ##############################")
print("#################################################################################")

for distance_type in distance_types:
    for k in ks:
        # Lanzamos la prediccion con k vecinos más cercanos
        predictions_k = k_nearest_neighbors_benchmark(train_data, test_data, variables, 'class', k, distance_type, True)
        encoded_predictions_k = np.array([true_label_dict[label] for label in predictions_k])
        # Calculamos los valores de accuracy, precision, recall y f1 score
        accuracy_k, precision_k, recall_k, f1_k = calcular_metricas(encoded_true_labels,
                                                                    encoded_predictions_k)

        # Guardamos los valores de accuracy, precision, recall y f1 score en un diccionario
        resultados[(distance_type, k)] = (accuracy_k, precision_k, recall_k, f1_k)

mostrar_resultados(resultados)
