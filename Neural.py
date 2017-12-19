import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Calcula el costo de una salida específica de una red neuronal de etiquetado
# con función de activación sigmoidal
def funcionCostoEtiquetado(a, y):
    j = -y*np.log(a)-((1-y)*(np.log(1-a)))
    J = np.sum(j)/y.shape[1]
    return J

# Regresa el resultado de la función de activación sigmoidal
def sigmoidal(z):
    A = 1/(1+np.exp(-z))
    return A

# Regresa el gradiente sigmoidal (no se utiliza)
def sigmoidalGradiente(z):
    g = sigmoidal(z)
    gp = g*(1-g)
    return gp

# Inicializa los pesos de la neurona aleatoriamente con valores entre [-ep, ep)
def randInicializaPesos(L_in, L_out):
    ep = 0.12
    rand = np.random.uniform(-ep, ep, (L_out, L_in))
    return rand

# Entrena una red neuronal de cualquier número de capas, con cualquier número de
# neuronas, y cualquier número de salidas, con función de activación sigmoidal
# el parámetro nn_params es una lista que contiene, en el elemento 0, el número de
# capas de la red (contando la de entrada y la de salida), y en los elementos 1 al n
# contiene la cantidad de neuronas de la capa de su mismo índice-1. La cuenta de las capas
# comienza en 0. (Ej. Los valores de la capa de entrada están en el índice 1, pero se considera el índice 0)
def entrenaRNGenerica(nn_params, X, y):
    # Definir hiperparámetros y variables auxiliares
    iteraciones = 4000
    alpha = .5
    err_hist = []
    m = y.shape[1]
    num_layers = nn_params[0]

    # Inicializamos un diccionario vaciío que guardará todos los parámetros y valores de la red
    nn = dict([])

    nn['a' + str(0)] = X
    for i in range(1, num_layers):
        nn['w' + str(i)] = randInicializaPesos(nn_params[i], nn_params[i+1])
        nn['b' + str(i)] = np.zeros(nn_params[i+1]).reshape(nn_params[i+1], 1)


    for it in range(iteraciones):
        # FW propagation
        for i in range(1, num_layers):
            nn['z' + str(i)] = np.dot(nn['w' + str(i)], nn['a' + str(i-1)]) + nn['b' + str(i)]
            nn['a' + str(i)] = sigmoidal(nn['z' + str(i)])

        # BW porpagation
        for i in range(num_layers-1, 0, -1):
            if i == (num_layers-1):
                nn['dz' + str(i)] = nn['a' + str(i)] - y
            else:
                nn['dz' + str(i)] = nn['da' + str(i)] * sigmoidalGradiente(nn['z' + str(i)])

            nn['dw' + str(i)] = (1/m) * np.dot(nn['dz' + str(i)], nn['a' + str(i-1)].T)
            nn['db' + str(i)] = (1/m) * np.sum(nn['dz' + str(i)], axis=1, keepdims=True)

            if i > 1:
                nn['da' + str(i-1)] = np.dot(nn['w' + str(i)].T, nn['dz' + str(i)])

        # Actualizacion de pesos
        for i in range(num_layers-1, 0, -1):
            nn['w' + str(i)] = nn['w' + str(i)] - alpha*nn['dw' + str(i)]
            nn['b' + str(i)] = nn['b' + str(i)] - alpha*nn['db' + str(i)]

        cost = funcionCostoEtiquetado(nn['a' + str(num_layers - 1)], y)
        err_hist.append(cost)

    # Graficamos el error a lo largo de las iteraciones
    plt.plot(err_hist)
    plt.show()

    return nn

# Predice la salida de una red neuronal ya entrenada. La función recibe la cantidad de
# capas de la red, al igual que el diccionario generado por la función que entrena.
# Recibe en X los valores que se quieren predecir.
def prediceRNYaEntrenadaGenerica(X, num_layers, trained_nn):
    # FW propagation (predicción)
    for i in range(1, num_layers):
        trained_nn['z' + str(i)] = np.dot(trained_nn['w' + str(i)], trained_nn['a' + str(i-1)]) + trained_nn['b' + str(i)]
        trained_nn['a' + str(i)] = sigmoidal(trained_nn['z' + str(i)])

    # Tomamos el valor más alto de cada salida
    r = np.argmax(trained_nn['a' + str(num_layers - 1)], axis=0)
    return r


def main(argc, argv):
    if argc < 2:
        print("Uso: Proyecto6.py [nombre del archivo de entrenamiento]")
        return 1

    path = os.getcwd()
    data = []
    path += '/' + str(argv[1])

    # Lee datos de entrenamiento
    with open(path, 'r') as fp:
        for line in fp:
            text = line.split()
            if(len(text) > 0):
                numbers = [float(x) for x in text]
                data.append(numbers)

    data = np.array(data)

    # Separa los datos en X y y
    X_data = data[:, 0:data.shape[1]-1]
    y_data = data[:, data.shape[1]-1]
    X_data = X_data.T

    # Definir parámetros de la red
    input_layer_size = X_data.shape[0]
    nn_params = []
    num_layers = 3
    num_labels = 2

    # Guardar parámetros de la red en una lista
    nn_params.append(num_layers)
    nn_params.append(input_layer_size)
    nn_params.append(25)
    nn_params.append(num_labels)

    # Convertir las y's a una matriz
    m = y_data.size
    real_y = np.zeros((num_labels, m))

    for i in range(m):
        real_y[int(y_data[i])%10, i] = 1

    # Entrenar y predecir para todos los valores
    trained_nn = entrenaRNGenerica(nn_params, X_data, real_y)
    res = prediceRNYaEntrenadaGenerica(X_data, num_layers, trained_nn)

    # Cuenta valores correctos e incorrectos
    right = 0
    wrong = 0
    classif = np.zeros(10)
    for i in range(y_data.size):
        if res[i] == y_data[i]%10: # es %10 por que en los datos de entrada el 10 es considerado el 0
            right += 1
        else:
            wrong += 1
        classif[res[i]] += 1


    print("Right: {0}".format(right))
    print("Wrong: {0}".format(wrong))
    print(classif)

    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
