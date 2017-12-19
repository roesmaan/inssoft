import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

# Dunción que encuentra el centroide más cercano a un dato y se lo asigna.
# Recibe datos de entrenamiento y los centroides iniciales como X y initial_centroids
# respectivamente
def findClosestCentroids(X, initial_centroids):
    examples = X.shape[0]
    K = initial_centroids.shape[0]
    idx = np.zeros(examples)
    for i in range(examples):
        # Calcula la distancia a cada centroide y selecciona el índice de la menor
        distances = np.zeros(K)
        for j in range(K):
            distances[j] = np.sqrt(np.sum(np.square(initial_centroids[j] - X[i])))
        idx[i] = np.argmin(distances)
        # Regresamos un arreglo con los índices apropiados
    return idx.astype(int)

# De un conjunto de datos, calcula los nuevos centroides, esto se hace promediando
# todos los valores de un conjunto específico. El vector idx contiene el número
# del conjunto al que corresponde un valor en X. Recibe datos de entrenamiento,
# el vector de índices que representa el clúster al que cada valor pertenece, y K,
# el número de clústers en X, idx y K respectivamente
def computeCentroids(X, idx, K):
    sums = np.zeros((K, X.shape[1]))
    count = np.zeros((K, 1))

    for i in range(X.shape[0]):
        sums[idx[i]] += X[i]
        count[idx[i]] += 1

    res = sums / count
    # Regresamos un vector con los nuevos centroides
    return res

# Corre el algoritmo K-means, algoritmo de aprendjizaje no supervisado que clasifica un conjunto de datos en K clusters
# dependiendo de su cercanía entre ellos. Recibe valores de entrenamiento, centroides iniciales, iteraciones máximas,
# y la opcioón para graficar o no lops resultados del algoritmo en X, initial_centroids, max_iters y true respectivamente
def runkMeans(X, initial_centroids, max_iters):
    # inicializamos algunos valores
    early = False
    centroids = initial_centroids
    K = initial_centroids.shape[0]
    centroid_hist = np.empty((initial_centroids.shape[0], initial_centroids.shape[1], 0))
    centroid_hist = np.append(centroid_hist, centroids[..., np.newaxis], axis=2)

    # corremos el algortimo por max_iters iteraciones
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

        # Salirse temprano si los centroides no cambian
        if np.allclose(centroids, centroid_hist[:, :, -1]) == True:
            print("Exited early after {0} iterations".format(i))
            early = True
            break

        # Agregar al historial los centroides
        centroid_hist = np.append(centroid_hist, centroids[..., np.newaxis], axis=2)

    # Si no se sale temprano, avisa.
    if early == False:
        print("Exited after {0} iterations (Max iterations)".format(max_iters))

    return idx, centroids

# Selecciona K valores al azar del conjunto de entrenamiento y los hace centroides.
# Recibe valores de entrenamiento y el número de clústers en X y K respectivamente.
def kMeansInitCentroids(X, K):
    indices = np.arange(0, X.shape[0])
    np.random.shuffle(indices)
    indices = indices[0:K]
    initial = np.take(X, indices, axis=0)
    return initial

# función main
def main(argc, argv):
    # recibimos 4 parámetros
    if argc < 4:
        print("Uso: Proyecto9.py [nombre del archivo de entrenamiento] [número de clusters] [número de iteraciones]")
        return 1

    path = os.getcwd()
    data = []
    path += '/' + str(argv[1])

    # Lee datos de entrada
    with open(path, 'r') as fp:
        for line in fp:
            text = line.split(',')
            if(len(text) > 0):
                numbers = [float(x) for x in text]
                data.append(numbers)

    data = np.array(data)
    # recortamos el último valor para no tener que modificar el archivo de entrenamiento que se
    # utiliza para otros algoritmos
    data = data[:, 0:data.shape[1]-1]

    num_clusters = int(argv[2])
    iters = int(argv[3])

    centroids = kMeansInitCentroids(data, num_clusters)
    results, final_centroids = runkMeans(data, centroids, iters)

    # Concatenamos resultados con los datos
    final_results = np.append(data, results.reshape((results.size, 1)), axis=1)

    # Guarda resultados en kmeansresults.txt
    with open('kmeansresults.txt', 'w') as f:
        f.write("Centroides finales:\n")
        for i in range(final_centroids.shape[0]):
            f.write('\n')
            for j in range(final_centroids.shape[1]-1):
                f.write(str(final_centroids[i, j]))
                f.write(' ')

        f.write("\n\n\n")
        f.write("Resultados finales:\n")
        for i in range(final_results.shape[0]):
            f.write('\n')
            for j in range(final_results.shape[1]-1):
                f.write(str(final_results[i, j]))
                f.write(' ')
            f.write(str(int(final_results[i, -1])))


    # Contar elementos por clúster
    unique, count = np.unique(results, return_counts=True)
    res = dict(zip(unique, count))
    print("Clústers: {0}".format(res))
    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
