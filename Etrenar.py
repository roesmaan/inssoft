import numpy
import matplotlib.pyplot as plt
import sys

# Función que lee datos de entrenamiento de un .csv (tienen que estar separados) por comas
# La última columna es el resultado (0 para negativo 1 para positivo)
# Reegresa los valores separados en x y y
def leerDatos(file):
    points = numpy.loadtxt(file, delimiter=',') # Save all the file's points in a Matrix
    cols = points.shape[1]
    x = numpy.c_[numpy.ones(points.shape[0]), points[:, numpy.arange(0,cols-1)]]# Create Matrix of Xs with 1s
    y = numpy.c_[points[:, cols-1]] # Create Y Vector, as a Matrix
    return x,y

# Función sigmoidal
def sigmoidal(z):
    s = numpy.exp(z)
    return 1/(1+s)

# Función de costo
def funcionCosto(theta,X,y):
    hyp = sigmoidal(X.dot(theta)*-1)
    J = -1/(y.size) * ((numpy.log(hyp).T.dot(y)) + (numpy.log(1-hyp).T.dot(1-y)))
    grad = (1.0/y.size) * (numpy.dot(X.T,(hyp - y)))
    return J, grad

# Función de aprendizaje que recibe un vector theta inicializado, X que son los datos de entrenamiento
# y son los valores esperados y el número de iteraciones
def aprende(theta,X,y,iteraciones):
    m = y.size # Numero de datos
    alpha = 0.003 # Hiperparámetro alpha de aprendizaje
    for i in range(0,iteraciones):
        hyp = sigmoidal(X.dot(theta)*-1)
        theta = theta - alpha * (1.0/m) * (numpy.dot(X.T,(hyp - y))) # Theta vector for each iteration
    return theta

# función que predice un valor
def predice(theta,X):
    x = numpy.c_[numpy.ones(X.shape[0]), X[:]]
    treshold = 0.5
    p = sigmoidal(x.dot(theta)*-1) >= treshold
    return(p.astype(int))


def main(argc, argv):
    if argc < 2:
        print("Uso: python3 Entrenar.py [nombre del archivo de entrenamiento]")
        return 1

    filename = "./" + argv[1]
    X,Y = leerDatos(filename)
    t = aprende(numpy.zeros((X.shape[1],1)), X,Y, 2000000)
    print("Saving Theta values...")
    numpy.savetxt('theta.csv', t, delimiter=',')
    return 0

if __name__ == "__main__":
    main(len(sys.argv), argv)
