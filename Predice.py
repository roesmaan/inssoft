import numpy

def readTheta():
    print("loading theta...")
    t = numpy.loadtxt('./theta.csv')
    print("load succesfull")
    return t

def sigmoidal(z):
    #print(z)
    s = numpy.exp(z)
    #print(s)
    return 1/(1+s)

def predice(theta,X):
    print("Predicting values for X:", X, '...')
    #print(X.shape)
    x = numpy.c_[numpy.ones(X.shape[0]), X[:]]
    #print(x)
    treshold = 0.5
    #print('p')
    print("Value:", sigmoidal(x.dot(theta)*-1))
    p = sigmoidal(x.dot(theta)*-1) >= treshold

    return(p.astype(int))

t = readTheta()
print("Prediccion: ", predice(t, numpy.matrix([5,99,74,27,0,29,0.203,32])))
