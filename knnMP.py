import numpy as np
import multiprocessing as mp
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from time import time

# Distancia Euclídea
def distEuc(p, q):
    return np.sqrt(np.sum(np.square(p - q)))


# KNN
def knn(K, xTest):
    distances = np.zeros((len(xTest), len(xTrain)))
    sLabels = []  # np.zeros(len(xTest))
    for id, u in enumerate(xTest):
        distances[id] = np.linalg.norm(xTrain - u, axis=1)
        sLabels.append(
            stats.mode(yTrain[np.argsort(distances[id])][:K], keepdims=True)[0]
        )
    return np.array(sLabels)


# Multi KNN
"""
    TRATAR DE HACER QUE ESTO TE REGRESE UNA LISTA CON LAS PURAS ETIQUETAS
    QUE SE ASOCIARON AL PUNTO ACTUAL, PARA LUEGO SAKR LA MODA YA EN EL FOR
    DONDE SE PASAN LOS PROCESOS:
    proc.append(mp.Process(target=cal_pi, args=(i+1, kmax, n, datos)))
    >>>TAL VEZ SIN K, K, CREO, SE USARÍA YA EN EL FOR DEL MAIN
"""


def multiKnn(xTest, starting, step):
    # distances = np.zeros((len(xTest), len(xTrain)))
    distances = []
    # for id, x in enumerate(xTest, start=starting):
    for i in range(starting, len(xTest), step):
        # Según esta madre debería estar regresando la distancia euclídea entre
        # el punto actual y cada uno de los xTrain que le toknxd
        # distances[id + step] = [
        #     yTrain[id + step],
        #     np.linalg.norm(xTrain[id + step] - x, axis=1),
        # ]
        # distances[id] = np.linalg.norm(xTrain - x, axis=1)
        distances.append(np.linalg.norm(xTrain - xTest[i]))
        return distances


# Desempeño
def accuracy(gottenLabels, y):
    return np.sum(gottenLabels == y) / len(y)


if __name__ == "__main__":
    # VALORES
    start = time()
    K = 3
    ###cpus = mp.cpu_count()
    # PARA EL SPLIT
    testSize = 0.1
    rand = 42
    # DATOS
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    ###x, y = load_digits(return_X_y=True)

    # SPLITEAR
    xTr, xTe, yTr, yTe = train_test_split(
        x, y, test_size=testSize, random_state=rand, stratify=y
    )

    # ENTRENARxd
    global xTrain, yTrain
    xTrain, yTrain = xTr, yTr

    # PROCESOS
    # Poolear
    ####pool = mp.Pool(cpus)
    # PROCESARxd
    # print(np.around(multiKnn(xTe)))
    labels = knn(K, xTe)
    # labels2 = multiKnn(xTe)

    """
        HACER LOS PROCESOS CON 2 FOR:
        1) for cpu in cpus:
        2)     for x in xTest:
        ALGO ASÍxd
        O TAL VEZ..:
        1) for x in xTest:
        2)     for cpu in cpus:
    """

    """ ESTA ONDA ES PARA CHEKR QUE ESTÉ CHIDO, PERO SIN PROCESOS
    labels = np.reshape(knn(K, xTe), len(yTe))


    print(labels.shape)
    labels = np.reshape(knn(K, xTe), len(yTe))
    print(f"Exactitud del conjunto de entrenamiento = {accuracy(labels, yTe)}")
    print(f"Tiempo final = {time() - start}")
    """
