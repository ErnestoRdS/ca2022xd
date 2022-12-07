from time import time


def sum_riemann1(funcion, b, a):
    x = a
    return eval(funcion) * b


def sum_riemann2(funcion, b, lInf, lSup):
    # Poner algo acáxd
    return 0


if __name__ == "__main__":
    funcion = "x**4 - x**3 + x**2 - x"
    lInf = 1
    lSup = 10
    partic = 100

    # Calcular deltaX
    dX = (lSup - lInf) / partic

    # variable para sumar
    suma = 0

    # Para medir el tiempo
    starting = time()

    for i in range(1, partic + 1):
        x = lInf + i * dX - (dX / 2)
        suma += eval(funcion) * dX

    print(f"Tiempo: {time() - starting}")
    print(suma)
