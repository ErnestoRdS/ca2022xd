import numpy as np

# Función para competir -> Regresar el
# que tenga menos 1's como el ganador
def competir(pop):
    # return pop[np.argmin(np.sum(pop, axis=1))], pop[np.argmax(np.sum(pop, axis=1))]
    return (
        pop[np.argsort(np.sum(pop, axis=-1))[1]],
        pop[np.argsort(np.sum(pop, axis=-1))[0]],
    )


if __name__ == "__main__":
    # A TOMAR EN CUENTA
    l = 50
    n = 100
    # GENERAR VECTOR DE PROBABILIDAD
    pv = np.full(l, 0.5)
    # REPETIR
    iter = 0
    while not ((np.sum(pv == 0) + np.sum(pv == 1)) == l):
        # Generar la población (de 2) para los vectores que competirán
        pop = np.random.random((2, l))
        # GENERAR VECTORES SHIDOS
        pop = (pop < pv).astype(int)
        # PONERLOS A COMPETIR
        winner, loser = competir(pop)
        # UPDATEAR VECTOR DE PROBABILIDAD
        for i in range(l):
            if winner[i] != loser[i]:
                if winner[i] == 1:
                    if pv[i] < 1:
                        pv[i] = pv[i] + (1 / n)
                else:
                    if pv[i] > 0:
                        pv[i] = pv[i] - (1 / n)
        iter += 1
        pv = np.around(pv, 4)
        np.clip(pv, 0, 1, pv)

    # print(iter)
    pv = np.around(pv, 2)
    print(pv)
