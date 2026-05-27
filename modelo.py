import numpy as np


def velocidad_optima(h):
    # Calcula la velocidad deseada segun la separacion
    return np.tanh(h - 2.0) + np.tanh(2.0)


def calcular_headways(x, L):
    # Calcula las distancias de seguimiento en una via circular
    x = np.asarray(x, dtype=float)
    headways = np.roll(x, -1) - x
    headways[-1] = x[0] + L - x[-1]
    return headways


def ovm(t, y, N, L, a):
    # Reduce el modelo de segundo orden a un sistema de primer orden
    y = np.asarray(y, dtype=float)
    x = y[:N]
    v = y[N:]
    headways = calcular_headways(x, L)
    dxdt = v
    dvdt = a * (velocidad_optima(headways) - v)
    return np.concatenate([dxdt, dvdt])


def normalizar_perturbaciones(N, perturbacion):
    # Convierte una perturbacion unica o multiple a pares (indice, amplitud)
    if np.isscalar(perturbacion):
        return [(N // 2, float(perturbacion))]

    perturbaciones = []
    for item in perturbacion:
        if isinstance(item, dict):
            indice = int(item["indice"])
            amplitud = float(item["amplitud"])
        else:
            indice, amplitud = item
            indice = int(indice)
            amplitud = float(amplitud)
        perturbaciones.append((indice % N, amplitud))
    return perturbaciones


def estado_inicial(N, L, perturbacion):
    # Construye posiciones uniformes con una o varias perturbaciones
    indices = np.arange(N, dtype=float)
    x0 = indices * L / N
    for indice, amplitud in normalizar_perturbaciones(N, perturbacion):
        x0[indice] += amplitud
    v0 = np.full(N, velocidad_optima(L / N), dtype=float)
    return np.concatenate([x0, v0])
