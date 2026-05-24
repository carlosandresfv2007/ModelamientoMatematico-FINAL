import numpy as np


def _mallado_tiempo(t0, tf, h):
    # Crea un mallado que incluye tf aunque el paso no divida exacto
    pasos = int(np.ceil((tf - t0) / h))
    t_values = t0 + np.arange(pasos + 1) * h
    t_values[-1] = tf
    return t_values


def euler(f, t0, tf, y0, h, args=()):
    # Aplica el metodo de Euler explicito
    t_values = _mallado_tiempo(t0, tf, h)
    y_values = np.zeros((len(t_values), len(y0)), dtype=float)
    y_values[0] = y0

    for k in range(len(t_values) - 1):
        dt = t_values[k + 1] - t_values[k]
        y_values[k + 1] = y_values[k] + dt * f(t_values[k], y_values[k], *args)

    return t_values, y_values


def heun(f, t0, tf, y0, h, args=()):
    # Aplica el metodo de Heun, tambien llamado RK2
    t_values = _mallado_tiempo(t0, tf, h)
    y_values = np.zeros((len(t_values), len(y0)), dtype=float)
    y_values[0] = y0

    for k in range(len(t_values) - 1):
        dt = t_values[k + 1] - t_values[k]
        k1 = f(t_values[k], y_values[k], *args)
        predictor = y_values[k] + dt * k1
        k2 = f(t_values[k + 1], predictor, *args)
        y_values[k + 1] = y_values[k] + 0.5 * dt * (k1 + k2)

    return t_values, y_values


def rk4(f, t0, tf, y0, h, args=()):
    # Aplica el metodo de Runge Kutta clasico de cuarto orden
    t_values = _mallado_tiempo(t0, tf, h)
    y_values = np.zeros((len(t_values), len(y0)), dtype=float)
    y_values[0] = y0

    for k in range(len(t_values) - 1):
        dt = t_values[k + 1] - t_values[k]
        t = t_values[k]
        y = y_values[k]
        k1 = f(t, y, *args)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
        k4 = f(t + dt, y + dt * k3, *args)
        y_values[k + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_values, y_values
