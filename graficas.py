import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analisis import tabla_errores
from modelo import calcular_headways
from simulacion import resolver_metodo, resolver_rk45
from utils import ruta_figura


def _guardar(fig, nombre):
    # Guarda una figura y libera memoria
    fig.tight_layout()
    fig.savefig(ruta_figura(nombre), dpi=160)
    plt.close(fig)


def grafica_evolucion_velocidades(resultado, N):
    t = resultado["t"]
    v = resultado["y"][:, N:]
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, v[:, 0], label="Vehiculo 1")
    ax[0].plot(t, v[:, N // 2], label=f"Vehiculo {N // 2 + 1}")
    ax[0].set_ylabel("Velocidad")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(t, np.std(v, axis=1), color="tab:red")
    ax[1].set_xlabel("Tiempo")
    ax[1].set_ylabel("Dispersion de velocidades")
    ax[1].grid(True, alpha=0.3)
    _guardar(fig, "evolucion_velocidades.png")


def grafica_regimenes(N, L, t0, tf, h, perturbacion):
    fig, ax = plt.subplots(figsize=(9, 5))
    for a, etiqueta in [(1.0, "a=1.0 inestable"), (3.0, "a=3.0 estable")]:
        resultado = resolver_metodo("RK4", N, L, a, t0, tf, h, perturbacion)
        v = resultado["y"][:, N:]
        ax.plot(resultado["t"], np.std(v, axis=1), label=etiqueta)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Desviación estándar de velocidades")
    ax.set_title("Regímenes estable e inestable")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, "regimenes_estable_inestable.png")


def grafica_convergencia(tabla):
    fig, ax = plt.subplots(figsize=(8, 5))
    for metodo, grupo in tabla.groupby("metodo"):
        grupo = grupo.sort_values("h")
        ax.loglog(grupo["h"], grupo["error_promedio"], marker="o", label=metodo)
    ax.set_xlabel("Tamano de paso h")
    ax.set_ylabel("Error promedio")
    ax.set_title("Convergencia de metodos")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _guardar(fig, "convergencia_metodos.png")


def grafica_plano_fase(resultado, N, L):
    x = resultado["y"][:, :N]
    v = resultado["y"][:, N:]
    separacion_vehiculo = np.array([calcular_headways(fila, L)[0] for fila in x])
    fig, ax = plt.subplots(figsize=(7, 6))
    puntos = ax.scatter(separacion_vehiculo, v[:, 0], c=resultado["t"], cmap="viridis", s=18)
    fig.colorbar(puntos, ax=ax, label="Tiempo")
    ax.set_xlabel("Separación h_i del vehículo 1")
    ax.set_ylabel("Velocidad vehiculo 1")
    ax.set_title("Plano de fase separación-velocidad")
    ax.grid(True, alpha=0.3)
    _guardar(fig, "plano_fase_headway_velocidad.png")


def grafica_comparacion_metodos(resultados, rk45, N, h=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, resultado in resultados.items():
        ax.plot(resultado["t"], resultado["y"][:, N], label=nombre, linewidth=1.5)
    ax.plot(rk45["t"], rk45["y"][:, N], "k--", label="RK45", linewidth=1.2)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Velocidad vehiculo 1")
    if h is None:
        ax.set_title("Comparación de métodos")
    else:
        ax.set_title(f"Comparación de métodos para h={h:g}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, "comparacion_metodos.png")


def grafica_sensibilidad(N, L, a, t0, tf, h):
    fig, ax = plt.subplots(figsize=(9, 5))
    for perturbacion in [0.20, 0.21]:
        resultado = resolver_metodo("RK4", N, L, a, t0, tf, h, perturbacion)
        v = resultado["y"][:, N:]
        ax.plot(resultado["t"], np.std(v, axis=1), label=f"perturbacion={perturbacion:.2f}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Desviación estándar de velocidades")
    ax.set_title("Sensibilidad a condiciones iniciales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, "sensibilidad_perturbacion.png")


def grafica_densidad(N, a, t0, tf, h, perturbacion):
    fig, ax = plt.subplots(figsize=(9, 5))
    for L in [40.0, 50.0]:
        resultado = resolver_metodo("RK4", N, L, a, t0, tf, h, perturbacion)
        v = resultado["y"][:, N:]
        ax.plot(resultado["t"], np.std(v, axis=1), label=f"L={L:.0f}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Desviación estándar de velocidades")
    ax.set_title("Escenario de densidad diferente")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, "densidad_diferente.png")


def generar_graficas_base(N, L, a, t0, tf, h, perturbacion, resultados=None, tabla=None):
    # Genera todas las figuras obligatorias del proyecto
    if resultados is None:
        resultados = {m: resolver_metodo(m, N, L, a, t0, tf, h, perturbacion) for m in ["Euler", "Heun", "RK4"]}
    if tabla is None:
        tabla = tabla_errores(N, L, a, t0, tf, [0.20, 0.10, 0.05], perturbacion)
    rk45 = resolver_rk45(N, L, a, t0, tf, resultados["RK4"]["t"], perturbacion)
    grafica_evolucion_velocidades(resultados["RK4"], N)
    grafica_regimenes(N, L, t0, tf, h, perturbacion)
    grafica_convergencia(tabla)
    grafica_plano_fase(resultados["RK4"], N, L)
    grafica_comparacion_metodos(resultados, rk45, N, h)
    grafica_sensibilidad(N, L, a, t0, tf, h)
    grafica_densidad(N, a, t0, tf, h, perturbacion)
