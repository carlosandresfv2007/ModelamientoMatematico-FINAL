from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analisis import tabla_errores
from animacion import crear_animacion
from modelo import calcular_headways
from simulacion import ejecutar_escenarios, resolver_metodo, resolver_rk45, resolver_todos
from utils import crear_carpetas, ruta_figura


st.set_page_config(page_title="OVM Tráfico Bando", layout="wide")
crear_carpetas()


@st.cache_data(show_spinner=False)
def _resolver_todos(N, L, a, t0, tf, h, perturbacion):
    return resolver_todos(N, L, a, t0, tf, h, perturbacion)


@st.cache_data(show_spinner=False)
def _resolver_metodo(nombre, N, L, a, t0, tf, h, perturbacion):
    return resolver_metodo(nombre, N, L, a, t0, tf, h, perturbacion)


@st.cache_data(show_spinner=False)
def _resolver_rk45(N, L, a, t0, tf, t_eval, perturbacion):
    return resolver_rk45(N, L, a, t0, tf, tuple(t_eval), perturbacion)


@st.cache_data(show_spinner=False)
def _tabla_errores(N, L, a, t0, tf, pasos, perturbacion):
    return tabla_errores(N, L, a, t0, tf, list(pasos), perturbacion)


@st.cache_data(show_spinner=False)
def _ejecutar_escenarios(N, L, t0, tf, h, perturbacion):
    return ejecutar_escenarios(N, L, t0, tf, h, perturbacion)


def _mostrar_figura(fig, nombre_archivo):
    fig.savefig(ruta_figura(nombre_archivo), dpi=160)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _construir_perturbaciones(N, cantidad, amplitud, separacion):
    if cantidad <= 1:
        return amplitud

    centro = N // 2
    inicio = centro - ((cantidad - 1) * separacion) // 2
    return [((inicio + i * separacion) % N, amplitud) for i in range(cantidad)]


def _texto_perturbaciones(perturbacion):
    if np.isscalar(perturbacion):
        return f"una perturbacion de amplitud {float(perturbacion):.2f}"
    return ", ".join(f"vehiculo {indice + 1}: {amplitud:.2f}" for indice, amplitud in perturbacion)


def _fig_velocidades(resultado, N):
    t = resultado["t"]
    v = resultado["y"][:, N:]
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, v[:, 0], label="Vehículo 1")
    ax[0].plot(t, v[:, N // 2], label=f"Vehículo {N // 2 + 1}")
    ax[0].set_ylabel("Velocidad")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(t, np.std(v, axis=1), color="tab:red")
    ax[1].set_xlabel("Tiempo")
    ax[1].set_ylabel("Dispersión de velocidades")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_plano_fase(resultado, N, L):
    x = resultado["y"][:, :N]
    v = resultado["y"][:, N:]
    separacion = np.array([calcular_headways(fila, L)[0] for fila in x])
    fig, ax = plt.subplots(figsize=(7, 6))
    puntos = ax.scatter(separacion, v[:, 0], c=resultado["t"], cmap="viridis", s=18)
    fig.colorbar(puntos, ax=ax, label="Tiempo")
    ax.set_xlabel("Separación h_i del vehículo 1")
    ax.set_ylabel("Velocidad vehículo 1")
    ax.set_title("Plano de fase separación-velocidad")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_comparacion(resultados, rk45, N, h):
    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, resultado in resultados.items():
        ax.plot(resultado["t"], resultado["y"][:, N], label=nombre, linewidth=1.5)
    ax.plot(rk45["t"], rk45["y"][:, N], "k--", label="RK45", linewidth=1.2)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Velocidad vehículo 1")
    ax.set_title(f"Comparación de métodos para h={h:g}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_convergencia(tabla):
    fig, ax = plt.subplots(figsize=(8, 5))
    for metodo, grupo in tabla.groupby("metodo"):
        grupo = grupo.sort_values("h")
        ax.loglog(grupo["h"], grupo["error_promedio"], marker="o", label=metodo)
    ax.set_xlabel("Tamaño de paso h")
    ax.set_ylabel("Error promedio")
    ax.set_title("Convergencia de métodos")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_eficiencia(tabla):
    fig, ax = plt.subplots(figsize=(8, 5))
    for metodo, grupo in tabla.groupby("metodo"):
        ax.scatter(grupo["tiempo_segundos"], grupo["error_promedio"], label=metodo, s=55)
        for _, fila in grupo.iterrows():
            ax.annotate(
                f"h={fila['h']:g}",
                (fila["tiempo_segundos"], fila["error_promedio"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tiempo de cómputo (s)")
    ax.set_ylabel("Error promedio")
    ax.set_title("Eficiencia: costo computacional vs error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_dispersion_escenarios(escenarios):
    fig, ax = plt.subplots(figsize=(9, 5))
    etiquetas = {
        "base": "base",
        "estable": "a=3.0 estable",
        "sensibilidad_020": "perturbacion=0.20",
        "sensibilidad_021": "perturbacion=0.21",
        "densidad_L40": "L=40",
        "densidad_L50": "L=50",
    }
    for nombre, resultado in escenarios.items():
        N_local = resultado["y"].shape[1] // 2
        v = resultado["y"][:, N_local:]
        ax.plot(resultado["t"], np.std(v, axis=1), label=etiquetas.get(nombre, nombre))
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Dispersión de velocidades")
    ax.set_title("Escenarios comparativos")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


st.title("OVM Tráfico Bando")

with st.sidebar:
    st.header("Parametros")
    N = st.slider("Número de vehículos", 8, 40, 20, 1)
    L = st.slider("Longitud de la via circular", 20.0, 80.0, 40.0, 1.0)
    a = st.slider("Sensibilidad a", 0.2, 4.0, 1.0, 0.1)
    cantidad_perturbaciones = st.slider("Cantidad de perturbaciones", 1, 5, 1, 1)
    amplitud_maxima = max(0.05, 0.45 * L / N)
    amplitud_defecto = min(0.2, amplitud_maxima)
    amplitud_perturbacion = st.slider(
        "Amplitud de perturbación",
        -amplitud_maxima,
        amplitud_maxima,
        amplitud_defecto,
        0.01,
    )
    separacion_perturbaciones = st.slider(
        "Separación entre perturbaciones",
        1,
        max(1, N // 2),
        max(1, N // 4),
        1,
        disabled=cantidad_perturbaciones == 1,
    )
    tf = st.slider("Tiempo final", 10.0, 80.0, 40.0, 5.0)
    h = st.select_slider("Paso h", options=[0.2, 0.1, 0.05], value=0.05)
    metodo_principal = st.selectbox("Metodo principal", ["RK4", "Heun", "Euler"], index=0)

t0 = 0.0
perturbacion = _construir_perturbaciones(
    N,
    cantidad_perturbaciones,
    amplitud_perturbacion,
    separacion_perturbaciones,
)
tabs = st.tabs(["Animación", "Velocidades", "Métodos", "Errores", "Plano de fase", "Escenarios"])

with tabs[0]:
    st.write(
        "La animación permite observar de manera intuitiva la formación de ondas de "
        "congestión y la transición entre flujo uniforme e inestabilidad."
    )
    st.caption(f"Perturbaciones actuales: {_texto_perturbaciones(perturbacion)}")
    if st.button("Generar animacion", type="primary"):
        with st.spinner("Simulando vehículos sobre el anillo..."):
            ruta = crear_animacion(N, L, a, t0, tf, h, perturbacion)
        st.success(f"Animación generada: {ruta}")
        ruta = Path(ruta)
        if ruta.suffix.lower() == ".gif":
            st.image(str(ruta))
        else:
            st.video(str(ruta))

with tabs[1]:
    with st.spinner("Calculando evolucion de velocidades..."):
        resultado = _resolver_metodo(metodo_principal, N, L, a, t0, tf, h, perturbacion)
    col1, col2, col3 = st.columns(3)
    velocidades = resultado["y"][:, N:]
    col1.metric("Tiempo de calculo", f"{resultado['tiempo']:.4f} s")
    col2.metric("Velocidad minima", f"{np.min(velocidades):.4f}")
    col3.metric("Velocidad maxima", f"{np.max(velocidades):.4f}")
    _mostrar_figura(_fig_velocidades(resultado, N), "evolucion_velocidades.png")

with tabs[2]:
    with st.spinner("Comparando Euler, Heun, RK4 y RK45..."):
        resultados = _resolver_todos(N, L, a, t0, tf, h, perturbacion)
        rk45 = _resolver_rk45(N, L, a, t0, tf, resultados["RK4"]["t"], perturbacion)
    _mostrar_figura(_fig_comparacion(resultados, rk45, N, h), "comparacion_metodos.png")
    tiempos = pd.DataFrame(
        [{"metodo": nombre, "tiempo_segundos": res["tiempo"]} for nombre, res in resultados.items()]
        + [{"metodo": "RK45", "tiempo_segundos": rk45["tiempo"]}]
    )
    st.dataframe(tiempos, width="stretch")

with tabs[3]:
    pasos = (0.20, 0.10, 0.05)
    with st.spinner("Calculando tabla de errores..."):
        tabla = _tabla_errores(N, L, a, t0, tf, pasos, perturbacion)
    _mostrar_figura(_fig_convergencia(tabla), "convergencia_metodos.png")
    _mostrar_figura(_fig_eficiencia(tabla), "eficiencia_metodos.png")
    st.dataframe(tabla, width="stretch")

with tabs[4]:
    with st.spinner("Calculando plano de fase..."):
        resultado = _resolver_metodo(metodo_principal, N, L, a, t0, tf, h, perturbacion)
    _mostrar_figura(_fig_plano_fase(resultado, N, L), "plano_fase_separacion_velocidad.png")

with tabs[5]:
    with st.spinner("Ejecutando escenarios comparativos..."):
        escenarios = _ejecutar_escenarios(N, L, t0, tf, h, perturbacion)
    _mostrar_figura(_fig_dispersion_escenarios(escenarios), "escenarios_comparativos.png")
    resumen = []
    for nombre, resultado in escenarios.items():
        N_local = resultado["y"].shape[1] // 2
        v = resultado["y"][:, N_local:]
        resumen.append(
            {
                "escenario": nombre,
                "dispersion_final": float(np.std(v[-1])),
                "dispersion_maxima": float(np.max(np.std(v, axis=1))),
                "tiempo_segundos": resultado["tiempo"],
            }
        )
    st.dataframe(pd.DataFrame(resumen), width="stretch")
