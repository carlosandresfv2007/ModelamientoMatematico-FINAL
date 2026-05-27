import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle

from simulacion import resolver_metodo
from utils import crear_carpetas, ruta_animacion


CMAP_VELOCIDAD = LinearSegmentedColormap.from_list(
    "velocidad_trafico",
    ["#d7191c", "#fdae1b", "#1a9641"],
)


def _colores_velocidad(v):
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        v_min -= 0.5
        v_max += 0.5
    normalizador = Normalize(vmin=v_min, vmax=v_max)
    return normalizador, CMAP_VELOCIDAD(normalizador(v))


def _texto_perturbacion(perturbacion):
    if np.isscalar(perturbacion):
        return f"perturbacion inicial = {float(perturbacion):.2f}"
    return f"perturbaciones iniciales = {len(perturbacion)}"


def crear_animacion(N, L, a, t0, tf, h, perturbacion):
    # Crea una animacion de vehiculos en una via circular
    crear_carpetas()
    resultado = resolver_metodo("RK4", N, L, a, t0, tf, h, perturbacion)
    t = resultado["t"]
    y = resultado["y"]
    x = y[:, :N] % L
    v = y[:, N:]
    dispersion = np.std(v, axis=1)

    max_frames = 180
    indices = np.linspace(0, len(t) - 1, min(max_frames, len(t)), dtype=int)
    angulos = 2.0 * np.pi * x / L
    radio = 1.0
    normalizador, _ = _colores_velocidad(v)

    fig, (ax_circulo, ax_dispersion) = plt.subplots(
        1,
        2,
        figsize=(11, 5.5),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    ax_circulo.set_aspect("equal")
    ax_circulo.set_xlim(-1.35, 1.35)
    ax_circulo.set_ylim(-1.35, 1.35)
    ax_circulo.axis("off")
    theta = np.linspace(0, 2.0 * np.pi, 300)
    ax_circulo.plot(np.cos(theta), np.sin(theta), color="0.18", linewidth=8, alpha=0.25, zorder=1)
    ax_circulo.plot(np.cos(theta), np.sin(theta), color="0.15", linewidth=1.4, zorder=2)
    ax_circulo.add_patch(Circle((0, 0), 0.78, fill=False, linestyle="--", linewidth=1.0, color="0.65", zorder=1))
    ax_circulo.set_title("Carros sobre la autopista circular", fontsize=12)
    carros = []
    for i in range(N):
        centro = (radio * np.cos(angulos[0, i]), radio * np.sin(angulos[0, i]))
        carro = Circle(centro, radius=0.052, ec="black", lw=0.7, zorder=5)
        carros.append(carro)
        ax_circulo.add_patch(carro)
    texto = ax_circulo.text(-1.28, 1.22, "", fontsize=10)
    explicacion = ax_circulo.text(
        -1.28,
        -1.26,
        "Rojo: lento   Amarillo: medio   Verde: rapido",
        fontsize=9,
        color="0.25",
    )

    ax_dispersion.set_xlim(t[0], t[-1])
    ax_dispersion.set_ylim(0, max(0.05, float(np.max(dispersion)) * 1.1))
    ax_dispersion.set_xlabel("Tiempo")
    ax_dispersion.set_ylabel("Dispersión de velocidades")
    ax_dispersion.set_title("Onda de frenado e inestabilidad")
    ax_dispersion.grid(True, alpha=0.3)
    linea, = ax_dispersion.plot([], [], color="tab:red")
    marcador, = ax_dispersion.plot([], [], marker="o", color="tab:red")

    def actualizar(frame):
        idx = indices[frame]
        colores = CMAP_VELOCIDAD(normalizador(v[idx]))
        for i, carro in enumerate(carros):
            carro.center = (radio * np.cos(angulos[idx, i]), radio * np.sin(angulos[idx, i]))
            carro.set_facecolor(colores[i])
        texto.set_text(f"t = {t[idx]:.2f}   {_texto_perturbacion(perturbacion)}")
        linea.set_data(t[: idx + 1], dispersion[: idx + 1])
        marcador.set_data([t[idx]], [dispersion[idx]])
        return [*carros, texto, explicacion, linea, marcador]

    animacion = FuncAnimation(fig, actualizar, frames=len(indices), interval=80, blit=False)
    gif = ruta_animacion("animacion_trafico_ovm.gif")
    mp4 = ruta_animacion("animacion_trafico_ovm.mp4")
    try:
        animacion.save(gif, writer=PillowWriter(fps=15))
        plt.close(fig)
        return gif
    except Exception:
        try:
            animacion.save(mp4, fps=15)
            plt.close(fig)
            return mp4
        except Exception as exc:
            plt.close(fig)
            raise RuntimeError(f"No se pudo guardar la animacion: {exc}") from exc
