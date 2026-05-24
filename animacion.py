import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from simulacion import resolver_metodo
from utils import ruta_animacion


def crear_animacion(N, L, a, t0, tf, h, perturbacion):
    # Crea una animacion de vehiculos en una via circular
    resultado = resolver_metodo("RK4", N, L, a, t0, tf, h, perturbacion)
    t = resultado["t"]
    y = resultado["y"]
    x = y[:, :N] % L
    v = y[:, N:]
    dispersion = np.std(v, axis=1)

    max_frames = 160
    indices = np.linspace(0, len(t) - 1, min(max_frames, len(t)), dtype=int)
    angulos = 2.0 * np.pi * x / L
    radio = 1.0

    fig, (ax_circulo, ax_dispersion) = plt.subplots(1, 2, figsize=(10, 5))
    ax_circulo.set_aspect("equal")
    ax_circulo.set_xlim(-1.25, 1.25)
    ax_circulo.set_ylim(-1.25, 1.25)
    ax_circulo.axis("off")
    theta = np.linspace(0, 2.0 * np.pi, 300)
    ax_circulo.plot(np.cos(theta), np.sin(theta), color="0.3", linewidth=2)
    puntos = ax_circulo.scatter([], [], s=45, color="tab:blue")
    texto = ax_circulo.text(-1.15, 1.12, "", fontsize=10)

    ax_dispersion.set_xlim(t[0], t[-1])
    ax_dispersion.set_ylim(0, max(0.05, float(np.max(dispersion)) * 1.1))
    ax_dispersion.set_xlabel("Tiempo")
    ax_dispersion.set_ylabel("Desviación estándar")
    ax_dispersion.grid(True, alpha=0.3)
    linea, = ax_dispersion.plot([], [], color="tab:red")

    def actualizar(frame):
        idx = indices[frame]
        coords = np.column_stack([radio * np.cos(angulos[idx]), radio * np.sin(angulos[idx])])
        puntos.set_offsets(coords)
        texto.set_text(f"t = {t[idx]:.2f}")
        linea.set_data(t[: idx + 1], dispersion[: idx + 1])
        return puntos, texto, linea

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
