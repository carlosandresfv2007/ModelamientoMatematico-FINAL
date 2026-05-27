import time

import numpy as np
from scipy.integrate import solve_ivp

from metodos import euler, heun, rk4
from modelo import estado_inicial, ovm


METODOS = {
    "Euler": euler,
    "Heun": heun,
    "RK4": rk4,
}


def resolver_metodo(nombre, N, L, a, t0, tf, h, perturbacion):
    # Ejecuta uno de los metodos implementados manualmente
    if nombre not in METODOS:
        raise ValueError(f"Metodo no reconocido: {nombre}")
    y0 = estado_inicial(N, L, perturbacion)
    inicio = time.perf_counter()
    t_values, y_values = METODOS[nombre](ovm, t0, tf, y0, h, args=(N, L, a))
    tiempo = time.perf_counter() - inicio
    return {"metodo": nombre, "t": t_values, "y": y_values, "tiempo": tiempo}


def resolver_todos(N, L, a, t0, tf, h, perturbacion):
    # Ejecuta Euler, Heun y RK4 para los mismos parametros
    return {
        nombre: resolver_metodo(nombre, N, L, a, t0, tf, h, perturbacion)
        for nombre in METODOS
    }


def resolver_rk45(N, L, a, t0, tf, t_eval, perturbacion):
    # Ejecuta RK45 solo como referencia de validacion
    y0 = estado_inicial(N, L, perturbacion)
    inicio = time.perf_counter()
    solucion = solve_ivp(
        ovm,
        (t0, tf),
        y0,
        method="RK45",
        t_eval=np.asarray(t_eval, dtype=float),
        args=(N, L, a),
        rtol=1e-9,
        atol=1e-11,
    )
    tiempo = time.perf_counter() - inicio
    if not solucion.success:
        raise RuntimeError(solucion.message)
    return {"metodo": "RK45", "t": solucion.t, "y": solucion.y.T, "tiempo": tiempo}


def ejecutar_escenarios(N, L, t0, tf, h, perturbacion):
    # Ejecuta casos interpretables para comparar ondas de congestion
    amplitud_base = 0.2 if np.isscalar(perturbacion) else 0.2
    amplitud_fuerte = min(0.45 * L / N, max(0.35, abs(amplitud_base) * 2.0))
    varias_perturbaciones = [
        (N // 4, amplitud_base),
        (N // 2, -0.5 * amplitud_base),
        ((3 * N) // 4, amplitud_base),
    ]
    L_alta_densidad = max(20.0, 0.75 * L)
    a_base = 1.0
    escenarios = {
        "flujo_uniforme": {"N": N, "L": L, "a": a_base, "perturbacion": 0.0},
        "perturbacion_leve": {"N": N, "L": L, "a": a_base, "perturbacion": 0.05},
        "perturbacion_fuerte": {"N": N, "L": L, "a": a_base, "perturbacion": amplitud_fuerte},
        "varias_perturbaciones": {"N": N, "L": L, "a": a_base, "perturbacion": varias_perturbaciones},
        "reaccion_lenta": {"N": N, "L": L, "a": 0.4, "perturbacion": amplitud_base},
        "reaccion_alta": {"N": N, "L": L, "a": 3.0, "perturbacion": amplitud_base},
        "alta_densidad": {"N": N, "L": L_alta_densidad, "a": a_base, "perturbacion": amplitud_base},
    }
    resultados = {}
    for nombre, params in escenarios.items():
        resultados[nombre] = resolver_metodo(
            "RK4",
            params["N"],
            params["L"],
            params["a"],
            t0,
            tf,
            h,
            params["perturbacion"],
        )
    return resultados
