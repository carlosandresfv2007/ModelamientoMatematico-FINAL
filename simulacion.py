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
    # Ejecuta los escenarios obligatorios con RK4 para graficas comparativas
    escenarios = {
        "base": {"N": N, "L": L, "a": 1.0, "perturbacion": perturbacion},
        "estable": {"N": N, "L": L, "a": 3.0, "perturbacion": perturbacion},
        "sensibilidad_020": {"N": N, "L": L, "a": 1.0, "perturbacion": 0.20},
        "sensibilidad_021": {"N": N, "L": L, "a": 1.0, "perturbacion": 0.21},
        "densidad_L40": {"N": N, "L": 40.0, "a": 1.0, "perturbacion": perturbacion},
        "densidad_L50": {"N": N, "L": 50.0, "a": 1.0, "perturbacion": perturbacion},
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
