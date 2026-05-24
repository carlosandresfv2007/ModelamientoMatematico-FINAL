import numpy as np
import pandas as pd

from simulacion import resolver_metodo, resolver_rk45
from utils import ruta_tabla


def error_temporal(y_metodo, y_rk45, N):
    # Calcula el error relativo al tamano del sistema
    diferencias = y_metodo - y_rk45
    return np.linalg.norm(diferencias, axis=1) / np.sqrt(2 * N)


def resumen_error(resultado_metodo, resultado_rk45, N):
    # Resume el error contra la referencia RK45
    errores = error_temporal(resultado_metodo["y"], resultado_rk45["y"], N)
    return {
        "error_promedio": float(np.mean(errores)),
        "error_maximo": float(np.max(errores)),
        "error_final": float(errores[-1]),
        "errores": errores,
    }


def tabla_errores(N, L, a, t0, tf, pasos, perturbacion, guardar=True):
    # Construye la tabla de convergencia para varios tamanos de paso
    filas = []
    for h in pasos:
        for metodo in ["Euler", "Heun", "RK4"]:
            resultado = resolver_metodo(metodo, N, L, a, t0, tf, h, perturbacion)
            referencia = resolver_rk45(N, L, a, t0, tf, resultado["t"], perturbacion)
            resumen = resumen_error(resultado, referencia, N)
            filas.append(
                {
                    "metodo": metodo,
                    "h": h,
                    "error_promedio": resumen["error_promedio"],
                    "error_maximo": resumen["error_maximo"],
                    "error_final": resumen["error_final"],
                    "tiempo_segundos": resultado["tiempo"],
                }
            )
    df = pd.DataFrame(filas)
    if guardar:
        df.to_csv(ruta_tabla("tabla_errores_ovm.csv"), index=False)
    return df


def verificar_minimos(N, L, a, t0, tf, h, perturbacion, tabla):
    # Ejecuta verificaciones simples pedidas en la especificacion
    from modelo import calcular_headways, estado_inicial

    y0 = estado_inicial(N, L, perturbacion)
    assert len(y0) == 2 * N, "El estado debe tener 2*N variables"
    assert np.all(calcular_headways(y0[:N], L) > 0), "Las separaciones iniciales deben ser positivas"
    base = tabla[tabla["h"] == h]
    euler_error = float(base[base["metodo"] == "Euler"]["error_promedio"].iloc[0])
    rk4_error = float(base[base["metodo"] == "RK4"]["error_promedio"].iloc[0])
    assert rk4_error < euler_error, "RK4 debe tener menor error que Euler"
    assert ruta_tabla("tabla_errores_ovm.csv").exists(), "El CSV de errores no existe"
    return True
