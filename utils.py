from pathlib import Path


RESULTADOS_DIR = Path("resultados")
TABLAS_DIR = RESULTADOS_DIR / "tablas"
FIGURAS_DIR = RESULTADOS_DIR / "figuras"
ANIMACIONES_DIR = RESULTADOS_DIR / "animaciones"
DOCS_DIR = Path("docs")


def crear_carpetas():
    # Crea la estructura de carpetas del proyecto
    for carpeta in [TABLAS_DIR, FIGURAS_DIR, ANIMACIONES_DIR, DOCS_DIR]:
        carpeta.mkdir(parents=True, exist_ok=True)


def ruta_tabla(nombre):
    return TABLAS_DIR / nombre


def ruta_figura(nombre):
    return FIGURAS_DIR / nombre


def ruta_animacion(nombre):
    return ANIMACIONES_DIR / nombre
