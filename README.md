# OVM Trafico Bando

Proyecto de metodos numericos para simular la formacion de trancones fantasma con el Optimal Velocity Model (OVM) de Bando en una via circular.

## Integrantes y roles

- Carlos Andres Florez Vargas: Administracion, programacion y organizacion del repositorio.
- Jose Alejandro Munoz Jaramillo: Modelamiento matematico, reduccion a primer orden y programacion.
- Andres Arroyave Cardona: Investigacion, referencias, simulacion y programacion.

## Modelo matematico

Para cada vehiculo se modelan la posicion `x_i(t)`, la velocidad `v_i(t)` y la separacion con el vehiculo de adelante `h_i(t)`. El modelo original es de segundo orden:

```text
x_i''(t) = a * (V(h_i(t)) - x_i'(t))
```

con

```text
V(h) = tanh(h - 2) + tanh(2)
```

En una via circular:

```text
h_i = x_(i+1) - x_i
h_N = x_1 + L - x_N
```

Es de segundo orden porque la ecuacion describe la aceleracion `x_i''(t)` de cada vehiculo. La velocidad cambia segun la diferencia entre la velocidad optima deseada y la velocidad actual.

## Reduccion a primer orden

Para resolver numericamente se define `v_i = x_i'`. Entonces:

```text
x_i' = v_i
v_i' = a * (V(h_i) - v_i)
```

El vector de estado usado por el codigo es:

```text
y = [x_1, x_2, ..., x_N, v_1, v_2, ..., v_N]
```

Para `N = 20`, el sistema reducido tiene `40` ecuaciones de primer orden.

## Metodos numericos

El archivo `metodos.py` implementa manualmente:

- Euler explicito.
- Heun / RK2.
- Runge-Kutta clasico de cuarto orden / RK4.

La funcion `solve_ivp` con `method="RK45"` se usa solo como referencia de validacion.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Antes de ejecutar los scripts del proyecto, activa siempre el entorno virtual:

```bash
source .venv/bin/activate
```

## Ejecucion principal

```bash
python main.py
```

Este comando genera una animacion real del trafico sobre un anillo en la carpeta `resultados/animaciones/`.

La animacion dibuja una pista circular con los vehiculos representados como circulos moviendose sobre la circunferencia. Los circulos cambian de color segun su velocidad: verde para rapido, amarillo para velocidad media y rojo para lento. Asi se observa de manera intuitiva como una pequena perturbacion puede generar una onda de frenado que se propaga por la fila.

## Interfaz Streamlit

```bash
streamlit run app.py
```

La interfaz permite cambiar `N`, `L`, `a`, la cantidad y amplitud de perturbaciones iniciales, `t_final` y `h` para generar y visualizar la animacion del anillo.

Para exponer la animacion:

> La animacion permite observar de manera intuitiva la formacion de ondas de congestion y la transicion entre flujo uniforme e inestabilidad.

## Resultados

- Tablas: `resultados/tablas/`
- Figuras: `resultados/figuras/`
- Animaciones: `resultados/animaciones/`

## Uso de IA

La IA se uso como apoyo conceptual y de organizacion para estructurar el codigo, documentar el modelo, preparar verificaciones y automatizar la generacion de resultados. Las ecuaciones, metodos y conclusiones deben revisarse y defenderse dentro del contexto del curso.
