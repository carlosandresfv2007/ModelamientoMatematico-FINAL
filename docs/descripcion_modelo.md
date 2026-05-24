# Descripcion del modelo OVM de Bando

El modelo de velocidad optima de Bando representa cada vehiculo mediante su posicion, velocidad y separacion con el vehiculo de adelante. La ecuacion original es:

```text
x_i''(t) = a * (V(h_i(t)) - x_i'(t))
```

Esta ecuacion es de segundo orden porque la variable principal es la posicion `x_i(t)` y aparece su segunda derivada, que corresponde a la aceleracion. El parametro `a` controla la rapidez con la que cada conductor ajusta su velocidad.

La velocidad optima se define como:

```text
V(h) = tanh(h - 2) + tanh(2)
```

La distancia de seguimiento en una via circular es:

```text
h_i = x_(i+1) - x_i
h_N = x_1 + L - x_N
```

Para implementar el sistema se introduce `v_i = x_i'`. Asi, el sistema de segundo orden se reduce a:

```text
x_i' = v_i
v_i' = a * (V(h_i) - v_i)
```

El vector de estado queda organizado como:

```text
y = [x_1, x_2, ..., x_N, v_1, v_2, ..., v_N]
```

Esta reduccion permite aplicar metodos numericos de primer orden sin cambiar la interpretacion del modelo original.
