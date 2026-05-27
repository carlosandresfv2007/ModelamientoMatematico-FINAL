from animacion import crear_animacion


def main():
    N = 20
    L = 40.0
    a = 1.0
    t0 = 0.0
    tf = 40.0
    h = 0.05
    perturbacion = 0.2

    ruta = crear_animacion(N, L, a, t0, tf, h, perturbacion)
    print(f"Animacion generada en: {ruta}")


if __name__ == "__main__":
    main()
