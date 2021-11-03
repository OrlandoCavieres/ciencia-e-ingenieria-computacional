import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parámetros constantes
sigma = 10
b = 8 / 3

# Discretización del tiempo
tf = 100
N = 20000
discretization = tf / N
time = np.linspace(0, tf, N)


# Funciones de cambio de la variable o actualización.
def variation_x(x, y):
    return sigma * (y - x)


def variation_y(x, y, z, coef_r):
    return coef_r * x - y - x * z


def variation_z(x, y, z):
    return x * y - b * z


# Método para cálculo numérico de ecuaciones diferenciales
def runge_kutta_4_lorenz(initial_x, initial_y, initial_z, coef_r, manilla = False, coef_w = 0):
    """
    Función que realiza Runge Kutta 4 en las ecuaciones conocidas de la ecuación de Lorenz con valores
    constantes de sigma = 10 y b = 8 / 3.
    :param float initial_x: Valor inicial de x.
    :param float initial_y: Valor inicial de y.
    :param float initial_z: Valor inicial de z.
    :param float coef_r: Valor del coeficiente r de la ecuación de Lorenz dada.
    :param bool manilla: Permite usar un r que tiene la forma 24.4 + sin(coef_w * t).
    :param float coef_w: Valor del coeficiente w del efecto manilla.
    :return: (Array de cantidades de conejos en el tiempo, Array de cantidades de zorros en el tiempo)
    """
    x_values = np.zeros(len(time))
    y_values = np.zeros(len(time))
    z_values = np.zeros(len(time))

    x_values[0] = initial_x
    y_values[0] = initial_y
    z_values[0] = initial_z

    for i in range(len(time) - 1):
        coef_r_manilla = (24.4 + np.sin(coef_w * time[i])) if manilla else 0

        k1_x = discretization * variation_x(x_values[i], y_values[i])
        k1_y = discretization * variation_y(x_values[i], y_values[i],
                                            z_values[i], coef_r_manilla if manilla else coef_r)
        k1_z = discretization * variation_z(x_values[i], y_values[i], z_values[i])

        k2_x = discretization * variation_x(x_values[i] + 0.5 * k1_x, y_values[i] + 0.5 * k1_y)
        k2_y = discretization * variation_y(x_values[i] + 0.5 * k1_x, y_values[i] + 0.5 * k1_y,
                                            z_values[i] + 0.5 * k1_z, coef_r_manilla if manilla else coef_r)
        k2_z = discretization * variation_z(x_values[i] + 0.5 * k1_x, y_values[i] + 0.5 * k1_y,
                                            z_values[i] + 0.5 * k1_z)

        k3_x = discretization * variation_x(x_values[i] + 0.5 * k2_x, y_values[i] + 0.5 * k2_y)
        k3_y = discretization * variation_y(x_values[i] + 0.5 * k2_x, y_values[i] + 0.5 * k2_y,
                                            z_values[i] + 0.5 * k2_z, coef_r_manilla if manilla else coef_r)
        k3_z = discretization * variation_z(x_values[i] + 0.5 * k2_x, y_values[i] + 0.5 * k2_y,
                                            z_values[i] + 0.5 * k2_z)

        k4_x = discretization * variation_x(x_values[i] + k3_x, y_values[i] + k3_y)
        k4_y = discretization * variation_y(x_values[i] + k3_x, y_values[i] + k3_y,
                                            z_values[i] + k3_z, coef_r_manilla if manilla else coef_r)
        k4_z = discretization * variation_z(x_values[i] + k3_x, y_values[i] + k3_y, z_values[i] + k3_z)

        x_values[i + 1] = x_values[i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0
        y_values[i + 1] = y_values[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6.0
        z_values[i + 1] = z_values[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6.0

    return x_values, y_values, z_values


def create_plots_scatters_lorenz(solution, coef_r):
    """
    Función que crea los gráficos solución para una solución de las ecuaciones de Lorenz dadas.
    :param solution: Array de arrays que contiene la solución con valor [soluciones de x, soluciones de y, soluciones de z].
    :param coef_r: Coeficiente r empleado. Es una representación para usar como string en el título.
    """
    color_primary = '#420B14'
    color_secundary = '#684812'
    color_tertiary = '#194111'
    fig = make_subplots(rows = 2, cols = 2, specs = [[{}, {"rowspan": 2}], [{}, None]])

    # Primer subplot -> izquierda superior.
    fig.add_trace(go.Scatter(x = time,
                             y = solution[0],
                             name = 'X en función del tiempo',
                             line = dict(color = color_primary, width = 2.5)),
                  row = 1, col = 1)
    # Segundo subplot -> izquierda inferior
    fig.add_trace(go.Scatter(x = time,
                             y = solution[1],
                             name = 'Y en función del tiempo',
                             line = dict(color = color_secundary, width = 2.5)),
                  row = 2, col = 1)
    # Tercer subplot -> derecha
    fig.add_trace(go.Scatter(x = solution[2],
                             y = solution[0],
                             name = 'Relación datos variables X y Z',
                             line = dict(color = color_tertiary, width = 2.5)),
                  row = 1, col = 2)
    # Modificación de los títulos de los ejes.
    fig.update_xaxes(title = dict(text = 'Tiempo', font = dict(color = color_primary)), row = 1, col = 1)
    fig.update_xaxes(title = dict(text = 'Tiempo', font = dict(color = color_primary)), row = 2, col = 1)
    fig.update_xaxes(title = dict(text = 'Z', font = dict(color = color_primary)), row = 1, col = 2)
    fig.update_yaxes(title = dict(text = 'X', font = dict(color = color_primary)), row = 1, col = 1)
    fig.update_yaxes(title = dict(text = 'Y', font = dict(color = color_primary)), row = 2, col = 1)
    fig.update_yaxes(title = dict(text = 'X', font = dict(color = color_primary)), row = 1, col = 2)

    fig.update_layout(
            width = 1800,
            height = 900,
            plot_bgcolor = '#DDD3D4',
            title = dict(font = dict(color = color_primary, size = 28),
                         text = f'Gráficos X(t), Y(t), X vs Z para r = {coef_r}',
                         x = 0.5)
    )

    fig.show()


def edit_time_or_steps(final_time = 100, number_steps = 20000):
    """
    Función que modifica o reseteas las variables globales relacionadas con número de pasos y tiempo final.
    :param final_time: Valor del tiempo final de integración a emplear.
    :param number_steps: Valor del número de pasos de integración a emplear para discretizar el tiempo.
    """
    global tf, N, discretization, time
    tf = final_time
    N = number_steps
    discretization = tf / N
    time = time = np.linspace(0, tf, N)


# Lógica principal del programa del usuario, para facilitar creación de gráficos de forma interactiva.
continuar = True
while continuar:
    print('\nOpciones:\n'
          '1 -> Crear gráficos X(t), Y(t), X vs Z para los r = [10, 22, 24.5, 100, 126.52, 400] con N = 20.000\n'
          '     para valores iniciales y tiempo final dados.\n'
          '2 -> Ver comportamiento de gráficos para distintos valores de N (cantidad de pasos de integración) dado un\n'
          '     valor de r y de valores iniciales.\n'
          '3 -> Ver comportamiento de gráficos para distintos valores de t (tiempo final) dado un valor de r y de\n'
          '     valores iniciales.\n'
          '4 -> Observar efecto de perilla dado un r = 24.4 + sin(wt), para un w, t y condiciones iniciales dadas\n'
          'q -> Salir\n')
    option = input('Ingrese opción escogida: ')

    if option == '1':
        edit_time_or_steps()
        r_values = [10, 22, 24.5, 100, 126.52, 400]
        x_0 = float(input('Ingrese valor inicial de X: '))
        y_0 = float(input('Ingrese valor inicial de Y: '))
        z_0 = float(input('Ingrese valor inicial de Z: '))
        print('Generando gráficos')

        for rval in r_values:
            sol = runge_kutta_4_lorenz(x_0, y_0, z_0, rval)
            create_plots_scatters_lorenz(sol, rval)

    elif option == '2':
        amount_steps = [300, 600, 1000, 5000, 10000, 20000]
        rval = float(input('Ingrese el valor del coeficiente r: '))
        x_0 = float(input('Ingrese valor inicial de X: '))
        y_0 = float(input('Ingrese valor inicial de Y: '))
        z_0 = float(input('Ingrese valor inicial de Z: '))
        print('Generando gráficos')

        for steps in amount_steps:
            edit_time_or_steps(number_steps = steps)
            sol = runge_kutta_4_lorenz(x_0, y_0, z_0, rval)
            create_plots_scatters_lorenz(sol, rval)

        edit_time_or_steps()

    elif option == '3':
        final_times = [1, 5, 10, 20, 30, 60, 100, 200, 300]
        rval = float(input('Ingrese el valor del coeficiente r: '))
        x_0 = float(input('Ingrese valor inicial de X: '))
        y_0 = float(input('Ingrese valor inicial de Y: '))
        z_0 = float(input('Ingrese valor inicial de Z: '))
        print('Generando gráficos')

        for time_f in final_times:
            edit_time_or_steps(final_time = time_f)
            sol = runge_kutta_4_lorenz(x_0, y_0, z_0, rval)
            create_plots_scatters_lorenz(sol, rval)
            
        edit_time_or_steps()

    elif option == '4':
        w = float(input('Ingrese el valor del coeficiente w: '))
        t = int(input('Ingrese el tiempo final a emplear: '))
        edit_time_or_steps(final_time = t)
        x_0 = float(input('Ingrese valor inicial de X: '))
        y_0 = float(input('Ingrese valor inicial de Y: '))
        z_0 = float(input('Ingrese valor inicial de Z: '))
        print('Generando gráficos')
        sol = runge_kutta_4_lorenz(x_0, y_0, z_0, coef_r = 10, manilla = True, coef_w = w)
        create_plots_scatters_lorenz(sol, f'24.4 + sin({w} * t)')
        edit_time_or_steps()

    elif option.lower() == 'q':
        print('Saliendo')
        continuar = False

    else:
        print('\nIngrese una opción válida')
