import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Condiciones iniciales
rabbits_0 = 20
foxes_0 = 10

# Discretización del tiempo
tf = 100
N = 1000
discretization = tf / N
time = np.linspace(0, tf, N)


# Funciones diferenciales para cambios en la población de conejos y de zorros
def rabbits_change(rabbits, foxes, coef_a, coef_b):
    return coef_a * rabbits - coef_b * rabbits * foxes


def foxes_change(rabbits, foxes, coef_c, coef_d):
    return -coef_c * foxes + coef_d * rabbits * foxes


# Método para cálculo numérico de ecuaciones diferenciales
def runge_kutta_4_lokta_volterra(initial_rabbits, initial_foxes, coef_a, coef_b, coef_c, coef_d):
    """
    Función que realiza Runge Kutta 4 en las ecuaciones conocidas de Lokta Volterra para calcular los cambios en las
    poblaciones de conejos y zorros en el tiempo.
    :param float initial_rabbits: Cantidad inicial de conejos.
    :param float initial_foxes: Cantidad inicial de zorros.
    :param float coef_a: Valor del coeficiente a en la ecuación de Lokta Volterra.
    :param float coef_b: Valor del coeficiente b en la ecuación de Lokta Volterra.
    :param float coef_c: Valor del coeficiente c en la ecuación de Lokta Volterra.
    :param float coef_d: Valor del coeficiente d en la ecuación de Lokta Volterra.
    :return: (Array de cantidades de conejos en el tiempo, Array de cantidades de zorros en el tiempo)
    """
    crec_rabbits = np.zeros(len(time))
    crec_foxes = np.zeros(len(time))

    crec_rabbits[0] = initial_rabbits
    crec_foxes[0] = initial_foxes

    for i in range(len(time) - 1):
        k1_rabbits = discretization * rabbits_change(crec_rabbits[i], crec_foxes[i], coef_a, coef_b)
        k1_foxes = discretization * foxes_change(crec_rabbits[i], crec_foxes[i], coef_c, coef_d)

        k2_rabbits = discretization * rabbits_change(crec_rabbits[i] + 0.5 * k1_rabbits,
                                                     crec_foxes[i] + 0.5 * k1_foxes,
                                                     coef_a, coef_b)
        k2_foxes = discretization * foxes_change(crec_rabbits[i] + 0.5 * k1_rabbits,
                                                 crec_foxes[i] + 0.5 * k1_foxes,
                                                 coef_c, coef_d)

        k3_rabbits = discretization * rabbits_change(crec_rabbits[i] + 0.5 * k2_rabbits,
                                                     crec_foxes[i] + 0.5 * k2_foxes,
                                                     coef_a, coef_b)
        k3_foxes = discretization * foxes_change(crec_rabbits[i] + 0.5 * k2_rabbits,
                                                 crec_foxes[i] + 0.5 * k2_foxes,
                                                 coef_c, coef_d)

        k4_rabbits = discretization * rabbits_change(crec_rabbits[i] + k3_rabbits,
                                                     crec_foxes[i] + k3_foxes,
                                                     coef_a, coef_b)
        k4_foxes = discretization * foxes_change(crec_rabbits[i] + k3_rabbits,
                                                 crec_foxes[i] + k3_foxes,
                                                 coef_c, coef_d)

        crec_rabbits[i + 1] = crec_rabbits[i] + (k1_rabbits + 2 * k2_rabbits + 2 * k3_rabbits + k4_rabbits) / 6.0
        crec_foxes[i + 1] = crec_foxes[i] + (k1_foxes + 2 * k2_foxes + 2 * k3_foxes + k4_foxes) / 6.0

    return crec_rabbits, crec_foxes


def create_plot_quantity_vs_time(quantity_prey, quantity_predator, time_discrete, name_prey, name_predator):
    """
    Función que crea un gráfico de tipo scatter line de cantidad de las poblaciones versus tiempo de integración.
    :param quantity_prey: Array que contiene la solución de las ecuaciones para la cantidad de presa.
    :param quantity_predator: Array que contiene la solución de las ecuaciones para la cantidad de depredador.
    :param time_discrete: Array de tiempo discretizado.
    :param string name_prey: Nombre de la presa del modelo.
    :param string name_predator: Nombre del depredador del modelo.
    :return:
    """
    grid_color = '#B6B6B6'
    primary_color = '#0E131F'
    fig = go.Figure()
    fig.add_scatter(x = time_discrete, y = quantity_prey, name = name_prey,
                    line = dict(color = '#7A542E', width = 3))
    fig.add_scatter(x = time_discrete, y = quantity_predator, name = name_predator,
                    line = dict(color = primary_color, width = 3))
    fig.update_layout(
            plot_bgcolor = '#F5F0F6',
            width = 1200,
            height = 900,
            title = dict(font = dict(color = primary_color, size = 26),
                         text = f'Evolución temporal de poblaciones {name_predator}/{name_prey}',
                         x = 0.5),
            xaxis = dict(title = 'Tiempo', gridcolor = grid_color),
            yaxis = dict(title = 'Cantidad', gridcolor = grid_color)
    )
    fig.show()


def create_plot_or_quiver_by_solutions(list_solutions, coef_a, coef_b, coef_c, coef_d, is_quiver, scale_arrow_quiver = 2.0):
    """
    Función que crea un scatter plot o un quiver plot según necesidad.
    :param list_solutions: Lista que contiene elementos Array con una o mas soluciones del sistema de ecuaciones.
    :param float coef_a: Valor del parámetro a
    :param float coef_b: Valor del parámetro b
    :param float coef_c: Valor del parámetro c
    :param float coef_d: Valor del parámetro d
    :param bool is_quiver: Determina si el gráfico que se necesita es o no un gráfico tipo quiver.
    :param scale_arrow_quiver: Determina la escala o tamaño de la flecha en un gráfico quiver.
    """
    rabbits_max = np.max([np.max(sol[0]) for sol in list_solutions]) * 1.05
    foxes_max = np.max([np.max(sol[1]) for sol in list_solutions]) * 1.05

    rabbits_discrete = np.linspace(0, rabbits_max, 20)
    foxes_discrete = np.linspace(0, foxes_max, 20)

    rabbits_mesh, foxes_mesh = np.meshgrid(rabbits_discrete, foxes_discrete)
    component_vector_rabbits = rabbits_change(rabbits_mesh, foxes_mesh, coef_a, coef_b)
    component_vector_foxes = foxes_change(rabbits_mesh, foxes_mesh, coef_c, coef_d)
    norm = np.sqrt(component_vector_rabbits ** 2 + component_vector_foxes ** 2)
    u = component_vector_rabbits / norm
    v = component_vector_foxes / norm

    fig = go.Figure()
    if is_quiver:
        fig = ff.create_quiver(rabbits_mesh, foxes_mesh, u, v, scale = scale_arrow_quiver, arrow_scale = 0.5,
                               name = 'Quiver', line = dict(color = 'black'))

    if coef_a != 0 and coef_b != 0 and coef_c != 0 and coef_d != 0 and is_quiver:
        fig.add_scatter(x = [0],
                        y = [0],
                        mode = 'markers',
                        name = 'Punto Inestable',
                        marker = dict(color = 'red', size = 15))
        fig.add_scatter(x = [coef_c / coef_d],
                        y = [coef_a / coef_b],
                        mode = 'markers',
                        name = 'Punto Centro',
                        marker = dict(color = 'darkgreen', size = 15))

    for sol in range(len(list_solutions)):
        fig.add_scatter(x = list_solutions[sol][0], y = list_solutions[sol][1],
                        mode = 'lines', name = f'Solución {sol + 1}',
                        line = dict(color = '#15184E', width = 5))
        fig.add_scatter(x = [list_solutions[sol][0][0]], y = [list_solutions[sol][1][0]],
                        mode = 'markers', name = f'Condición Inicial {sol + 1}',
                        marker = dict(color = '#15184E', size = 15))

    fig.update_layout(
            plot_bgcolor = '#DEE2F3',
            width = 1000,
            height = 900,
            xaxis = dict(range = [-0.5, rabbits_max], title = 'Conejos'),
            yaxis = dict(range = [-0.5, foxes_max], title = 'Zorros')
    )
    fig.show()


def create_countour_plot_for_solution(solution, coef_a, coef_b, coef_c, coef_d):
    """
    Función que permite crear un gráfico de contornos para observar soluciones de un sistema de ecuaciones diferencial,
    a través del valor constante de órbita.
    :param solution: Array que contiene la solución inicial del sistema de ecuaciones.
    :param float coef_a: Valor del parámetro a.
    :param float coef_b: Valor del parámetro b.
    :param float coef_c: Valor del parámetro c.
    :param float coef_d: Valor del parámetro d.
    """
    rabbits_max = np.max(solution[0]) * 1.5
    foxes_max = np.max(solution[1]) * 1.5
    rabbits_amount = np.linspace(0, rabbits_max, 100)
    foxes_amount = np.linspace(0, foxes_max, 100)
    rabbits_mesh, foxes_mesh = np.meshgrid(rabbits_amount, foxes_amount)

    fig = go.Figure()
    fig.add_contour(
            x = rabbits_amount,
            y = foxes_amount,
            z = calcular_cantidad_constante(rabbits_mesh, foxes_mesh, coef_a, coef_b, coef_c, coef_d),
            colorscale = [[0, '#0E0B0B'], [0.4, '#222277'], [0.75, '#A2AEFF'], [1, '#F4FFFF']],
            line_smoothing = 0.85,
            contours = dict(start = -10, end = 4, size = 0.2),
            line = dict(width = 2),
            name = 'Contours'
    )
    fig.add_scatter(
            x = [coef_c / coef_d],
            y = [coef_a / coef_b],
            mode = 'markers',
            marker = dict(color = 'black', size = 15),
            name = 'Punto centro'
    )
    fig.update_layout(
            width = 1000,
            height = 800,
            xaxis = dict(range = [0, rabbits_max], title = 'Conejos'),
            yaxis = dict(range = [0, foxes_max], title = 'Zorros')
    )
    fig.show()


def calcular_cantidad_constante(rabbits, foxes, coef_a, coef_b, coef_c, coef_d):
    """
    Función que permite calcular el valor constante dentro de una órbita solución del punto centro.
    :param rabbits: Cantidad de conejos.
    :param foxes: Cantidad de zorros.
    :param float coef_a: Valor del parámetro a.
    :param float coef_b: Valor del parámetro b.
    :param float coef_c: Valor del parámetro c.
    :param flaot coef_d: Valor del parámetro d.
    :return: Valor constante de la órbita.
    """
    return coef_a * np.log(foxes) - coef_b * foxes + coef_c * np.log(rabbits) - coef_d * rabbits


# Parámetros constantes a emplear y valores para test de dichos parámetros.
coefs = [0.5, 0.012, 1, 0.033]
coef_testing_values = [0, 0.5, 1]

# Lógica del programa para facilitar la creación de gráficos y soluciones.
continuar = True
while continuar:
    print(f'\nOpciones:\n'
          f'1 -> Obtener gráficos para cambios en el valor del coeficiente a, con el resto de coeficientes en 0\n'
          f'2 -> Obtener gráficos para cambios en el valor del coeficiente b, con el resto de coeficientes en 0\n'
          f'3 -> Obtener gráficos para cambios en el valor del coeficiente c, con el resto de coeficientes en 0\n'
          f'4 -> Obtener gráficos para cambios en el valor del coeficiente d, con el resto de coeficientes en 0\n'
          f'5 -> Obtener gráfico de evolución temporal de las poblaciones de conejos y zorros en base a sus\n'
          f'     cantidades iniciales\n'
          f'6 -> Obtener gráfico tipo quiver para una o más soluciones a partir de valores iniciales\n'
          f'7 -> Obtener gráfico de contornos para la solución a partir de valores iniciales\n'
          f'8 -> Cambiar tiempo máximo\n'
          f'9 -> Cambiar número de divisiones del tiempo\n'
          f'q -> Quit\n')
    opcion = input('Ingrese una opción: ')

    if opcion == '1':
        for a in coef_testing_values:
            solucion = runge_kutta_4_lokta_volterra(rabbits_0, foxes_0, a, 0, 0, 0)
            create_plot_or_quiver_by_solutions([solucion], a, 0, 0, 0, False)

    elif opcion == '2':
        for b in coef_testing_values:
            solucion = runge_kutta_4_lokta_volterra(rabbits_0, foxes_0, 0, b, 0, 0)
            create_plot_or_quiver_by_solutions([solucion], 0, b, 0, 0, False)

    elif opcion == '3':
        for c in coef_testing_values:
            solucion = runge_kutta_4_lokta_volterra(rabbits_0, foxes_0, 0, 0, c, 0)
            create_plot_or_quiver_by_solutions([solucion], 0, 0, c, 0, False)

    elif opcion == '4':
        for d in coef_testing_values:
            solucion = runge_kutta_4_lokta_volterra(rabbits_0, foxes_0, 0, 0, 0, d)
            create_plot_or_quiver_by_solutions([solucion], 0, 0, 0, d, False)

    elif opcion == '5':
        rabbits_initial = int(input('Ingrese el número de conejos inicial: '))
        foxes_initial = int(input('Ingrese el número de zorros inicial: '))
        solucion = runge_kutta_4_lokta_volterra(rabbits_initial, foxes_initial, *coefs)
        create_plot_quantity_vs_time(solucion[0], solucion[1], time, 'Conejos', 'Zorros')

    elif opcion == '6':
        initials = []
        pedir_iniciales = True
        while pedir_iniciales:
            x_0 = int(input('Ingrese valor inicial de conejos: '))
            y_0 = int(input('Ingrese valor inicial de zorros: '))
            initials.append([x_0, y_0])
            agregar_mas = input('Desea agregar más (s = si, else = no)? ')
            if agregar_mas.lower() != 's':
                pedir_iniciales = False
        soluciones = []
        for init_values in initials:
            solucion = runge_kutta_4_lokta_volterra(*init_values, *coefs)
            soluciones.append(solucion)
        es_quiver_plot = input('Es un gráfico tipo quiver (s = si, else = no)? ')
        if es_quiver_plot.lower() == 's':
            scale_arrow = float(input('Ingrese el tamaño en escala para las flechas del quiver (mínimo = 0.1): '))
            create_plot_or_quiver_by_solutions(soluciones, *coefs, True, scale_arrow_quiver = scale_arrow)
        else:
            create_plot_or_quiver_by_solutions(soluciones, *coefs, False)

    elif opcion == '7':
        rabbits_initial = int(input('Ingrese el número de conejos inicial: '))
        foxes_initial = int(input('Ingrese el número de zorros inicial: '))
        solucion = runge_kutta_4_lokta_volterra(rabbits_initial, foxes_initial, *coefs)
        create_countour_plot_for_solution(solucion, *coefs)

    elif opcion == '8':
        tf = int(input('Ingrese nuevo tiempo final (entero): '))
        discretization = tf / N
        time = np.linspace(0, tf, N)

    elif opcion == '9':
        N = int(input('Ingrese el nuevo número de divisiones (N) con el que discretizar el tiempo: '))
        discretization = tf / N
        time = np.linspace(0, tf, N)

    elif opcion.lower() == 'q':
        print('Saliendo')
        continuar = False

    else:
        print('Opción no válida\n')
