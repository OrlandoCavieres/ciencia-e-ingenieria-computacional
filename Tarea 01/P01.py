import numpy as np
import math
import plotly.graph_objects as go

# Parámetros constantes conocidos
gravity = 9.8
large = 0.5
discretization = 0.0001
times = np.arange(0, 20, discretization)


def get_angular_acceleration(angle):
    """
    Función que calcula la aceleración angular mediante la función dada, empleando un ángulo en radianes.
    :param float angle: Ángulo en radianes.
    :return: Aceleración calculada.
    """
    return -(gravity / large) * np.sin(angle)


def runge_kutta_4_pendulum(initial_angle, initial_velocity):
    """
    Función que realiza Runge Kutta 4 en las ecuaciones conocidas del enunciado con el fin de calcular
    posicion y velocidad angular en un péndulo.
    :param float initial_angle: Ángulo inicial en grados.
    :param float initial_velocity: Velocidad angular inicial.
    :return: (Array de posiciones angulares calculadas, Array de velocidades angulares calculadas)
    """
    angular_position = np.zeros(len(times))
    angular_velocity = np.zeros(len(times))

    angular_position[0] = np.radians(initial_angle)
    angular_velocity[0] = np.radians(initial_velocity)

    for i in range(len(times) - 1):
        k1_angular_position = discretization * angular_velocity[i]
        k1_angular_velocity = discretization * get_angular_acceleration(angular_position[i])

        k2_angular_position = discretization * (angular_velocity[i] + 0.5 * k1_angular_velocity)
        k2_angular_velocity = discretization * get_angular_acceleration(angular_position[i] + 0.5 * k1_angular_position)

        k3_angular_position = discretization * (angular_velocity[i] + 0.5 * k2_angular_velocity)
        k3_angular_velocity = discretization * get_angular_acceleration(angular_position[i] + 0.5 * k2_angular_position)

        k4_angular_position = discretization * (angular_velocity[i] + k3_angular_velocity)
        k4_angular_velocity = discretization * get_angular_acceleration(angular_position[i] + k3_angular_position)

        angular_position[i + 1] = angular_position[i] + (k1_angular_position + 2 * k2_angular_position +
                                                         2 * k3_angular_position + k4_angular_position) / 6.0
        angular_velocity[i + 1] = angular_velocity[i] + (k1_angular_velocity + 2 * k2_angular_velocity +
                                                         2 * k3_angular_velocity + k4_angular_velocity) / 6.0

    return angular_position, angular_velocity


def aproximate_pendulum_period(angular_velocity, initial_angle):
    """
    Función que estima el período del movimiento de un péndulo, dado por un ángulo inicial y las velocidades asociadas
    calculadas mediante Runge Kutta 4. Se establece cambios de signo en la velocidad angular, que es la derivada de la
    posición para encontrar máximos y mínimos consecutivos, obtener la diferencia de tiempos entre estos y calcular el de
    promedio de estos tiempos, lo que corresponde al período para dicho ángulo inicial. Se aproxima el infinito a 8, que es
    un valor lo suficientemente grande dado el comportamiento de la curva para obtener una representación gráfica adecuada
    después.
    :param angular_velocity: Lista de velocidades angulares calculadas mediante Runge Kutta 4 para un ángulo específico.
    :param initial_angle: Ángulo inicial en grados.
    :return: Período aproximado del movimiento del péndulo.
    """
    local_mins = []
    local_maxs = []
    for i in range(len(angular_velocity) - 1):
        if angular_velocity[i] >= 0 and angular_velocity[i + 1] <= 0:
            local_mins.append(i)
        if angular_velocity[i] <= 0 and angular_velocity[i + 1] >= 0:
            local_maxs.append(i)

    if initial_angle == 0:
        return 2 * np.radians(180) * np.sqrt(large / gravity)
    elif initial_angle == 180:
        return 8
    elif len(local_mins) >= 2 and len(local_maxs) >= 2:
        diff_time_between_maxs = times[local_maxs[1]] - times[local_maxs[0]]
        diff_time_between_mins = times[local_mins[1]] - times[local_mins[0]]
        promedio = (diff_time_between_mins + diff_time_between_maxs) / 2
        return promedio
    elif len(local_mins) >= 2 and len(local_maxs) < 2:
        return times[local_mins[1]] - times[local_mins[0]]
    elif len(local_maxs) >= 2 and len(local_mins) < 2:
        return times[local_maxs[1]] - times[local_maxs[0]]
    elif len(local_mins) == 1 and local_mins[0] == 0 and local_maxs == []:
        return 8


def create_period_scatter_plot(x, y, title, x_label, y_label):
    """
    Función que realiza un gráfico de dispersión de datos, para observar comportamiento del período.
    :param x: Valores de ángulo inicial.
    :param y: Valores de período normalizado (Período / Período pequeñas oscilaciones)
    :param title: Título del gráfico.
    :param x_label: Título del eje x.
    :param y_label: Título del eje y.
    """
    fig = go.Figure()
    fig.add_scatter(
        x = x, y = y,
        line = dict(color = '#191951', width = 5),
    )
    fig.update_layout(
        height = 800,
        width = 1000,
        title = dict(text = title, x = 0.5, font = dict(color = '#000066', size = 26)),
        xaxis = dict(title = dict(font = dict(color = '#000066'), text = x_label)),
        yaxis = dict(title = dict(font = dict(color = '#000066'), text = y_label), range = [0.8, 4.8])
    )
    fig.show()


def create_plot_with_slider(initial_values, y_data, y_label):
    """
    Función que crea un gráfico de los datos ingresados para el eje y contra el tiempo, el que fue
    establecido de forma global, para distintos valores de ángulo inicial empleando un slider para
    su visualización.
    :param initial_values: Lista de ángulos iniciales a considerar para slider.
    :param y_data: Datos del eje y, calculados previamente mediante la función runge_kutta_4_pendulum.
    :param y_label: Título del eje y.
    """
    fig = go.Figure()
    for i in range(len(initial_values)):
        fig.add_scatter(
            visible = False,
            line = dict(color = '#191951', width = 5),
            name = f'Ángulo inicial = {i}',
            x = times,
            y = y_data[i]
        )
    fig.data[1]['visible'] = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method = 'update',
            args = [{'visible': [False] * len(fig.data)},
                    {'title': f'Ángulo inicial seleccionado: {initial_values[i]}'}],
            label = f'{initial_values[i]}'
        )
        step['args'][0]['visible'][i] = True
        steps.append(step)
    sliders = [dict(
        active = 1,
        activebgcolor = '#191951',
        bgcolor = '#B5B9BD',
        currentvalue = {'prefix': 'Ángulo Inicial: '},
        pad = {'t': len(initial_values)},
        steps = steps
    )]
    fig.update_layout(
        sliders = sliders,
        plot_bgcolor = '#DFDFFF',
        xaxis = dict(title = dict(font = dict(color = '#000066'), text = 'tiempo (s)')),
        yaxis = dict(title = dict(font = dict(color = '#000066'), text = y_label))
    )
    fig.show()


# Obtener los períodos para cada ángulo inicial, y las listas de posiciones y velocidades angulares
# calculadas por Runge Kutta 4 para dicho ángulo.
periods = []
initial_angles = np.arange(0, 181, 3)
angular_positions = []
angular_velocities = []

for angle_0 in initial_angles:
    position_angular, velocity_angular = runge_kutta_4_pendulum(angle_0, initial_velocity = 0.0)
    period = aproximate_pendulum_period(velocity_angular, angle_0)
    periods.append(period)
    angular_positions.append(position_angular)
    angular_velocities.append(velocity_angular)

# Gráficos con slider para posicion y velocidad angular.
create_plot_with_slider(initial_angles, angular_positions, 'posición angular (rad)')
create_plot_with_slider(initial_angles, angular_velocities, 'velocidad angular (rad/s)')

# Imprimir periodos obtenidos y normalizarlos por el período de pequeñas oscilaciones.
print(f'Periodos: {periods}')
little_oscillations_period = 2 * math.pi * np.sqrt(large / gravity)
normalized_periods = []

for j in range(len(initial_angles)):
    normed_period = periods[j] / little_oscillations_period
    normalized_periods.append(normed_period)

# Graficar periodos normalizados vs ángulo inicial.
create_period_scatter_plot(initial_angles, normalized_periods, 'Normalización del periodo en función del ángulo inicial',
                           'Ángulo inicial (grados)', 'T/Tp')
