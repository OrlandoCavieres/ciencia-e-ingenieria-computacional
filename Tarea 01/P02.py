import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Parámetros constantes conocidos
gravity = 9.8
mass = 75
discretization = 0.001
times = np.arange(0, 70, discretization)


def get_acceleration(object_velocity, coef_b_arrastre):
    """
    Función que calcula la aceleración mediante la función dada, empleando un coeficiente de arrastre b y
    una velocidad del objeto en caída.
    :param float object_velocity: Velocidad asociada al objeto en caída.
    :param float coef_b_arrastre: Valor del coeficiente b de arrastre.
    :return: Aceleración calculada.
    """
    return -gravity + (coef_b_arrastre / mass) * (object_velocity ** 2)


def runge_kutta_4_fall(initial_height, initial_velocity, coef_b_arrastre):
    """
    Función que realiza Runge Kutta 4 en las ecuaciones conocidas del enunciado con el fin de calcular altura
    y velocidad asociadas.
    :param float initial_height: Altura inicial del problema.
    :param float initial_velocity: Velocidad inicial del problema.
    :param float coef_b_arrastre: Valor del coeficiente b de arrastre.
    :return: (Array de valores de alturas calculadas, Array de valores de velocidades calculadas)
    """
    heights = np.zeros(len(times))
    velocities = np.zeros(len(times))

    heights[0] = initial_height
    velocities[0] = initial_velocity

    for i in range(len(times) - 1):
        k1_height = discretization * velocities[i]
        k1_velocity = discretization * get_acceleration(velocities[i], coef_b_arrastre)

        k2_height = discretization * (velocities[i] + 0.5 * k1_velocity)
        k2_velocity = discretization * get_acceleration(velocities[i] + 0.5 * k1_velocity, coef_b_arrastre)

        k3_height = discretization * (velocities[i] + 0.5 * k2_velocity)
        k3_velocity = discretization * get_acceleration(velocities[i] + 0.5 * k2_velocity, coef_b_arrastre)

        k4_height = discretization * (velocities[i] + k3_velocity)
        k4_velocity = discretization * get_acceleration(velocities[i] + k3_velocity, coef_b_arrastre)

        heights[i + 1] = heights[i] + (k1_height + 2 * k2_height + 2 * k3_height + k4_height) / 6.0
        velocities[i + 1] = velocities[i] + (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) / 6.0

    return heights, velocities


def create_plot_with_slider(initial_values, y_data, y_label):
    """
    Función que crea un gráfico de los datos ingresados para el eje y contra el tiempo, el que fue
    establecido de forma global, para distintos valores de altura inicial empleando un slider para
    su visualización.
    :param initial_values: Array de alturas iniciales a emplear.
    :param y_data: Datos del eje y calculados previamente con la función runge_kutta_4_fall.
    :param str y_label: Título del eje y.
    """
    fig = go.Figure()
    for i in range(len(initial_values)):
        fig.add_scatter(
            visible = False,
            line = dict(color = '#003300', width = 5),
            name = f'Altura inicial = {i}',
            x = times,
            y = y_data[i]
        )
    fig.data[0]['visible'] = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method = 'update',
            args = [{'visible': [False] * len(fig.data)},
                    {'title': f'Altura inicial seleccionada: {initial_values[i]}'}],
            label = f'{initial_values[i]}'
        )
        step['args'][0]['visible'][i] = True
        steps.append(step)
    sliders = [dict(
        active = 0,
        bgcolor = '#003300',
        currentvalue = {'prefix': 'Altura inicial: '},
        pad = {'t': 35},
        steps = steps
    )]
    fig.update_layout(
        sliders = sliders,
        plot_bgcolor = '#E4F9E4',
        xaxis = dict(title = dict(font = dict(color = '#000066'), text = 'tiempo (s)')),
        yaxis = dict(title = dict(font = dict(color = '#000066'), text = y_label))
    )
    fig.show()


def func_log_fit(x, a, b, c):
    """
    Función para ajuste logarítmico.
    """
    return a * np.log(b * x) + c


def log_function_string(a, b, c):
    """
    Función que entrega un string representando un ajuste logarítmico.
    """
    return f"{a:.4f} log({b:.4f}x) + {c:.4f}"


def func_exp_fit(x, a, b, c, d):
    """
    Función para ajuste exponencial.
    """
    return a * np.exp(b * x + c) + d


def exp_function_string(a, b, c, d):
    """
    Función que entrega un string representando un ajuste exponencial.
    """
    return f"{a:.4f} e^({b:.4f}x + {c:.4f}) + {d:.4f}"


def func_rational_square_fit(x, a, b, c, d, e, f):
    """
    Función para ajuste racional cuadrático.
    """
    return (a + b * x + c * x ** 2) / (d + e * x + f * x ** 2)


def rational_square_function_string(a, b, c, d, e, f):
    """
    Función que entrega un string representando un ajuste racional cuadrático.
    """
    return f"({a:.4f} + {b:.4f}x + {c:.4f}x²)/({d:.4f} + {e:.4f}x + {f:.4f}x²)"


def func_power_fit(x, a, b):
    """
    Función para ajuste de potencia.
    """
    return a * np.power(x, b)


def power_function_string(a, b):
    """
    Función que entrega un string representando un ajuste para potencia.
    """
    return f"{a:.4f} x^({b:.4f})"


def create_scatter_plot_curve_fit(x, y, title, x_label, y_label, curve_fit_type_function, function_fit_string):
    """
    Función que realiza un gráfico de dispersión de los datos en x e y, en conjunto a un ajuste de estos
    datos dado por una función de ajuste entregada, entregando su forma final en un string y calculando R2,
    los que finalmente muestra en la leyenda del gráfico.
    :param x: Valores del eje x.
    :param y: Valores del eje y.
    :param title: Título del gráfico.
    :param x_label: Título del eje x.
    :param y_label: Título del eje y.
    :param curve_fit_type_function: Función para realizar el ajuste de los datos.
    :param function_fit_string: Función que entrega la representación en string del ajuste empleado.
    """
    # Ajuste de la curva
    coeficients, _ = curve_fit(curve_fit_type_function, x, y)

    # Calculo del valor del y del ajuste.
    y_fit = curve_fit_type_function(x, *coeficients)

    # Calculo de R2
    terminal_velocities_mean = np.mean(y)
    ecm_real = np.sum([(term_velocity - terminal_velocities_mean) ** 2 for term_velocity in y])
    ecm_predicted = np.sum([(term_velocity_pred - terminal_velocities_mean) ** 2 for term_velocity_pred in y_fit])
    r2 = ecm_predicted / ecm_real

    # Creación del gráfico
    fig = go.Figure()
    fig.add_scatter(
        x = x, y = y,
        mode = 'markers',
        marker = dict(color = '#191951', size = 10),
        name = 'datos'
    )
    fig.add_scatter(
        x = x, y = curve_fit_type_function(x, *coeficients),
        line = dict(color = '#CC6600', width = 2),
        name = f"{function_fit_string(*coeficients)}; R2 = {r2:.6f}"
    )
    fig.update_layout(
        height = 800,
        width = 1000,
        legend = dict(orientation = 'h', y = 0.98, x = 0.99, xanchor = 'right'),
        title = dict(text = title, x = 0.5, font = dict(color = '#000066', size = 26)),
        xaxis = dict(title = dict(font = dict(color = '#000066'), text = x_label)),
        yaxis = dict(title = dict(font = dict(color = '#000066'), text = y_label))
    )
    fig.show()


# Alturas inicial y listas para guardar alturas y velocidades calculadas con runge kutta 4
initial_heights = np.arange(2500, 4751, 750)
height_curves = []
velocity_curves = []

for height_0 in initial_heights:
    height_curve, velocity_curve = runge_kutta_4_fall(height_0, 0, 0.15)
    height_curves.append(height_curve)
    velocity_curves.append(velocity_curve)

# Gráficos con slider para altura y velocidad dadas alturas iniciales.
create_plot_with_slider(initial_heights, height_curves, 'altura (m)')
create_plot_with_slider(initial_heights, velocity_curves, 'velocidad (m/s)')

# Calculo de velocidades terminales dados distintos valores de b, empleando runge kutta 4.
b_coefs = np.linspace(0.01, 2, 100)
terminal_velocities = []

for b_cf in b_coefs:
    h, fall_vel = runge_kutta_4_fall(4000, 0, b_cf)
    terminal_velocities.append(abs(fall_vel[-1]))

# Gráfico loglog para estudiar comportamiento curva
plt.loglog(b_coefs, terminal_velocities)
plt.show()
# Se concluye que es logarítmica, exponencial o racional.

# Gráficos de los ajustes realizados.
create_scatter_plot_curve_fit(x = b_coefs,
                              y = terminal_velocities,
                              title = 'Ajuste de curva para velocidad terminal<br>en función de coeficiente b',
                              x_label = 'coeficiente b',
                              y_label = 'Velocidad Terminal (m/s)',
                              curve_fit_type_function = func_log_fit,
                              function_fit_string = log_function_string)

create_scatter_plot_curve_fit(x = b_coefs,
                              y = terminal_velocities,
                              title = 'Ajuste de curva para velocidad terminal<br>en función de coeficiente b',
                              x_label = 'coeficiente b',
                              y_label = 'Velocidad Terminal (m/s)',
                              curve_fit_type_function = func_exp_fit,
                              function_fit_string = exp_function_string)

create_scatter_plot_curve_fit(x = b_coefs,
                              y = terminal_velocities,
                              title = 'Ajuste de curva para velocidad terminal<br>en función de coeficiente b',
                              x_label = 'coeficiente b',
                              y_label = 'Velocidad Terminal (m/s)',
                              curve_fit_type_function = func_rational_square_fit,
                              function_fit_string = rational_square_function_string)

create_scatter_plot_curve_fit(x = b_coefs,
                              y = terminal_velocities,
                              title = 'Ajuste de curva para velocidad terminal<br>en función de coeficiente b',
                              x_label = 'coeficiente b',
                              y_label = 'Velocidad Terminal (m/s)',
                              curve_fit_type_function = func_power_fit,
                              function_fit_string = power_function_string)

# Calculando manualmente mínimos cuadrados para loglog de una potencia.
logs_b_coefs = [np.log(b) for b in b_coefs]
logs_vels = [np.log(vel) for vel in terminal_velocities]
sum_logs_b = np.sum(logs_b_coefs)
sum_logs_vel = np.sum(logs_vels)
sum_logs_b_vel = np.sum([logs_b_coefs[i] * logs_vels[i] for i in range(len(logs_vels))])
sum_logs_b2 = np.sum([b ** 2 for b in logs_b_coefs])

coef_a = ((sum_logs_b_vel / sum_logs_b) - np.mean(logs_vels)) / ((sum_logs_b2 / sum_logs_b) - np.mean(logs_b_coefs))
coef_b = np.mean(logs_vels) - coef_a * np.mean(logs_b_coefs)

vels_predicted = [np.exp(coef_b) * np.power(xi, coef_a) for xi in b_coefs]

vel_prom = np.mean(logs_vels)
ecm_base = np.sum([(vel_real - vel_prom) ** 2 for vel_real in logs_vels])
ecm_modelo_fit = np.sum([(np.log(vel_pred) - vel_prom) ** 2 for vel_pred in vels_predicted])
r2_fit = ecm_modelo_fit / ecm_base

# Creación del gráfico
fg = go.Figure()
fg.add_scatter(
    x = b_coefs, y = terminal_velocities,
    mode = 'markers',
    marker = dict(color = '#191951', size = 10),
    name = 'datos'
)
fg.add_scatter(
    x = b_coefs, y = vels_predicted,
    line = dict(color = '#CC6600', width = 2),
    name = f"{np.exp(coef_b)} x^({coef_a}) ; R2 = {r2_fit:.6f}"
)
fg.update_layout(
    height = 800,
    width = 1000,
    legend = dict(orientation = 'h', y = 0.98, x = 0.99, xanchor = 'right'),
    title = dict(text = 'Ajuste de curva para velocidad terminal<br>en función de coeficiente b (Calculado manual)',
                 x = 0.5, font = dict(color = '#000066', size = 26)),
    xaxis = dict(title = dict(font = dict(color = '#000066'), text = 'coeficiente b')),
    yaxis = dict(title = dict(font = dict(color = '#000066'), text = 'Velocidad terminal'))
)
fg.show()
