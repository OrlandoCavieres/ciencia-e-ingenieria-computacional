import time
import concurrent.futures
import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, Piecewise, lambdify, sin, pi
from typing import *


def f(coef_r: float, x: float) -> Any:
    if 0 <= x <= 0.5:
        return coef_r * x
    elif 0.5 < x <= 1:
        return coef_r - coef_r * x


def create_cobweb_plot(function: Callable, x_initial: float, coef_r: float) -> np.ndarray:
    divisions: np.ndarray = np.linspace(0, 1, 1000)
    f_t: List[float] = [function(coef_r, xi) for xi in divisions]

    fig = go.Figure()
    primary_color: str = '#722E2E'
    secondary_color: str = '#2E2D72'

    fig.add_scatter(x = divisions, y = list(f_t), mode = 'lines', line = dict(color = secondary_color, width = 4))
    fig.add_scatter(x = divisions, y = divisions, mode = 'lines', line = dict(color = 'black', width = 3))

    x: np.ndarray = np.zeros(int(len(divisions) / 8))
    x[0] = x_initial
    x[1] = function(coef_r, x_initial)

    fig.add_scatter(x = [x[0]], y = [x[1]], mode = 'markers', marker = dict(color = primary_color, size = 12))

    for i in range(2, int(len(divisions) / 8)):
        x[i] = function(coef_r, x[i - 1])
        fig.add_scatter(x = [x[i - 2], x[i - 1]], y = [x[i - 1], x[i - 1]],
                        mode = 'lines', line = dict(color = primary_color, width = 2))
        fig.add_scatter(x = [x[i - 1], x[i - 1]], y = [x[i - 1], x[i]],
                        mode = 'lines', line = dict(color = primary_color, width = 2))

    fig.update_layout(
        xaxis = dict(title = 'X (n)', range = [-0.01, 1.01]),
        yaxis = dict(title = 'X (n + 1)', range = [-0.01, 1.01]),
        height = 900,
        plot_bgcolor = '#E2E1FF',
        width = 1000,
        title = dict(font = dict(color = secondary_color, size = 28),
                     text = f'Cobweb empleando x\u2080 = {x_initial} y r = {coef_r}',
                     x = 0.5),
        showlegend = False,
    )
    fig.show()

    return x


def create_scatter_plot_triple(datos, coef_r: float):
    primary_color: str = '#0F3E22'

    fig = go.Figure()
    fig.add_scattergl(x = list(range(len(datos[0]))), y = datos[0],
                      line = dict(color = primary_color, width = 4), name = f'x\u2080 = {0.25}')
    fig.add_scattergl(x = list(range(len(datos[1]))), y = datos[1],
                      line = dict(color = '#879645', width = 4), name = f'x\u2080 = {0.5}')
    fig.add_scattergl(x = list(range(len(datos[2]))), y = datos[2],
                      line = dict(color = '#967B45', width = 4), name = f'x\u2080 = {0.75}')
    fig.update_layout(
        height = 800,
        plot_bgcolor = '#D6E1DB',
        title = dict(font = dict(color = primary_color, size = 28),
                     text = f'Evolución temporal del mapa logístico<br>empleando r = {coef_r}', x = 0.5),
        width = 1200,
        xaxis = dict(title = 'N'),
        yaxis = dict(title = 'Xn')
    )
    fig.show()


def create_scatter_plot(x_axis_values, y_axis_values, x_label: str = 'eje x', y_label: str = 'eje y', title: str = 'Dispersión',
                        bg_color: str = '#D6E1DB', line_color: str = '#0F3E22'):
    fig = go.Figure()
    fig.add_scattergl(x = x_axis_values, y = y_axis_values, line = dict(color = line_color, width = 4))
    fig.update_layout(
        height = 800,
        plot_bgcolor = bg_color,
        showlegend = False,
        title = dict(font = dict(color = line_color, size = 28), text = title, x = 0.5),
        width = 1200,
        xaxis = dict(title = x_label),
        yaxis = dict(title = y_label)
    )
    fig.show()


def logistic_iteration_value(coef_r: float, x_initial: float, max_number_iter: int) -> float:
    while max_number_iter > 0:
        x_initial = f(coef_r, x_initial)
        max_number_iter -= 1
    return float(x_initial)


def process_bifurcation(r_min: float, r_max: float, n_r_values: int, x_initial: float,
                        max_number_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    r_values: np.ndarray = np.linspace(r_min, r_max, n_r_values)
    xn_values: np.ndarray = np.zeros_like(r_values)
    for i in range(len(r_values)):
        xn_values[i] = logistic_iteration_value(r_values[i], x_initial, max_number_iter)

    return xn_values, r_values


def create_bifurcation_plot(x_initial: float, title: str, max_number_iter: int = 1000,
                            r_min: float = 0, r_max: float = 2, n_r_values: int = 1000000, number_processes: int = 8,
                            range_x= (-0.005, 2.005), range_y = (-0.005, 1.005), zoom: bool = False):
    n: int = int(n_r_values / number_processes)
    r_process: np.ndarray = np.linspace(r_min, r_max, number_processes + 1)

    start: float = time.perf_counter()
    print('Bifurcation started...')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [[r_process[i], r_process[i + 1], n, x_initial, max_number_iter] for i in range(number_processes)]
        results = [executor.submit(process_bifurcation, *arg) for arg in args]

    xn_values: np.ndarray = np.concatenate([res.result()[0] for res in results])
    r_values: np.ndarray = np.concatenate([res.result()[1] for res in results])

    finish: float = time.perf_counter()
    print(f'Bifurcation elapsed in {finish - start:.3f} seconds')

    primary_color: str = '#0C2A54'

    fig = go.Figure()
    fig.add_pointcloud(x = r_values, y = xn_values, marker = dict(sizemin = 1.5, sizemax = 30, color = primary_color))
    fig.update_layout(
        height = 800,
        plot_bgcolor = '#E2E5E8',
        showlegend = False,
        title = dict(font = dict(color = primary_color, size = 28), text = title, x = 0.5),
        width = 1200 if not zoom else 1000,
        xaxis = dict(title = 'coeficiente r', range = range_x if not zoom else [0.98, 2.005]),
        yaxis = dict(title = 'X', range = range_y)
    )
    fig.show()


def obtain_lyapunov_values(sympy_function, base_function, x_initial: float, number_iterations: int, number_subdivisions: int,
                           min_r: float = 0, max_r: float = 2) -> Tuple[np.ndarray, np.ndarray]:
    r_values: np.ndarray = np.linspace(min_r, max_r, number_subdivisions)
    lyapunov_values: np.ndarray = np.zeros_like(r_values)

    differential = sympy_function.diff(X)
    _f_diff = lambdify([X, R], differential, "numpy")
    print(differential)

    x_vals: np.ndarray = np.zeros(number_iterations)

    start: float = time.perf_counter()
    print('Lyapunov started...')

    for i in range(len(r_values)):
        x_vals[0] = x_initial
        for j in range(1, number_iterations):
            x_vals[j] = base_function(r_values[i], x_vals[j - 1])
        lyapunov_values[i] = np.sum(np.log(abs(_f_diff(x_vals, r_values[i]))))

    lyapunov_values = lyapunov_values / number_iterations

    finish: float = time.perf_counter()
    print(f'Lyapunov elapsed in {finish - start:.3f} seconds')

    return r_values, lyapunov_values


def f2(coef_r: float, x: float) -> float:
    return coef_r * np.sin(np.pi * x)


def biseccion(r_inicial, r_final, lyapunov_inicial, lyapunov_final, tolerancia = 0.00000001):
    recta = ((R - r_inicial) / (r_final - r_inicial)) * (lyapunov_final - lyapunov_inicial) + lyapunov_inicial
    r_medio = error = (r_inicial + r_final) / 2

    while error > tolerancia:
        if recta.evalf(subs = {R: r_inicial}) * recta.evalf(subs = {R: r_medio}) > 0:
            r_inicial = r_medio
        else:
            r_final = r_medio
        r_medio_recalculado = (r_inicial + r_final) / 2
        error = abs(r_medio - r_medio_recalculado)
        r_medio = r_medio_recalculado

    return r_medio


def calculate_delta_n(lyapunov_values, r_values, tolerancia: float, acotacion_r: int = 0) -> List[float]:
    rn_values: List[float] = []

    if acotacion_r == 0:
        for i in range(len(lyapunov_values)):
            if len(rn_values) == 0 and abs(lyapunov_values[i]) <= tolerancia:
                rn_values.append(r_values[i])
            elif abs(lyapunov_values[i]) <= tolerancia and abs(r_values[i] - rn_values[-1]) >= 0.01:
                rn_values.append(r_values[i + 1])

    elif acotacion_r == 1:
        for i in range(len(lyapunov_values) - 1):
            if len(rn_values) == 0 and abs(lyapunov_values[i]) <= tolerancia:
                rn_values.append(r_values[i])
            elif abs(lyapunov_values[i]) <= tolerancia and \
                    abs(round(r_values[i], 6) - round(rn_values[-1], 6)) > 0 and \
                    lyapunov_values[i] <= 0 and lyapunov_values[i + 1] >= 0:
                rn_values.append(r_values[i + 1])

    elif acotacion_r == 2:
        for i in range(len(lyapunov_values) - 1):
            if len(rn_values) == 0 and abs(lyapunov_values[i]) <= tolerancia:
                rn_values.append(biseccion(r_values[i], r_values[i + 1], lyapunov_values[i], lyapunov_values[i + 1]))
            elif abs(lyapunov_values[i]) <= tolerancia and \
                    (abs(round(r_values[i], 4)) - abs(round(rn_values[-1], 4))) >= 0.01 and \
                    lyapunov_values[i] <= 0 and lyapunov_values[i + 1] >= 0:
                rn_values.append(biseccion(r_values[i], r_values[i + 1], lyapunov_values[i], lyapunov_values[i + 1]))

    # else:
    #     for i in range(len(lyapunov_values) - 1):
    #         if abs(lyapunov_values[i]) <= tolerancia and abs(lyapunov_values[i + 1]) <= tolerancia:
    #             rn_values.append(r_values[i])

    delta_n_values: List[float] = []
    for i in range(2, len(rn_values)):
        delta = (rn_values[i - 1] - rn_values[i - 2]) / (rn_values[i] - rn_values[i - 1])
        delta_n_values.append(delta)

    print(delta_n_values)

    return delta_n_values


if __name__ == '__main__':
    list_x_0: List[float] = [0.25, 0.5, 0.75]
    r_vals: List[float] = [0.2, 0.5, 0.8, 1, 1.2, 1.5, 1.999, 2]

    np.seterr(divide = 'ignore')

    list_x_values_structures: List[np.ndarray] = []
    # P1-A -> Creación de gráficos cobweb para los r estipulados y sus respectivas evoluciones x vs N.
    for r_value in r_vals:
        for x_0 in list_x_0:
            x_values: np.ndarray = create_cobweb_plot(f, x_0, r_value)
            list_x_values_structures.append(x_values)

    count: int = 0
    for x0 in range(0, len(list_x_values_structures) - 2, 3):
        print(len(list_x_values_structures[x0:x0 + 3]))
        create_scatter_plot_triple(list_x_values_structures[x0:x0 + 3], r_vals[count])
        count += 1

    # P1-B -> Creación del mapa de bifurcación dado un x inicial.
    for x0 in list_x_0:
        create_bifurcation_plot(x_initial = x0,
                                title = f'Diagrama de bifurcación con x\u2080 = {x0}<br>y {500000:,} puntos',
                                n_r_values = 500000,
                                number_processes = 12)

    # P1-C -> Creación del gráfico del coeficiente de Lyapunov vs r. Se obtienen los valores de lyapunov y de r usados para
    # emplear en inciso D
    R = Symbol('R')
    X = Symbol('X')
    func_p1 = Piecewise((R * X, (X >= 0) & (X <= 0.5)), (R - R * X, (X > 0.5) & (X <= 1)))

    list_r = []
    list_lyapunov_values = []

    for x0 in list_x_0 + [0.9]:
        list_r, list_lyapunov_values = obtain_lyapunov_values(func_p1, f, x0, 200, 30000)

        create_scatter_plot(x_axis_values = list_r,
                            y_axis_values = list_lyapunov_values,
                            x_label = 'coeficiente r',
                            y_label = 'λ',
                            title = f'Gráfico de valores de exponente de Lyapunov vs coeficiente r, usando x\u2080 = {x0}')
        # Lyapunov vs r no se ve afectado por el valor del x inicial

    # P1-D -> Gráficos SemiLogX, SemiLogY y LogLog para los datos obtenidos en el inciso anterior.
    log_list_r: np.ndarray = np.log(list_r)
    log_list_lyapunov: np.ndarray = np.log(list_lyapunov_values)

    create_scatter_plot(x_axis_values = log_list_r,
                        y_axis_values = list_lyapunov_values,
                        x_label = 'coeficiente r',
                        y_label = 'λ',
                        title = 'Gráfico SemiLogX de valores de exponente de Lyapunov vs coeficiente r')

    create_scatter_plot(x_axis_values = list_r,
                        y_axis_values = log_list_lyapunov,
                        x_label = 'coeficiente r',
                        y_label = 'λ',
                        title = 'Gráfico SemiLogY de valores de exponente de Lyapunov vs coeficiente r')

    create_scatter_plot(x_axis_values = log_list_r,
                        y_axis_values = log_list_lyapunov,
                        x_label = 'coeficiente r',
                        y_label = 'λ',
                        title = 'Gráfico LogLog de valores de exponente de Lyapunov vs coeficiente r')

    # P2-A -> Creación del gráfico del coeficiente de Lyapunov vs r para la función dada
    func_p2 = R * sin(pi * X)
    list2_r, list2_lyapunov_values = obtain_lyapunov_values(func_p2, f2, 0.9, 1000, 500000, max_r = 1)  # 0.9, 500.000

    create_scatter_plot(x_axis_values = list2_r,
                        y_axis_values = list2_lyapunov_values,
                        x_label = 'coeficiente r',
                        y_label = 'λ',
                        title = f'Gráfico de valores de exponente de Lyapunov vs coeficiente r, usando x\u2080 = {0.9}')

    tolerance: List[float] = [0.05, 0.03, 0.02]

    for tol in tolerance:
        delta_vals: List[float] = calculate_delta_n(list2_lyapunov_values, list2_r, tol)
        create_scatter_plot(x_axis_values = list(range(0, len(delta_vals))),
                            y_axis_values = delta_vals,
                            x_label = 'N',
                            y_label = 'delta n',
                            title = f'Gráfico de δn vs N - tolerancia = {tol} (cercania de λ a 0 y por tramos)')

        delta_vals2: List[float] = calculate_delta_n(list2_lyapunov_values, list2_r, tol, 1)
        create_scatter_plot(x_axis_values = list(range(0, len(delta_vals2))),
                            y_axis_values = delta_vals2,
                            x_label = 'N',
                            y_label = 'delta n',
                            title = f'Gráfico de δn vs N - tolerancia = {tol}, evitando repetición por ráfaga de oscilaciones')

        delta_vals3: List[float] = calculate_delta_n(list2_lyapunov_values, list2_r, tol, 2)
        create_scatter_plot(x_axis_values = list(range(0, len(delta_vals3))),
                            y_axis_values = delta_vals3,
                            x_label = 'N',
                            y_label = 'delta n',
                            title = f'Gráfico de δn vs N - tolerancia = {tol}, evitando repetición por ráfaga<br>'
                                    f'de oscilaciones y obteniendo λ = 0 mediante método de bisección')
