import datetime
import random
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

cimport numpy as np
cimport cython

ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_cluster(int L, int N_t, makegraphic = False, calculate_mean_height = False, calculate_rugosity = False):
    cdef np.ndarray[DTYPE_t, ndim = 2] matriz = np.zeros((N_t, L), dtype = int)
    cdef int max_global = N_t - 1

    if calculate_rugosity:
        calculate_mean_height = True

    cdef np.ndarray columns_heights = np.ones(L, dtype = int) * N_t
    cdef np.ndarray height_means_in_time = np.zeros(int(N_t / L), dtype = np.float32)
    cdef np.ndarray rugosity_in_time = np.zeros_like(height_means_in_time, dtype = np.float32)

    cdef float time_start = time.perf_counter()

    cdef int i, columna, fila_actual, fila
    cdef float height_mean, rugosity

    for i in range(N_t):
        columna = random.randrange(0, L)
        fila_actual = N_t - 1

        for fila in range(max_global - 1, matriz.shape[0]):
            elemento_cercano = False

            if fila != matriz.shape[0] - 1:
                if matriz[fila + 1, columna] == 1:
                    elemento_cercano = True

                if columna == (L - 1):
                    if matriz[fila, L - 2] == 1:
                        elemento_cercano = True
                elif columna == 0:
                    if matriz[fila, columna + 1] == 1:
                        elemento_cercano = True
                else:
                    if matriz[fila, columna + 1] == 1:
                        elemento_cercano = True
                    if matriz[fila, columna - 1] == 1:
                        elemento_cercano = True

            if elemento_cercano:
                fila_actual = fila
                break

        if fila_actual < max_global:
            max_global = fila_actual
        matriz[fila_actual, columna] = 1
        columns_heights[columna] = fila_actual

        if calculate_mean_height:
            if i > 0 and i % L == 0:
                height_transformed = np.abs(np.subtract(columns_heights, N_t, dtype = int), dtype = int)
                height_mean = np.mean(height_transformed)
                height_means_in_time[i // L] = height_mean

                if calculate_rugosity:
                    diff = height_transformed - height_mean
                    rugosity = np.sqrt(np.mean(diff ** 2, dtype = np.float32), dtype = np.float32)
                    rugosity_in_time[i // L] = rugosity

    cdef float time_finish = time.perf_counter()
    print(f'Tiempo de ejecución: {time_finish - time_start} segundos')

    if makegraphic:
        fig = px.imshow(matriz[max_global:, :], color_continuous_scale = 'gray')
        fig.update_layout(
            height = 900,
            width = 600,
            title = dict(text = f'Crecimiento balístico para crecimiento de superficies<br>con L = {L} y N = {N_t}', x = 0.5)
        )
        fig.show()

    return height_means_in_time, rugosity_in_time


print('\nInciso a y b\n')
height_mean_by_time, _ = generate_cluster(250, 50000, makegraphic = True, calculate_mean_height = True)

fig2 = go.Figure()
fig2.add_scatter(x = list(range(0, len(height_mean_by_time))),
                 y = height_mean_by_time,
                 mode = 'lines',
                 line = dict(color = '#04434A', width = 4))
fig2.update_layout(
    height = 850,
    width = 1400,
    plot_bgcolor = '#C5D8DB',
    xaxis = dict(title = dict(text = 'tiempo (N/L)', font = dict(color = '#04434A')), range = [0, 201]),
    yaxis = dict(title = dict(text = 'altura promedio', font = dict(color = '#04434A'))),
    title = dict(font = dict(color = '#04434A', size = 26),
                 text = 'Altura promedio del cluster para valores de L = 250 y N = 50.000',
                 x = 0.5)
)
fig2.show()

print('\nInciso c\n')
_, rugosity_by_time = generate_cluster(500, 3000000, calculate_rugosity = True)

fig3 = go.Figure()
fig3.add_scatter(x = list(range(0, len(rugosity_by_time))),
                 y = rugosity_by_time,
                 mode = 'lines',
                 line = dict(color = '#330000', width = 4))
fig3.update_layout(
    height = 850,
    width = 1400,
    plot_bgcolor = '#ECE4DE',
    xaxis = dict(title = dict(text = 'tiempo (N/L)', font = dict(color = '#330000'))),
    yaxis = dict(title = dict(text = 'rugosidad (w)', font = dict(color = '#330000'))),
    title = dict(font = dict(color = '#330000', size = 26),
                 text = 'Rugosidad del cluster empleando L = 500 y N = 3.000.000<br>que establece t = 6.000',
                 x = 0.5)
)
fig3.show()

print('\nInciso d')
cdef float ti_d = time.perf_counter()

cdef np.ndarray sum_rugosity_L_125 = np.zeros(int(500000 / 125), dtype = np.float32)
cdef np.ndarray sum_rugosity_L_250 = np.zeros(int(1000000 / 250), dtype = np.float32)
cdef np.ndarray sum_rugosity_L_500 = np.zeros(int(2000000 / 500), dtype = np.float32)

cdef int r

for r in range(50):
    print(f'\nIteración: {r + 1}')
    _, rugosity_125 = generate_cluster(125, 500000, calculate_rugosity = True)
    _, rugosity_250 = generate_cluster(250, 1000000, calculate_rugosity = True)
    _, rugosity_500 = generate_cluster(500, 2000000, calculate_rugosity = True)
    sum_rugosity_L_125 += rugosity_125
    sum_rugosity_L_250 += rugosity_250
    sum_rugosity_L_500 += rugosity_500

mean_rugosity_L_125 = np.true_divide(sum_rugosity_L_125, 50, dtype = np.float32)
mean_rugosity_L_250 = np.true_divide(sum_rugosity_L_250, 50, dtype = np.float32)
mean_rugosity_L_500 = np.true_divide(sum_rugosity_L_500, 50, dtype = np.float32)

time_range = np.arange(0, len(mean_rugosity_L_125), dtype = int)
log_time_range = np.log10(time_range, dtype = np.float32)
log_rugosity_L_125 = np.log10(mean_rugosity_L_125, dtype = np.float32)
log_rugosity_L_125[0] = 0
log_rugosity_L_250 = np.log10(mean_rugosity_L_250, dtype = np.float32)
log_rugosity_L_250[0] = 0
log_rugosity_L_500 = np.log10(mean_rugosity_L_500, dtype = np.float32)
log_rugosity_L_500[0] = 0

cdef float tf_d = time.perf_counter()
print(f'Tiempo total empleado en calcular inciso d: {datetime.timedelta(seconds = tf_d - ti_d)}')

fig4 = go.Figure()
fig4.add_scatter(x = log_time_range,
                 y = log_rugosity_L_125,
                 mode = "lines",
                 name = "L = 125; N = 500.000",
                 line = dict(color = '#022024', width = 3.5))
fig4.add_scatter(x = log_time_range,
                 y = log_rugosity_L_250,
                 mode = "lines",
                 name = "L = 250; N = 1.000.000",
                 line = dict(color = '#935B07', width = 3.5))
fig4.add_scatter(x = log_time_range,
                 y = log_rugosity_L_500,
                 mode = "lines",
                 name = "L = 500; N = 2.000.000",
                 line = dict(color = '#530F0F', width = 3.5))
fig4.update_layout(
    height = 850,
    width = 1400,
    plot_bgcolor = "#E3D3D3",
    title = dict(font = dict(color = '#530F0F', size = 26),
                 text = 'Rugosidad del sistema como promedio de distintas realizaciones<br>(50 iteraciones)',
                 x = 0.5
    ),
    xaxis = dict(title = dict(text = 'log tiempo (N / L)', font = dict(color = '#530F0F'))),
    yaxis = dict(title = dict(text = 'log rugosidad (w)', font = dict(color = '#530F0F')))
)
fig4.show()

print('\nInciso e\n')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)
def func_first_trace(t, a):
    return a * t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)
def func_second_trace(t, b, beta):
    return b * t ** beta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.binding(True)
def func_third_trace(c):
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
def get_coeficients(log_time, log_rugosity):
    sum_logs_time = np.sum(log_time)
    sum_logs_rugosity = np.sum(log_rugosity)
    sum_logs_time_rugosity = np.sum(np.multiply(log_time, log_rugosity))
    sum_logs_time_2 = np.sum(log_time ** 2)

    coef_b = (np.multiply(sum_logs_rugosity, sum_logs_time_2) - np.multiply(sum_logs_time, sum_logs_time_rugosity)) / \
             (len(log_rugosity) * sum_logs_time_2 - (sum_logs_time ** 2))
    coef_a = (np.multiply(len(log_rugosity), sum_logs_time_rugosity) -
              np.multiply(sum_logs_time, sum_logs_rugosity)) / \
             (len(log_rugosity) * sum_logs_time_2 - (sum_logs_time ** 2))
    return coef_a, coef_b


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_r2(residual, y_real):
    ss_res = np.sum(residual ** 2)
    ss_total = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / ss_total


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_parameters_second_trace(time_real, time_log, rugosity_real, rugosity_log):
    r2_t2_actual = 1
    t_t2_actual = 10
    coef_B_actual = 0
    m_t2_actual = 0

    for pos in range(100, 4000, 25):
        m_t2, coef_B_t2 = get_coeficients(time_log[10:pos], rugosity_log[10:pos])
        residuals_t2 = rugosity_real[10:pos] - np.power(10, coef_B_t2) * np.power(time_real[10:pos], m_t2)

        r2_fit_t2 = calculate_r2(residuals_t2, rugosity_real[10:pos])

        print(f'\nRango actual tiempo: 10-{pos}\nCoeficiente B: {np.power(10, coef_B_t2)}\n'
              f'Coeficiente m: {m_t2}\nValor R2 = {r2_fit_t2}')

        if r2_fit_t2 >= 0.9:
            r2_t2_actual = r2_fit_t2
            t_t2_actual = pos
            coef_B_actual = coef_B_t2
            m_t2_actual = m_t2
        else:
            break

    print('\nDatos finales tramo 2\n')
    print(f'Rango actual tiempo: 10-{pos}\nCoeficiente B: {np.power(10, coef_B_actual)}\n'
          f'Coeficiente m: {m_t2_actual}\nValor R2 = {r2_t2_actual}')
    return t_t2_actual, m_t2_actual, coef_B_actual, r2_t2_actual


print('Primer tramo L125:\n')

m_L_125, coef_A_L_125 = get_coeficients(log_time_range[1:11], log_rugosity_L_125[1:11])
residuals_t1 = mean_rugosity_L_125[1:11] - np.power(10, coef_A_L_125) * np.power(time_range[1:11], m_L_125)

r2_fit_t1 = calculate_r2(residuals_t1, mean_rugosity_L_125[1:11])

print(f'Coeficiente A: {np.power(10, coef_A_L_125)}\nCoeficiente m: {m_L_125}\nValor R2 = {r2_fit_t1}')

print('\nSegundo tramo L125:')
t_t2_L_125, m_t2_L_125, coef_B_L_125, r2_t2_L_125 = calculate_parameters_second_trace(time_range, log_time_range,
                                                                                      mean_rugosity_L_125, log_rugosity_L_125)
print('\nTercer tramo L125:')
print(f'Coeficiente C: {np.power(10, coef_B_L_125) * np.power(t_t2_L_125, m_t2_L_125)}')


print('Primer tramo L250:\n')

m_L_250, coef_A_L_250 = get_coeficients(log_time_range[1:11], log_rugosity_L_250[1:11])
residuals_t1_L_250 = mean_rugosity_L_250[1:11] - np.power(10, coef_A_L_250) * np.power(time_range[1:11], m_L_250)

r2_fit_t1_L_250 = calculate_r2(residuals_t1_L_250, mean_rugosity_L_250[1:11])

print(f'Coeficiente A: {np.power(10, coef_A_L_250)}\nCoeficiente m: {m_L_250}\nValor R2 = {r2_fit_t1_L_250}')

print('\nSegundo tramo L250:')
t_t2_L_250, m_t2_L_250, coef_B_L_250, r2_t2_L_250 = calculate_parameters_second_trace(time_range, log_time_range,
                                                                                      mean_rugosity_L_250, log_rugosity_L_250)
print('\nTercer tramo L250:')
print(f'Coeficiente C: {np.power(10, coef_B_L_250) * np.power(t_t2_L_250, m_t2_L_250)}')


print('Primer tramo L500:\n')

m_L_500, coef_A_L_500 = get_coeficients(log_time_range[1:11], log_rugosity_L_500[1:11])
residuals_t1_L_500 = mean_rugosity_L_500[1:11] - np.power(10, coef_A_L_500) * np.power(time_range[1:11], m_L_500)

r2_fit_t1_L_500 = calculate_r2(residuals_t1_L_500, mean_rugosity_L_500[1:11])

print(f'Coeficiente A: {np.power(10, coef_A_L_500)}\nCoeficiente m: {m_L_500}\nValor R2 = {r2_fit_t1_L_500}')

print('\nSegundo tramo L500:')
t_t2_L_500, m_t2_L_500, coef_B_L_500, r2_t2_L_500 = calculate_parameters_second_trace(time_range, log_time_range,
                                                                                      mean_rugosity_L_500, log_rugosity_L_500)
print('\nTercer tramo L500:')
print(f'Coeficiente C: {np.power(10, coef_B_L_500) * np.power(t_t2_L_500, m_t2_L_500)}')


fig5 = go.Figure()
fig5.add_scatter(x = time_range[1:11],
                 y = np.power(10, coef_A_L_125) * np.power(time_range[1:11], m_L_125),
                 mode = 'lines',
                 name = 'L = 125 Tramo 1',
                 line = dict(color = '#4C8577', width = 4))
fig5.add_scatter(x = time_range[10:t_t2_L_125],
                 y = np.power(10, coef_B_L_125) * np.power(time_range[10:t_t2_L_125], m_t2_L_125),
                 mode = 'lines',
                 name = 'L = 125 tramo 2',
                 line = dict(color = '#621708', width = 4))
fig5.add_scatter(x = time_range[t_t2_L_125:],
                 y = [np.power(10, coef_B_L_125) * np.power(time_range[t_t2_L_125], m_t2_L_125) for i in range(time_range[t_t2_L_125:].size)],
                 mode = 'lines',
                 name = 'L = 125 tramo 3',
                 line = dict(color = '#38405F', width = 4))

fig5.add_scatter(x = time_range[1:11],
                 y = np.power(10, coef_A_L_250) * np.power(time_range[1:11], m_L_250),
                 mode = 'lines',
                 name = 'L = 250 Tramo 1',
                 line = dict(color = '#4C8577', width = 4, dash = 'dash'))
fig5.add_scatter(x = time_range[10:t_t2_L_250],
                 y = np.power(10, coef_B_L_250) * np.power(time_range[10:t_t2_L_250], m_t2_L_250),
                 mode = 'lines',
                 name = 'L = 250 tramo 2',
                 line = dict(color = '#621708', width = 4, dash = 'dash'))
fig5.add_scatter(x = time_range[t_t2_L_250:],
                 y = [np.power(10, coef_B_L_250) * np.power(time_range[t_t2_L_250], m_t2_L_250) for i in range(time_range[t_t2_L_250:].size)],
                 mode = 'lines',
                 name = 'L = 250 tramo 3',
                 line = dict(color = '#38405F', width = 4, dash = 'dash'))

fig5.add_scatter(x = time_range[1:11],
                 y = np.power(10, coef_A_L_500) * np.power(time_range[1:11], m_L_500),
                 mode = 'lines',
                 name = 'L = 500 Tramo 1',
                 line = dict(color = '#4C8577', width = 4, dash = 'dashdot'))
fig5.add_scatter(x = time_range[10:t_t2_L_500],
                 y = np.power(10, coef_B_L_500) * np.power(time_range[10:t_t2_L_500], m_t2_L_500),
                 mode = 'lines',
                 name = 'L = 500 tramo 2',
                 line = dict(color = '#621708', width = 4, dash = 'dashdot'))
fig5.add_scatter(x = time_range[t_t2_L_500:],
                 y = [np.power(10, coef_B_L_500) * np.power(time_range[t_t2_L_500], m_t2_L_500) for i in range(time_range[t_t2_L_500:].size)],
                 mode = 'lines',
                 name = 'L = 500 tramo 3',
                 line = dict(color = '#38405F', width = 4, dash = 'dashdot'))
fig5.update_layout(
    height = 850,
    width = 1400,
    plot_bgcolor = '#EEEEFF',
    title = dict(font = dict(color = '#38405F', size = 26),
                 text = 'Curva de aproximación por mínimos cuadrados dados los regimenes del enunciado',
                 x = 0.5),
    xaxis = dict(title = dict(text = 'tiempo (N/L)', font = dict(color = '#38405F'))),
    yaxis = dict(title = dict(text = 'rugocidad predecida (w)', font = dict(color = '#38405F')))
)
fig5.show()

table = go.Figure()
table.add_table(
    header = dict(values = ['Valor de L', 'A', 'B', 'β', 'C', 'ts']),
    cells = dict(values = [
            [125, 250, 500],
            [np.power(10, coef_A_L_125), np.power(10, coef_A_L_250), np.power(10, coef_A_L_500)],
            [np.power(10, coef_B_L_125), np.power(10, coef_B_L_250), np.power(10, coef_B_L_500)],
            [m_t2_L_125, m_t2_L_250, m_t2_L_500],
            [np.power(10, coef_B_L_125) * np.power(time_range[t_t2_L_125], m_t2_L_125),
             np.power(10, coef_B_L_250) * np.power(time_range[t_t2_L_250], m_t2_L_250),
             np.power(10, coef_B_L_500) * np.power(time_range[t_t2_L_500], m_t2_L_500)],
            [time_range[t_t2_L_125], time_range[t_t2_L_250], time_range[t_t2_L_500]]])
)
table.update_layout(
    width = 1200,
)
table.show()