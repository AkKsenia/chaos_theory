import matplotlib.pyplot as plt
import numpy as np


def mapping(x, y):
    return 2.12 - x * x - 0.3 * y, x


def equations_in_variations(x, y, x_wave, y_wave):
    return -2 * x * x_wave - 0.3 * y_wave, x_wave


# выбираем точку на аттракторе
x0 = 0
y0 = 6

# находим проекции касательного вектора
Δx, Δy = mapping(x0, y0)

# определяем начальные векторы возмущений
x0_wave_0 = np.array([1, 0])  # по идее ([Δx - x0, 0]), но после нормировки ([1, 0])
y0_wave_0 = np.array([0, 1])  # по идее ([0, Δy - y0]), но после нормировки ([0, 1])


def Gram_Schmidt_orthogonalization(x_prev, y_prev, x_wave_prev_0, y_wave_prev_0, s1, s2):
    x_wave, y_wave = equations_in_variations(x_prev, y_prev, x_wave_prev_0, y_wave_prev_0)

    x_wave_norm = np.sqrt(x_wave[0] ** 2 + x_wave[1] ** 2)
    x_wave_0 = x_wave / x_wave_norm

    y_wave_streak = y_wave - (y_wave * x_wave_0) * x_wave_0
    y_wave_streak_norm = np.sqrt(y_wave_streak[0] ** 2 + y_wave_streak[1] ** 2)
    y_wave_0 = y_wave_streak / y_wave_streak_norm

    s1.append(np.log(x_wave_norm))
    s2.append(np.log(y_wave_streak_norm))

    return x_wave_0, y_wave_0


def Lyapunov_exponents(M):
    s1, s2 = [], []

    x_wave_0, y_wave_0 = Gram_Schmidt_orthogonalization(x0, y0, x0_wave_0, y0_wave_0, s1, s2)
    x, y = mapping(x0, y0)
    for i in range(M - 1):
        x_wave_0, y_wave_0 = Gram_Schmidt_orthogonalization(x, y, x_wave_0, y_wave_0, s1, s2)
        x, y = mapping(x, y)

    S1, S2 = 0, 0

    for i in range(M):
        S1 += s1[i]
        S2 += s2[i]

    return S1, S2


L = [Lyapunov_exponents(i) for i in range(1, 14)]

λ1 = ([subarray[0] for subarray in L][1] - [subarray[0] for subarray in L][0])
λ2 = ([subarray[1] for subarray in L][1] - [subarray[1] for subarray in L][0])
print(λ1, λ2)

plt.plot([i for i in range(1, 14)], [subarray[0] for subarray in L], color='black')
plt.plot([i for i in range(1, 14)], [subarray[1] for subarray in L], color='black')
plt.title('К вычислению двух Ляпуновских показателей', fontsize=14)
plt.xlabel('M', rotation=0, labelpad=10)
plt.ylabel('S', rotation=0, labelpad=10)
plt.show()

# сигнатура для устойчивой неподвижной точки <-,->
