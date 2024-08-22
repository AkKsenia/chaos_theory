import numpy as np
import matplotlib.pyplot as plt


a = 2


def X(t, x, y):
    return x * (x * (1 - x) - y)


def Y(t, x, y):
    return y * (x - a)


# задание интервала
t_beginning = 0
t_ending = 50

# шаг
Ꚍ = 0.01
# число узлов сетки
n = 1 + (t_ending - t_beginning) / Ꚍ

# узлы
t = np.linspace(t_beginning, t_ending, int(n))


# метод Рунге-Кутты (4-ый порядок точности)


def Runge_Kutta(t):
    x = [1]
    y = [1]

    for i in range(len(t) - 1):
        q0 = Y(t[i], x[i], y[i])
        k0 = X(t[i], x[i], y[i])
        q1 = Y(t[i] + Ꚍ / 2, x[i] + k0 * Ꚍ / 2, y[i] + q0 * Ꚍ / 2)
        k1 = X(t[i] + Ꚍ / 2, x[i] + k0 * Ꚍ / 2, y[i] + q0 * Ꚍ / 2)
        q2 = Y(t[i] + Ꚍ / 2, x[i] + k1 * Ꚍ / 2, y[i] + q1 * Ꚍ / 2)
        k2 = X(t[i] + Ꚍ / 2, x[i] + k1 * Ꚍ / 2, y[i] + q1 * Ꚍ / 2)
        q3 = Y(t[i] + Ꚍ, x[i] + k2 * Ꚍ, y[i] + q2 * Ꚍ)
        k3 = X(t[i] + Ꚍ, x[i] + k2 * Ꚍ, y[i] + q2 * Ꚍ)

        x.append(x[i] + (Ꚍ / 6) * (k0 + 2 * k1 + 2 * k2 + k3))
        y.append(y[i] + (Ꚍ / 6) * (q0 + 2 * q1 + 2 * q2 + q3))

    return x, y


sp1 = plt.subplot(121)
plt.plot(t, Runge_Kutta(t)[0], color='blue', lw=1)
plt.xlabel('t', rotation=0, labelpad=10)
plt.ylabel('x', rotation=0, labelpad=10)
plt.grid(True)

sp2 = plt.subplot(122)
plt.plot(t, Runge_Kutta(t)[1], color='red', lw=1)
plt.xlabel('t', rotation=0, labelpad=10)
plt.ylabel('y', rotation=0, labelpad=10)

plt.grid(True)
plt.subplots_adjust(wspace=0.3)

plt.show()