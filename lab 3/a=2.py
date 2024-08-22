import matplotlib.pyplot as plt
import numpy as np


def ODE_system(a, x, y):
    return x * (x * (1 - x) - y), y * (x - a)


# задаем a
a = 2

# строим сетку точек пространства состояний
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)

# задаем координаты начал векторов поля
X, Y = np.meshgrid(x, y)

# создаем массивы для хранения координат концов векторов поля
ΔX = np.zeros(X.shape)
ΔY = np.zeros(Y.shape)

shape1, shape2 = Y.shape

# заполняем эти массивы
for i in range(shape1):
    for j in range(shape2):
        ΔX[i, j], ΔY[i, j] = ODE_system(a, X[i, j], Y[i, j])


# моделируем траекторию в пространстве состояний для выбранных начальных условий


def Runge_Kutta_4(a, x, y, h):
    k0, q0 = ODE_system(a, x, y)
    k1, q1 = ODE_system(a, x + k0 * h / 2, y + q0 * h / 2)
    k2, q2 = ODE_system(a, x + k1 * h / 2, y + q1 * h / 2)
    k3, q3 = ODE_system(a, x + k2 * h, y + q2 * h)

    return x + (h / 6) * (k0 + 2 * k1 + 2 * k2 + k3), y + (h / 6) * (q0 + 2 * q1 + 2 * q2 + q3)


time = 100000
h = 0.001
solution1 = [(3, 0.7)]
solution2 = [(3, 1.3)]
solution3 = [(3, 3)]
solution4 = [(3, 2)]

solution5 = [(0.05, -3)]
solution6 = [(0, 3)]
solution7 = [(0.4, 3)]
solution8 = [(0.1, -3)]

solution9 = [(0.8, -3)]
solution10 = [(0.3, -3)]
solution11 = [(0.2, -3)]
solution12 = [(0, -3)]

solution13 = [(0.1, -3)]
solution14 = [(0.6, -3)]
solution15 = [(1.5, -3)]
solution16 = [(1, -3)]

solution17 = [(3, -0)]
solution18 = [(3, -0.8)]
solution19 = [(3, -1.6)]

solution20 = [(3, -0.3)]
solution21 = [(3, -2)]
solution22 = [(3, -2.5)]


for i in range(1, time):
    solution1.append(Runge_Kutta_4(a, solution1[i - 1][0], solution1[i - 1][1], h))
    solution2.append(Runge_Kutta_4(a, solution2[i - 1][0], solution2[i - 1][1], h))
    solution3.append(Runge_Kutta_4(a, solution3[i - 1][0], solution3[i - 1][1], h))
    solution4.append(Runge_Kutta_4(a, solution4[i - 1][0], solution4[i - 1][1], h))

    solution5.append(Runge_Kutta_4(a, solution5[i - 1][0], solution5[i - 1][1], h))
    solution6.append(Runge_Kutta_4(a, solution6[i - 1][0], solution6[i - 1][1], h))
    solution7.append(Runge_Kutta_4(a, solution7[i - 1][0], solution7[i - 1][1], h))
    solution8.append(Runge_Kutta_4(a, solution8[i - 1][0], solution8[i - 1][1], h))

    solution9.append(Runge_Kutta_4(a, solution9[i - 1][0], solution9[i - 1][1], h))
    solution10.append(Runge_Kutta_4(a, solution10[i - 1][0], solution10[i - 1][1], h))
    solution11.append(Runge_Kutta_4(a, solution11[i - 1][0], solution11[i - 1][1], h))
    solution12.append(Runge_Kutta_4(a, solution12[i - 1][0], solution12[i - 1][1], h))

    solution13.append(Runge_Kutta_4(a, solution13[i - 1][0], solution13[i - 1][1], h))
    solution14.append(Runge_Kutta_4(a, solution14[i - 1][0], solution14[i - 1][1], h))
    solution15.append(Runge_Kutta_4(a, solution15[i - 1][0], solution15[i - 1][1], h))
    solution16.append(Runge_Kutta_4(a, solution16[i - 1][0], solution16[i - 1][1], h))

    solution17.append(Runge_Kutta_4(a, solution17[i - 1][0], solution17[i - 1][1], h))
    solution18.append(Runge_Kutta_4(a, solution18[i - 1][0], solution18[i - 1][1], h))
    solution19.append(Runge_Kutta_4(a, solution19[i - 1][0], solution19[i - 1][1], h))

    solution20.append(Runge_Kutta_4(a, solution20[i - 1][0], solution20[i - 1][1], h))
    solution21.append(Runge_Kutta_4(a, solution21[i - 1][0], solution21[i - 1][1], h))
    solution22.append(Runge_Kutta_4(a, solution22[i - 1][0], solution22[i - 1][1], h))


plt.plot([solution1[i][0] for i in range(time)], [solution1[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution2[i][0] for i in range(time)], [solution2[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution3[i][0] for i in range(time)], [solution3[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution4[i][0] for i in range(time)], [solution4[i][1] for i in range(time)], color='blue', linewidth=1)

plt.plot([solution5[i][0] for i in range(time)], [solution5[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution6[i][0] for i in range(time)], [solution6[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution7[i][0] for i in range(time)], [solution7[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution8[i][0] for i in range(time)], [solution8[i][1] for i in range(time)], color='blue', linewidth=1)

plt.plot([solution9[i][0] for i in range(time)], [solution9[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution10[i][0] for i in range(time)], [solution10[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution11[i][0] for i in range(time)], [solution11[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution12[i][0] for i in range(time)], [solution12[i][1] for i in range(time)], color='blue', linewidth=1)

plt.plot([solution13[i][0] for i in range(time)], [solution13[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution14[i][0] for i in range(time)], [solution14[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution15[i][0] for i in range(time)], [solution15[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution16[i][0] for i in range(time)], [solution16[i][1] for i in range(time)], color='blue', linewidth=1)

plt.plot([solution17[i][0] for i in range(time)], [solution17[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution18[i][0] for i in range(time)], [solution18[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution19[i][0] for i in range(time)], [solution19[i][1] for i in range(time)], color='blue', linewidth=1)

plt.plot([solution20[i][0] for i in range(time)], [solution20[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution21[i][0] for i in range(time)], [solution21[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution22[i][0] for i in range(time)], [solution22[i][1] for i in range(time)], color='blue', linewidth=1)


# строим касательный вектор
plt.quiver(X, Y, ΔX, ΔY, color='black')

x_points = np.array([0, 1, a])
y_points = np.array([0, 0, a - a ** 2])
plt.scatter(x_points, y_points, color='red')

plt.title('Фазовый портрет', fontsize=14)
plt.xlabel('x', rotation=0, labelpad=10)
plt.ylabel('y', rotation=0, labelpad=10)
plt.ylim(-3, 3)
plt.xlim(-0.02, 3)
plt.show()