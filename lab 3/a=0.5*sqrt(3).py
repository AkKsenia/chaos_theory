import matplotlib.pyplot as plt
import numpy as np


def ODE_system(a, x, y):
    return x * (x * (1 - x) - y), y * (x - a)


# задаем a
a = np.sqrt(3)/2

# строим сетку точек пространства состояний
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 3, 20)

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
solution1 = [(2, 0.05)]
solution2 = [(2, 0.25)]
solution3 = [(2, 0.5)]
solution4 = [(2, 1)]

solution5 = [(2, 1.5)]
solution6 = [(2, 2)]
solution7 = [(2, 2.5)]
solution8 = [(2, 0.75)]


for i in range(1, time):
    solution1.append(Runge_Kutta_4(a, solution1[i - 1][0], solution1[i - 1][1], h))
    solution2.append(Runge_Kutta_4(a, solution2[i - 1][0], solution2[i - 1][1], h))
    solution3.append(Runge_Kutta_4(a, solution3[i - 1][0], solution3[i - 1][1], h))
    solution4.append(Runge_Kutta_4(a, solution4[i - 1][0], solution4[i - 1][1], h))

    solution5.append(Runge_Kutta_4(a, solution5[i - 1][0], solution5[i - 1][1], h))
    solution6.append(Runge_Kutta_4(a, solution6[i - 1][0], solution6[i - 1][1], h))
    solution7.append(Runge_Kutta_4(a, solution7[i - 1][0], solution7[i - 1][1], h))
    solution8.append(Runge_Kutta_4(a, solution8[i - 1][0], solution8[i - 1][1], h))


plt.plot([solution1[i][0] for i in range(time)], [solution1[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution2[i][0] for i in range(time)], [solution2[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution3[i][0] for i in range(time)], [solution3[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution4[i][0] for i in range(time)], [solution4[i][1] for i in range(time)], color='blue', linewidth=1)

plt.plot([solution5[i][0] for i in range(time)], [solution5[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution6[i][0] for i in range(time)], [solution6[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution7[i][0] for i in range(time)], [solution7[i][1] for i in range(time)], color='blue', linewidth=1)
plt.plot([solution8[i][0] for i in range(time)], [solution8[i][1] for i in range(time)], color='blue', linewidth=1)


# строим касательный вектор
plt.quiver(X, Y, ΔX, ΔY, color='black')

x_points = np.array([0, 1, a])
y_points = np.array([0, 0, a - a ** 2])
plt.scatter(x_points, y_points, color='red')

plt.title('Фазовый портрет', fontsize=14)
plt.xlabel('x', rotation=0, labelpad=10)
plt.ylabel('y', rotation=0, labelpad=10)
plt.ylim(-0.05, 3)
plt.xlim(-0.02, 2)
plt.show()