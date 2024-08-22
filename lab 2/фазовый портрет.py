import matplotlib.pyplot as plt
import numpy as np


# двумерное отображение описывает динамику системы с дискретным временем:
# каждое следующее значение может быть выражено через предыдущее, поэтому оно
# может быть использовано для аппроксимации решения системы дифференциальных уравнений:
# dx/dt = 2.12 - x ** 2 - 0.3 * y
# dy/dt = x

# для изображения фазового портрета необходимо построить векторное поле направлений траекторий системы
# в каждой точке фазовой плоскости
# вектор направления или вектор, касательный к траектории в точке (x, y), задается через {dx/dt, dy/dt},
# поскольку он указывает направление изменения функций x(t) и y(t) в этой точке


def mapping(x, y):
    return 2.12 - x ** 2 - 0.3 * y, x


# строим сетку точек пространства состояний
x = np.linspace(-2.5, 2.5, 20)
y = np.linspace(4, 9, 20)

# задаем координаты начал векторов поля
X, Y = np.meshgrid(x, y)

# создаем массивы для хранения координат концов векторов поля
ΔX = np.zeros(X.shape)
ΔY = np.zeros(Y.shape)

shape1, shape2 = Y.shape

# заполняем эти массивы
for i in range(shape1):
    for j in range(shape2):
        ΔX[i, j], ΔY[i, j] = mapping(X[i, j], Y[i, j])


# моделируем траекторию в пространстве состояний для выбранных начальных условий


def Runge_Kutta_4(x, y, h):
    k0, q0 = mapping(x, y)
    k1, q1 = mapping(x + k0 * h / 2, y + q0 * h / 2)
    k2, q2 = mapping(x + k1 * h / 2, y + q1 * h / 2)
    k3, q3 = mapping(x + k2 * h, y + q2 * h)

    return x + (h / 6) * (k0 + 2 * k1 + 2 * k2 + k3), y + (h / 6) * (q0 + 2 * q1 + 2 * q2 + q3)


time = 100000
h = 0.001
solution1 = [(0, 7)]
solution2 = [(0, 6)]
solution3 = [(0, 5)]
solution4 = [(0, 4)]

for i in range(1, time):
    solution1.append(Runge_Kutta_4(solution1[i - 1][0], solution1[i - 1][1], h))
    solution2.append(Runge_Kutta_4(solution2[i - 1][0], solution2[i - 1][1], h))
    solution3.append(Runge_Kutta_4(solution3[i - 1][0], solution3[i - 1][1], h))
    solution4.append(Runge_Kutta_4(solution4[i - 1][0], solution4[i - 1][1], h))


plt.plot([solution1[i][0] for i in range(time)], [solution1[i][1] for i in range(time)], color='r', linewidth=1)
plt.plot([solution2[i][0] for i in range(time)], [solution2[i][1] for i in range(time)], color='r', linewidth=1)
plt.plot([solution3[i][0] for i in range(time)], [solution3[i][1] for i in range(time)], color='r', linewidth=1)
plt.plot([solution4[i][0] for i in range(time)], [solution4[i][1] for i in range(time)], color='r', linewidth=1)


# строим касательный вектор
plt.quiver(X, Y, ΔX, ΔY, color='black')


plt.title('Фазовый портрет', fontsize=14)
plt.xlabel('x', rotation=0, labelpad=10)
plt.ylabel('y', rotation=0, labelpad=10)
plt.xlim(-2.5, 2.5)
plt.show()

# стационарная точка - (0, 106/15), тип - центр