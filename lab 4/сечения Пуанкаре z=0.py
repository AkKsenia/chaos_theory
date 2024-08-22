import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def model(t, f):
    x, y, z = f
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x * y
    return np.array([dxdt, dydt, dzdt])


# сначала система уравнений решается на отрезке времени [0, 100]
f0 = np.array([1, 1, 0.5])

t_span = [0.0, 100.0]

sol = solve_ivp(model, t_span, f0, t_eval=np.linspace(*t_span, 100000))

# полученное решение принимается начальным условием на аттракторе
new_f0 = sol.y[:, -1]

# затем система снова интегрируется с новым начальным условием на отрезке времени [0, 10000]
new_t_span = [0.0, 10000.0]

sol = solve_ivp(model, new_t_span, new_f0, t_eval=np.linspace(*new_t_span, 1000000))

# задаем плоскость для сечения Пуанкаре
plane_normal = np.array([0, 0, 1])  # вектор нормали к плоскости
plane_point = np.array([0, 0, 0])


intersections = []

decreasing = []
increasing = []

# поиск точек пересечения с плоскостью
for i in range(1, len(sol.t)):
    prev_point = sol.y[:, i - 1]
    curr_point = sol.y[:, i]
    prev_dot = np.dot(prev_point - plane_point, plane_normal)
    curr_dot = np.dot(curr_point - plane_point, plane_normal)

    # условие проверяет, перемещается ли траектория с одной стороны плоскости
    # на другую сторону между текущей и предыдущей временными точками
    if prev_dot * curr_dot < 0:

        # вычисление времени и положения пересечения с помощью линейной интерполяции
        t_interp = sol.t[i - 1] - prev_dot * (sol.t[i] - sol.t[i - 1]) / (curr_dot - prev_dot)
        x_interp = prev_point + (curr_point - prev_point) * (t_interp - sol.t[i - 1]) / (sol.t[i] - sol.t[i - 1])
        intersections.append(x_interp)

        if curr_dot > 0:
            decreasing.append(x_interp)
        else:
            increasing.append(x_interp)


intersections = np.array(intersections)
decreasing = np.array(decreasing)
increasing = np.array(increasing)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_plane, y_plane = np.meshgrid(np.linspace(-6, 6, 500), np.linspace(-6, 6, 500))
z_plane = np.zeros_like(x_plane)
ax.plot_surface(x_plane, y_plane, z_plane, color='c', alpha=0.4, zorder=1)

ax.plot(sol.y[0], sol.y[1], sol.y[2], c='black', lw=0.05, zorder=3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# отрисуем двустороннее сечение Пуанкаре
fig, ax = plt.subplots()
ax.scatter(intersections[:, 0], intersections[:, 1], s=1, alpha=0.5, color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# отрисуем односторонне сечение Пуанкаре в направлении убывания переменной z
fig, ax = plt.subplots()
ax.scatter(decreasing[:, 0], decreasing[:, 1], s=1, alpha=0.5, color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# отрисуем односторонне сечение Пуанкаре в направлении возрастания переменной z
fig, ax = plt.subplots()
ax.scatter(increasing[:, 0], increasing[:, 1], s=1, alpha=0.5, color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()