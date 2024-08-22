import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model(y, t, α, β, ω, f):
    x, dxdt = y
    return [dxdt, -α * dxdt - β * np.exp(-x) * (1 - np.exp(-x)) + f * np.cos(ω * t)]


α = 0.8
β = 8
f = 3.07
ω = 1.2

y0 = [3, 0]

fig, ax = plt.subplots()
t = np.arange(0, 100000, 2 * np.pi / ω)
sol = odeint(model, y0, t, args=(α, β, ω, f))
sol = sol[100:]

epsilon = 0.4
x_values, y_values = sol[:, 0], sol[:, 1]
min_x, max_x = np.min(x_values), np.max(x_values)
min_y, max_y = np.min(y_values), np.max(y_values)
num_cubes_x = int((max_x - min_x) / epsilon) + 1
num_cubes_y = int((max_y - min_y) / epsilon) + 1
num_cubes = num_cubes_x * num_cubes_y

amount_of_cubes = 0

for i in range(num_cubes_x):
    for j in range(num_cubes_y):
        x_min = min_x + i * epsilon
        x_max = x_min + epsilon
        y_min = min_y + j * epsilon
        y_max = y_min + epsilon
        if np.any((x_values >= x_min) & (x_values < x_max) & (y_values >= y_min) & (y_values < y_max)):
            amount_of_cubes += 1
            ax.fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max], color='royalblue', alpha=0.2)


ax.plot(sol[:, 0], sol[:, 1], ls='', marker=',', color='black')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Poincare section with covering squares (ω = 1.2)')
plt.show()

print(amount_of_cubes)