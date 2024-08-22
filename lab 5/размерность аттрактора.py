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

x_values, y_values = sol[:, 0], sol[:, 1]
min_x, max_x = np.min(x_values), np.max(x_values)
min_y, max_y = np.min(y_values), np.max(y_values)


epsilon_range = np.arange(0.1, 1.01, 0.01)
amount_of_cubes = np.zeros(len(epsilon_range))

for i, epsilon in enumerate(epsilon_range):
    num_cubes_x = int((max_x - min_x) / epsilon) + 1
    num_cubes_y = int((max_y - min_y) / epsilon) + 1
    num_cubes = num_cubes_x * num_cubes_y

    for j in range(num_cubes_x):
        for k in range(num_cubes_y):
            x_min = min_x + j * epsilon
            x_max = x_min + epsilon
            y_min = min_y + k * epsilon
            y_max = y_min + epsilon
            if np.any((x_values >= x_min) & (x_values < x_max) & (y_values >= y_min) & (y_values < y_max)):
                amount_of_cubes[i] += 1


plt.plot(np.log(1 / epsilon_range), np.log(amount_of_cubes), '-o', color='royalblue', lw=0.5, markersize=1)
plt.xlabel('$\\log\\left(\\frac{1}{\\epsilon}\\right)$')
ax.set_ylabel('$\\log\\left(N(\\epsilon)\\right)$')
plt.show()

print(np.abs(np.log(amount_of_cubes)[-1] - np.log(amount_of_cubes)[0]) / np.abs(np.log(1 / epsilon_range)[-1] - np.log(1 / epsilon_range)[0]))

# размерность = 1.2007137339640135