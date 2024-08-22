import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def model(t, f):
    x, y, z = f
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x * y
    return np.array([dxdt, dydt, dzdt])


f0 = np.array([1, 1, 0.5])

t_span = [0.0, 10000.0]

sol = solve_ivp(model, t_span, f0, t_eval=np.linspace(*t_span, 1000000))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(sol.y[0], sol.y[1], c='black', lw=0.05)

# стационарные точки
M1 = np.array([1, 1, 0])
M2 = np.array([-1, -1, 0])

ax.scatter(M1[0], M1[1], c='r', marker='o')
ax.scatter(M2[0], M2[1], c='r', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()