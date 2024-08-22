import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def model(t, f):
    x, y, z = f
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x * y
    return np.array([dxdt, dydt, dzdt])


f0_1 = np.array([1, 1, 0.5])
f0_2 = np.array([1.01, 1.01, 0.5])

t_span = [0.0, 1000.0]

t = np.linspace(0, 1000, 1000000)

sol_1 = solve_ivp(model, t_span, f0_1, t_eval=np.linspace(*t_span, 1000000))
sol_2 = solve_ivp(model, t_span, f0_2, t_eval=np.linspace(*t_span, 1000000))

diff = np.sqrt(np.sum((sol_1.y - sol_2.y)**2, axis=1))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(t, sol_1.y[0], c='royalblue', lw=0.5)
ax.plot(t, sol_2.y[0], c='mediumspringgreen', lw=0.5)
ax.set_xlabel('t')
ax.set_ylabel('x')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(t, sol_1.y[1], c='royalblue', lw=0.5)
ax.plot(t, sol_2.y[1], c='mediumspringgreen', lw=0.5)
ax.set_xlabel('t')
ax.set_ylabel('y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(t, sol_1.y[2], c='royalblue', lw=0.5)
ax.plot(t, sol_2.y[2], c='mediumspringgreen', lw=0.5)
ax.set_xlabel('t')
ax.set_ylabel('z')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

norm = np.sqrt(np.sqrt((sol_1.y[0] - sol_2.y[0]) ** 2 + (sol_1.y[1] - sol_2.y[1]) ** 2 + (sol_1.y[2] - sol_2.y[2]) ** 2))
ax.plot(t, norm, c='royalblue', lw=0.5)
ax.set_xlabel('t')
ax.set_ylabel('norm')
plt.show()