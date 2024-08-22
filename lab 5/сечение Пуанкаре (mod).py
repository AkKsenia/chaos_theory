import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model(y, t, α, β, ω, f):
    x, dxdt = y
    return [dxdt, -α * dxdt - β * np.exp(-x) * (1 - np.exp(-x)) + f * np.cos(ω * t)]


α = 0.8
β = 8
f = 3.07
ω_range = np.arange(0.8, 1.201, 0.001)

y0 = [3, 0]

fig, ax = plt.subplots()

for ω in ω_range:
    t = np.arange(0, 1000, 2 * np.pi / ω)
    sol = odeint(model, y0, t, args=(α, β, ω, f))
    sol = sol[100:]

    ax.plot(sol[:, 0], sol[:, 1], ls='', marker=',')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Poincare section')
plt.show()