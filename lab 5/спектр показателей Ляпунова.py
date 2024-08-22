import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def initial_model(t, g, α, β, ω, f):
    x, y, z = g

    # исходная система
    dxdt = y
    dydt = - α * y - β * np.exp(-x) * (1 - np.exp(-x)) + f * np.cos(z)
    dzdt = ω

    return [dxdt, dydt, dzdt]


def model(t, g, α, β, ω, f):
    x, y, z, δx1, δy1, δz1, δx2, δy2, δz2, δx3, δy3, δz3 = g

    # исходная система
    dxdt = y
    dydt = - α * y - β * np.exp(-x) * (1 - np.exp(-x)) + f * np.cos(z)
    dzdt = ω

    # уравнения в вариациях
    δx1δt = δy1
    δy1δt = - α * δy1 - f * np.sin(z) * δz1 - β * (- np.exp(-x) + 2 * np.exp(-2 * x)) * δx1
    δz1δt = 0
    δx2δt = δy2
    δy2δt = - α * δy2 - f * np.sin(z) * δz2 - β * (- np.exp(-x) + 2 * np.exp(-2 * x)) * δx2
    δz2δt = 0
    δx3δt = δy3
    δy3δt = - α * δy3 - f * np.sin(z) * δz3 - β * (- np.exp(-x) + 2 * np.exp(-2 * x)) * δx3
    δz3δt = 0
    return [dxdt, dydt, dzdt, δx1δt, δy1δt, δz1δt, δx2δt, δy2δt, δz2δt, δx3δt, δy3δt, δz3δt]


def Gram_Schmidt_orthogonalization(with_wave):
    # список для хранения ортогональных векторов
    ortho_vectors = []

    # ортогонализация каждого вектора относительно предыдущих
    for vector in with_wave:

        ortho_vector = vector.copy()

        for ortho_vec in ortho_vectors:
            ortho_vector -= np.dot(ortho_vector, ortho_vec) / (np.dot(ortho_vec, ortho_vec) + epsilon) * ortho_vec

        ortho_vectors.append(ortho_vector)

    return ortho_vectors


α = 0.8
β = 8
f = 3.07
ω_range = np.arange(0.8, 1.201, 0.001)

ω_values = []
s0_values = []
s1_values = []
s2_values = []

M = 100
epsilon = 1e-10
T = 1
t_span = (0, T)

for ω in ω_range:

    s = [[0] for i in range(3)]

    sol = solve_ivp(initial_model, (0, 1000), [3, 0, 0], t_eval=np.arange(0, 1000, 2 * np.pi / ω), args=(α, β, ω, f))
    g0 = [sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], 1, 0, 0, 0, 1, 0, 0, 0, 1]

    for i in range(M):
        sol = solve_ivp(model, t_span, g0, t_eval=np.linspace(*t_span, 1000), args=(α, β, ω, f))
        g0 = sol.y[:, -1]

        with_wave = Gram_Schmidt_orthogonalization([g0[3:6], g0[6:9], g0[9:12]])

        for i in range(3):
            s[i].append(np.log(np.dot(with_wave[i], with_wave[i]) + epsilon) + s[i][-1])
            with_wave[i] /= (np.dot(with_wave[i], with_wave[i]) + epsilon)  # нормируем

        # обновляем g0
        g0[3:6] = with_wave[0]
        g0[6:9] = with_wave[1]
        g0[9:12] = with_wave[2]

    ω_values.append(ω)
    s0_values.append(s[0][-1] / M * T)
    s1_values.append(s[1][-1] / M * T)
    s2_values.append(s[2][-1] / M * T)

plt.plot(ω_values, s0_values, '-o', color='royalblue', lw=0.5, label=r'$\lambda_1$', markersize=1)
plt.plot(ω_values, s1_values, '-o', color='mediumspringgreen', lw=0.5, label=r'$\lambda_2$', markersize=1)
plt.plot(ω_values, s2_values, '-o', color='red', lw=0.5, label=r'$\lambda_3$', markersize=1)

plt.xlabel('ω')
plt.ylabel('λ')
plt.legend()
plt.show()

# по формуле Каплана-Йорка при ω = 1.2:
λ1 = s0_values[-1]
λ2 = s1_values[-1]
λ3 = s2_values[-1]
print(1 + (λ1 + λ3) / np.abs(λ2))

# размерность = 1.1050997056448435