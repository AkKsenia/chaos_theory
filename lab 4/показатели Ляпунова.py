import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def model(t, f):
    x, y, z, δx1, δy1, δz1, δx2, δy2, δz2, δx3, δy3, δz3 = f

    # исходная система
    dxdt = y * z
    dydt = x - y
    dzdt = 1 - x * y

    # уравнения в вариациях
    δx1δt = δy1 * z + y * δz1
    δy1δt = δx1 - δy1
    δz1δt = - δx1 * y - x * δy1
    δx2δt = δy2 * z + y * δz2
    δy2δt = δx2 - δy2
    δz2δt = - δx2 * y - x * δy2
    δx3δt = δy3 * z + y * δz3
    δy3δt = δx3 - δy3
    δz3δt = - δx3 * y - x * δy3
    return [dxdt, dydt, dzdt, δx1δt, δy1δt, δz1δt, δx2δt, δy2δt, δz2δt, δx3δt, δy3δt, δz3δt]


def Gram_Schmidt_orthogonalization(with_wave):
    # список для хранения ортогональных векторов
    ortho_vectors = []

    # ортогонализация каждого вектора относительно предыдущих
    for vector in with_wave:

        ortho_vector = vector.copy()

        for ortho_vec in ortho_vectors:
            ortho_vector -= np.dot(ortho_vector, ortho_vec) / np.dot(ortho_vec, ortho_vec) * ortho_vec

        ortho_vectors.append(ortho_vector)

    return ortho_vectors


T = 1
t_span = (0, T)
M = 10000
s = [[0] for i in range(3)]
f0 = [1, 1, 0.5, 1, 0, 0, 0, 1, 0, 0, 0, 1]

for i in range(M):

    sol = solve_ivp(model, t_span, f0, t_eval=np.linspace(*t_span, 10000))
    f0 = sol.y[:, -1]

    with_wave = Gram_Schmidt_orthogonalization([f0[3:6], f0[6:9], f0[9:12]])

    for i in range(3):
        s[i].append(np.log(np.dot(with_wave[i], with_wave[i])) + s[i][-1])
        with_wave[i] /= np.dot(with_wave[i], with_wave[i])  # нормируем

    # обновляем f0
    f0[3:6] = with_wave[0]
    f0[6:9] = with_wave[1]
    f0[9:12] = with_wave[2]


m = np.linspace(0, M, M + 1)

plt.plot(m, np.array(s[0]) / M * T, label='S1', color='royalblue', lw=0.5)
plt.plot(m, np.array(s[1]) / M * T, label='S2', color='orange', lw=0.5)
plt.plot(m, np.array(s[2]) / M * T, label='S3', color='gold', lw=0.5)

plt.xlabel('M')
plt.ylabel('S')
plt.legend()
plt.show()

print(s[0][-1] / M * T, s[1][-1] / M * T, s[2][-1] / M * T)