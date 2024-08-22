from matplotlib import pyplot as plt
import numpy as np

# построение графика зависимости показателя Ляпунова от параметра отображенияя 𝑥𝑛+1 = 𝑟(𝑠𝑖𝑛(𝑥𝑛− 1.2))^2, где 𝑟 ∈ [0,3]

amount_of_steps = 1000


def mapping(x, r, idx):
    for i in range(amount_of_steps - 1):
        x[i + 1] = r[idx] * (np.sin(x[i] - 1.2) ** 2)
    return x


h = 0.001
r = np.arange(0.0, 3.0 + h, h)
x = np.zeros(amount_of_steps)
x[0] = 3.0

x_r = np.zeros((len(r), amount_of_steps))
for i in range(len(r)):
    x_r[i] = mapping(x, r, i)


def λ(x, r):
    s = np.zeros(len(r))
    for i in range(len(r)):
        for j in range(amount_of_steps):
            s[i] = s[i] + (1 / amount_of_steps) * np.log(np.abs(r[i] * np.sin(2 * (x[i][j] - 1.2))))
    return s


plt.plot(r, λ(x_r, r), marker=',', color='black')
plt.plot(r, [0] * len(r), color='black')
plt.xlim(0.0, 3.0)
plt.ylim(-4.0, 1.0)
plt.title('Показатели Ляпунова')
plt.xlabel('r', rotation=0)
plt.ylabel('λ', rotation=0, labelpad=10)
plt.show()