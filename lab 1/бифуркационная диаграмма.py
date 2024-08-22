from matplotlib import pyplot as plt
import numpy as np

# построение бифуркационной диаграммы отображения 𝑥𝑛+1 = 𝑟(𝑠𝑖𝑛(𝑥𝑛− 1.2))^2, где 𝑟 ∈ [0,3]

amount_of_steps = 10000
number_of_values_to_skip = 1000


def mapping(x, r, idx):
    for i in range(amount_of_steps - 1):
        x[i + 1] = r[idx] * (np.sin(x[i] - 1.2) ** 2)
    return x[number_of_values_to_skip::]


h = 0.001
r = np.arange(0.0, 3.0 + h, h)
x = np.zeros(amount_of_steps)
x[0] = 3.0

x_r = np.zeros((len(r), amount_of_steps - number_of_values_to_skip))
for i in range(len(r)):
    x_r[i] = mapping(x, r, i)

plt.plot(r, x_r, ls='', marker=',', color='black')
plt.xlim(0.0, 3.0)
plt.title('Бифуркационная диаграмма')
plt.xlabel('r', rotation=0)
plt.ylabel('$x_{n}$', rotation=0, labelpad=10)
plt.show()