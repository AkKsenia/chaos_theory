import numpy as np
import matplotlib.pyplot as plt

# построение итерационной диаграммы отображения 𝑥𝑛+1 = 𝑟(𝑠𝑖𝑛(𝑥𝑛− 1.2))^2, где 𝑟 ∈ [0,3]

h = 0.001


def iterative_diagram(r, x0, amount_of_iterations):
    x = np.arange(0.0, 3.0 + h, h)

    fig, ax = plt.subplots()
    ax.plot(x, [r * (np.sin(x[i] - 1.2) ** 2) for i in range(len(x))], color='black')
    ax.plot(x, x, color='black')

    x_iter = np.zeros(amount_of_iterations + 1)
    y_iter = np.zeros(amount_of_iterations + 1)
    x_iter[0] = x0  # старт итераций происходит из точки (x0,0)
    for n in range(1, amount_of_iterations, 2):
        x_iter[n] = x_iter[n - 1]
        y_iter[n] = r * (np.sin(x_iter[n - 1] - 1.2) ** 2)  # находим 𝑓(𝑥𝑛)=𝑟(𝑠𝑖𝑛(𝑥𝑛-1− 1.2))^2
        x_iter[n + 1] = y_iter[n]
        y_iter[n + 1] = y_iter[n]

    ax.plot(x_iter, y_iter, color='red')

    ax.minorticks_on()
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('f', rotation=0, labelpad=10)
    ax.set_title('Итерационная диаграмма')


iterative_diagram(2.8, 0.5, 150)
plt.show()