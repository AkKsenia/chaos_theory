from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# построение обриты отображения 𝑥𝑛+1 = 𝑟(𝑠𝑖𝑛(𝑥𝑛− 1.2))^2, где 𝑟 ∈ [0,3]

amount_of_steps = 71


def mapping(x_n, r):
    for i in range(amount_of_steps - 1):
        x_n[i + 1] = r * (np.sin(x_n[i] - 1.2) ** 2)
    return x_n


n = np.arange(0.0, amount_of_steps)
x_n = np.zeros(amount_of_steps)
x_n[0] = 3.0

r = 0.0

fig, ax = plt.subplots()
line, = ax.plot(n, mapping(x_n, r), marker='.', color='black', linewidth=1)
ax.set_xlabel('n', rotation=0)
ax.set_ylabel('$x_{n}$', rotation=0, labelpad=10)
ax.set_title('Орбита отображения при r ∈ [0,3]')

fig.subplots_adjust(left=0.1, bottom=0.25)

r_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
r_slider = Slider(
    ax=r_ax,
    label='r',
    valmin=0.0,
    valmax=3.0,
    valinit=r,
    color='black',
    initcolor='black'
)


def update(val):
    line.set_ydata(mapping(x_n, r_slider.val))
    fig.canvas.draw_idle()


r_slider.on_changed(update)

plt.show()