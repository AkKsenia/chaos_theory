from matplotlib import pyplot as plt
import numpy as np


# построить бассейны притяжения аттракторов двумерного дискретного отображения, если известно, что одним из
# аттракторов является точка на бесконечности


def mapping(x, y):
    return 2.12 - x ** 2 - 0.3 * y, x


# задаем сетку
x0_min, x0_max, y0_min, y0_max = -2.5, 2.5, -2.5, 2.5
n = 400
x_n = np.linspace(x0_min, x0_max, n)
y_n = np.linspace(y0_min, y0_max, n)

# задаем бассейны притяжения аттракторов - области в пространстве системы,
# в которых все начальные условия (точки) сходятся к данным аттракторам.

# 0 будем обозначать тот факт, что точка в результате итераций не "улетает" на бесконечность,
# 1 - тот факт, что точка в результате итераций на бесконечность, наоборот, "улетает"
pools_of_attraction = np.zeros((n, n))

# задаем число итераций, после которых должно быть понятно, "улетает" точка на бесконечность или нет
number_of_iterations = 100

for i in range(n):
    for j in range(n):
        # выбираем конкретную точку сетки
        x_tmp = x_n[i]
        y_tmp = y_n[j]
        for k in range(number_of_iterations):
            x_tmp, y_tmp = mapping(x_tmp, y_tmp)
            # условие на бесконечность весьма абстрактно:
            # если точка удаляется на расстояние, большее чем 1000 от начала координат,
            # то эта точка считается бесконечностью (необязательно задавать 1000, надо посмотреть на размер сетки)
            if x_tmp ** 2 + y_tmp ** 2 > 1000:
                pools_of_attraction[i, j] = 1
                break


plt.imshow(pools_of_attraction, extent=(x0_min, x0_max, y0_min, y0_max), cmap='viridis')
plt.colorbar()
plt.title('Бассейны притяжения аттракторов отображения')
plt.xlabel('$x_{n}$', rotation=0, labelpad=10)
plt.ylabel('$y_{n}$', rotation=0, labelpad=10)
plt.show()