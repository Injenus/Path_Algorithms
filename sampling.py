"""
Вход - numpy массив точек карты
Выход - numpy массив квадратиков карты

"""
input_file = 'Map_np_data.npy'
ouput_file = 'Sampled_map'

import numpy as np
import matplotlib.pyplot as plt
import time

dl = 0.15  # длина ребра квадратика
koef_d = 3  # кол-во голосов для признания препятсвия
tolerance = 0.03  # допуск редьюса

raw_map = np.load(input_file)
min_xy = np.amin(raw_map, axis=0)
max_xy = np.amax(raw_map, axis=0)
# [min_xy[0], max_xy[1]] - левая верхняя точка
len_xy = np.array([max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]])
count_xy = np.ceil(np.divide(len_xy, dl))


def reduce_by_axis(arr, toler, axis='y', direc=1):
    if axis == 'y':
        arr = arr[arr[:, 0].argsort()[::direc]]
    elif axis == 'x':
        arr = arr[arr[:, 1].argsort()[::direc]]
    reduce_arr = np.ndarray(shape=(0, 2), dtype=float)
    reduce_arr = np.append(reduce_arr, [[arr[0][0], arr[0][1]]],
                           axis=0)
    for i in range(len(arr)):
        if ((arr[i][0] - reduce_arr[-1][0]) ** 2 + (
                arr[i][1] - reduce_arr[-1][1]) ** 2) ** 0.5 > toler:
            reduce_arr = np.append(reduce_arr, [[arr[i][0], arr[i][1]]],
                                   axis=0)
    return reduce_arr


sampled_map = np.zeros((1, int(count_xy[0]), 3))
raw_map = reduce_by_axis(raw_map, tolerance, 'y')
raw_map = reduce_by_axis(raw_map, tolerance, 'x')

for i in range(int(count_xy[1])):
    row_map = np.ndarray(shape=(0, 3), dtype=float)
    #print('создание/ строка {}'.format(i))
    for j in range(int(count_xy[0])):
        # xc, yc = min_xy[0] + j * dl + dl / 2, max_xy[1] - i * dl + dl / 2
        xc, yc = j * dl + dl / 2, -i * dl + dl / 2
        row_map = np.append(row_map, [[xc, yc, 0]], axis=0)
    sampled_map = np.concatenate(
        (sampled_map, np.reshape(row_map, (-1, int(count_xy[0]), 3))), axis=0)
sampled_map = np.delete(sampled_map, 0, axis=0)
print(sampled_map.shape)
raw_map = raw_map[raw_map[:, 1].argsort()[::1]]
for i in range(int(count_xy[1])):
    print('стр.{}(~{}%)'.format(i, round((i + 1) / int(count_xy[1]) * 100, 1)))
    for j in range(int(count_xy[0])):
        xc, yc = sampled_map[i][j][0]+min_xy[0], sampled_map[i][j][1]+max_xy[1]
        num = 0
        k = 0
        while k < len(raw_map):
            p_xy = raw_map[k]
            if xc + dl / 2 > p_xy[0] >= xc - dl / 2 and yc + dl / 2 >= p_xy[
                1] > yc - dl / 2:
                num += 1
                raw_map = np.delete(raw_map, k, axis=0)
                k -= 1
                if num == koef_d:
                    break
            k += 1
        sampled_map[i][j][2] = 1 if num // koef_d else 0

# for i in range(int(count_xy[1])):
#     print('строка {}'.format(i))
#     for j in range(int(count_xy[0])):
#         # print('столбец {}, строка {}'.format(j, i))
#         xc, yc = min_xy[0] + j * dl + dl / 2, max_xy[1] - i * dl + dl / 2
#         num = 0
#         k = 0
#         while k < len(indexed_raw_map):
#             it += 1
#             if True:  # indexed_raw_map[k][2] not in added:
#                 t += 1
#                 # print('в строке {} столбца {} для точки {} '.format(i,j,k))
#                 p_xy = indexed_raw_map[k]
#
#                 if xc + dl / 2 > p_xy[0] >= xc - dl / 2 and yc + dl / 2 >= \
#                         p_xy[
#                             1] > yc - dl / 2:
#                     num += 1
#                     # added.append(p_xy[2])
#                     indexed_raw_map = np.delete(indexed_raw_map, k, axis=0)
#                     p += 1
#                     k-=1
#                     if num == 5:
#                         break
#             k += 1
#         num = 1 if num >= 5 else 0
#         blank_map = np.append(blank_map, [[xc, yc, num]], axis=0)
# print(blank_map, blank_map.shape)
# np.save('blank_map', blank_map)
np.save('Sampled_map', sampled_map)
# for point in range(len(blank_map)):
#     print(blank_map[point])

# print(added, len(added))
# print('всего проверок', t)
# print('всего просм. точек', it)
# print('всего подошло точек', p)
# plt.scatter(blank_map[:, 0], blank_map[:, 1])
# plt.show()
