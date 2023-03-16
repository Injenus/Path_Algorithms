import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import sys
import math

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

"""
Робот круглый с диаметром - n_dl - кол-во дискреток (можно не целое)
Если робот больше одной дискретки, 
то препятсвия раздуваются на delta/2 с каждой стороны

Карту можно сделать замкнутой (поверхность тора) - параметр isClosedSurface
"""
isClosedSurface = False
r_bot = 0.65  # диаметр робота в метрах
speed = 4  # скорость движения робота в м/с

pygame.init()
display_info = pygame.display.Info()
prelim_size = (
    int(display_info.current_w * 0.85), int(display_info.current_h * 0.85))

# sampled_map = np.load('Sampled_map_022.npy')
sampled_map = np.load('Sampled_map_as_center_022_003.npy')
colmns, rows = sampled_map.shape[1], sampled_map.shape[0]

margin = int(1)
offset = (0, 0)
dl = min(prelim_size[0] // colmns - margin,
         (prelim_size[1] - 2 * offset[1]) // rows - margin)
print(dl)
size = ((dl + margin) * colmns, (dl + margin) * rows)
koef_obj = 1.  # коэффициент размера объектов в клетке
real_dl = sampled_map[0][1][0] - sampled_map[0][0][0]
print('Размер ячейки: {} м'.format(round(real_dl, 3)))
n_dl = r_bot / real_dl  # размер робота в дискретах
bot_isUnder = False
if n_dl > 1.18:
    bot_isUnder = True
print('Диаметр робота: {} м'.format(r_bot))
w_adj, w_diag = real_dl, 2 ** 0.5 * real_dl  # веса рёбер для смеж. и диаг. вершин
fps_static = 60
fps_move = speed / real_dl
fps = fps_static

if not isClosedSurface:
    for i in range(len(sampled_map)):
        for j in range(len(sampled_map[i])):
            if i == 0 or i == len(sampled_map) - 1 or j == 0 or j == len(
                    sampled_map[i]) - 1:
                sampled_map[i][j][2] = 1


# рашсирение препятсвий для корректного конфигурационного пространства
# такие математические препятсвия имеют цвет >1 (+2)
def increase_barrier(arr,
                     base_color=1,
                     set_color=2):  # увеличивает препт. на 1 дискр. с каждой стороны
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j][2] == base_color:
                try:
                    if not arr[i - 1][j][2]:
                        arr[i - 1][j][2] += set_color  # |^
                except IndexError:
                    pass
                try:
                    if not arr[i - 1][j + 1][2]:
                        arr[i - 1][j + 1][2] += set_color  # /^
                except IndexError:
                    pass
                try:
                    if not arr[i][j + 1][2]:
                        arr[i][j + 1][2] += set_color  # ->
                except IndexError:
                    pass
                try:
                    if not arr[i + 1][j + 1][2]:
                        arr[i + 1][j + 1][2] += set_color  # \.
                except IndexError:
                    pass
                try:
                    if not arr[i + 1][j][2]:
                        arr[i + 1][j][2] += set_color  # |.
                except IndexError:
                    pass
                try:
                    if not arr[i + 1][j - 1][2]:
                        arr[i + 1][j - 1][2] += set_color  # /.
                except IndexError:
                    pass
                try:
                    if not arr[i][j - 1][2]:
                        arr[i][j - 1][2] += set_color  # <-
                except IndexError:
                    pass
                try:
                    if not arr[i - 1][j - 1][2]:
                        arr[i - 1][j - 1][2] += set_color  # ^\
                except IndexError:
                    pass
    return arr


for i in range(math.ceil((n_dl - 1) / 2)):
    sampled_map = increase_barrier(sampled_map, 2 ** i, 2 ** (i + 1))

# print(size)
# print(dl)
# print(sampled_map.shape)

os.environ['SDL_VIDEO_WINDOW_POS'] = str(
    int(display_info.current_w - size[0] - 21)) + "," + str(42)
screen = pygame.display.set_mode(size)

pygame.display.set_caption(
    "'LMB' - set Start (PINK), 'RMB' - set Finish (GREEN), 'D' - Dijkstra, 'A' - A*, 'B' - bA*, 'M' - mA*, 'U' - uA*, 'R' - RRT.")


def draw_bot(ind_x, ind_y, w=0):  # аналог j и i
    x_r = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0] + dl / 2
    y_r = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1] + dl / 2
    pygame.draw.circle(screen, yellow, (x_r, y_r), ((dl + margin) * n_dl) / 2)


def draw_track(ind_x, ind_y):
    x_t = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0] + dl / 2
    y_t = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1] + dl / 2
    pygame.draw.circle(screen, (255, 255, 0), (x_t, y_t),
                       min(dl / 2 * 0.5, ((dl + margin) * n_dl) / 2 * 0.5))


def draw_finish(ind_x, ind_y):
    ds = dl * koef_obj
    x_t = (sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0]) * 1 + (
            dl + margin) * ind_x + offset[0]
    y_t = (-sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1]) * 1 + (
            dl + margin) * ind_y + offset[1]
    pygame.draw.polygon(screen, color_f, [[x_t + dl / 2, y_t + (dl - ds) / 2],
                                          [x_t + (dl + ds) / 2, y_t + dl / 2],
                                          [x_t + dl / 2, y_t + (dl + ds) / 2],
                                          [x_t + (dl - ds) / 2, y_t + dl / 2]])
    # pygame.draw.rect(screen, color_f,
    #                  (x_t + (dl - ds) / 2, y_t + (dl - ds) / 2, ds, ds))


def draw_start(ind_x, ind_y):
    ds = dl * koef_obj
    x_t = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0]
    y_t = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1]
    pygame.draw.polygon(screen, color_s, [[x_t + dl / 2, y_t + (dl - ds) / 2],
                                          [x_t + (dl + ds) / 2, y_t + dl / 2],
                                          [x_t + dl / 2, y_t + (dl + ds) / 2],
                                          [x_t + (dl - ds) / 2, y_t + dl / 2]])
    # pygame.draw.rect(screen, color_s,
    #                  (x_t + (dl - ds) / 2, y_t + (dl - ds) / 2, ds, ds))


def redraw_map(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j][2] == 0.5 or arr[i][j][2] == 0.25:
                arr[i][j][2] = 0
    return arr


def draw_node(node, value):
    if value == 0.25:
        if not sampled_map[int(node) // sampled_map.shape[1]][
            int(node) % sampled_map.shape[1]][2]:
            sampled_map[int(node) // sampled_map.shape[1]][
                int(node) % sampled_map.shape[1]][2] = value
    elif value == 0.5:
        if not sampled_map[int(node) // sampled_map.shape[1]][
            int(node) % sampled_map.shape[1]][2] or \
                sampled_map[int(node) // sampled_map.shape[1]][
                    int(node) % sampled_map.shape[1]][2] == 0.25:
            sampled_map[int(node) // sampled_map.shape[1]][
                int(node) % sampled_map.shape[1]][2] = 0.5


def d_et(s, f):
    print(
        '< Эталонный Дейкстра ищет путь среди {} вершин и {} рёбер... >'.format(
            graph_size,

            G.number_of_edges()))
    try:
        lenght, path = nx.single_source_dijkstra(G, s, f)
        print('Длина пути: {} м; шагов: {}'.format(round(lenght, 3),
                                                   len(path) - 1))
    except nx.exception.NetworkXNoPath:
        path = []
        print('Путь не существует!')
    return path


def dijkstra(s, f):
    if s == f:
        print(
            '< Дейкстра даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]
    print(
        '< Дейкстра ищет путь среди {} вершин и {} рёбер... >'.format(
            graph_size, G.number_of_edges()))

    """
    min_label хранит пары - [имя метка]
    unvisited_nodes - то же самое, но в нём удалются посещённые узлы
    arr[np.where(arr == 4)[0][0]] - выведет инфу о узле 4
    """
    # init_time = time.time()
    min_label = np.ndarray(shape=(0, 2), dtype=float)
    unvisited_nodes = np.ndarray(shape=(0, 2), dtype=float)
    for i in range(graph_size):
        min_label = np.append(min_label, [[i, float('inf')]], axis=0)
        unvisited_nodes = np.append(unvisited_nodes, [[i, float('inf')]],
                                    axis=0)
    min_label[s][1] = 0
    unvisited_nodes[s][1] = 0
    pathes = [[s] for i in range(graph_size)]

    k = 0
    while k < graph_size:  # можно без len(..)
        unvisited_nodes = unvisited_nodes[unvisited_nodes[:, 1].argsort()[::1]]
        curr_node = unvisited_nodes[0]
        if curr_node[1] == float('inf'):
            print('< Изучены все доступные вершины >')
            break
        else:
            to_vist = [n for n in G.neighbors(curr_node[0])]
            for i, node in enumerate(to_vist):
                try:
                    cur_ind = np.where(unvisited_nodes == node)[0][0]
                except IndexError:
                    # print('<нет такого>')
                    continue
                curr_label = curr_node[1] + G.edges[[curr_node[0], node]][
                    'weight']
                if curr_label < unvisited_nodes[cur_ind][1]:
                    unvisited_nodes[cur_ind][1] = curr_label
                    min_label[np.where(min_label == node)[0][0]][
                        1] = curr_label
                    pathes[node] = pathes[int(curr_node[0])] + [int(node)]
            unvisited_nodes = np.delete(unvisited_nodes, 0, axis=0)
            sampled_map[int(curr_node[0]) // sampled_map.shape[1]][
                int(curr_node[0]) % sampled_map.shape[1]][2] = 0.5
        k += 1
    # print('< Посещено узлов: {}, время работы: {} с >'.format(k,
    #                                                           time.time() - init_time))
    print('< Изучено вершин: {} >'.format(k))
    path = pathes[f]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(
            round(min_label[np.where(min_label == f)[0][0]][1], 3),
            len(path) - 1))
    else:
        path = []
        print('Путь не существует!')
    return path


def fru_a_star(s, f):
    if s == f:
        print(
            '< Тру A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (f % sampled_map.shape[1], f // sampled_map.shape[1])
        if isClosedSurface:
            # return 0  # 0 на торе стабильнее синусов :)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))
            # return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
            #         koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])), abs(
                koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
        else:
            # return ((xy_o[0] - xy_i[0]) ** 2 + (xy_o[1] - xy_i[1]) ** 2) ** 0.5
            return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))

    print(
        '< Тру A* ищет путь среди {} вершин и {} рёбер... >'.format(graph_size,
                                                                    G.number_of_edges()))
    min_label = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes = np.ndarray(shape=(0, 4), dtype=float)
    for i in range(graph_size):
        to_append = [i, float('inf'), huer(i), float('inf')]
        min_label = np.append(min_label, [to_append], axis=0)
        unvisited_nodes = np.append(unvisited_nodes, [to_append], axis=0)
    min_label[s][1] = 0
    unvisited_nodes[s][1] = 0
    pathes = [[s] for i in range(graph_size)]

    k = 1
    cur_node = s
    sampled_map[s // sampled_map.shape[1]][s % sampled_map.shape[1]][2] = 0.5
    while k < graph_size:
        if cur_node == f and len(pathes[f]) > 1:
            break
        to_vist = [n for n in G.neighbors(int(cur_node))]
        target_node = [float('inf'),
                       float('inf')]  # содер. имя и сумму узла, в кот. пойдём
        for i, node in enumerate(to_vist):
            try:
                ind = np.where(unvisited_nodes == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes[np.where(unvisited_nodes == cur_node)[0][0]][
                    1] + \
                G.edges[[cur_node, node]]['weight']
            if label < unvisited_nodes[ind][1]:
                unvisited_nodes[ind][1] = label
                unvisited_nodes[ind][3] = label + unvisited_nodes[ind][2]
                min_label[np.where(min_label == node)[0][0]][1] = label
                min_label[np.where(min_label == node)[0][0]][3] = label + \
                                                                  min_label[
                                                                      np.where(
                                                                          min_label == node)[
                                                                          0][
                                                                          0]][
                                                                      2]
                pathes[node] = pathes[int(cur_node)] + [int(node)]
            if unvisited_nodes[ind][2] == float('inf'):
                unvisited_nodes[ind][2] = \
                    min_label[np.where(min_label == node)[0][0]][2] = huer(
                    node)


            draw_node(node, 0.25)

        unvisited_nodes = np.delete(unvisited_nodes,
                                    np.where(unvisited_nodes == cur_node)[0][
                                        0], axis=0)

        unvisited_nodes = unvisited_nodes[
            unvisited_nodes[:, 3].argsort()[::1]]
        target_node[0] = unvisited_nodes[0][0]
        if target_node[0] == float('inf'):
            print('bc')
        cur_node = target_node[0]

        draw_node(cur_node, 0.5)
        k += 1
    # print('< Посещено узлов: {}, время работы: {} с >'.format(k,
    #                                                           time.time() - init_time))
    print('< Изучено вершин: {} >'.format(k))
    path = pathes[f]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(
            round(min_label[np.where(min_label == f)[0][0]][1], 3),
            len(path) - 1))
    # elif cur_node == f:
    #     path = []
    #     print(
    #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
    else:
        # print(path)
        path = []
        print('Путь не существует!')
    # print(pathes)
    return path


def a_star(s, f):
    """
    Идея как у Дейкстры, но берём вершину из to_list по минимальной сумме
    метки с эвристикой
    min_label и unvisited_nodes содержит вершину, метку (длина пути от старта),
    эвристику (синусовый Пифагор до финиша) и сумму метки и эврист.
    Чтобы корректно считать расстояние на торе, перевожу координаты ху в
    sin(ху_радиановые) (например, длина оси Х = 3 (0,1,2), тогда ПИ_по_оси_Х=3 ->
    чтобы sin(x_радиновый) имел период 3. Причём, ПИ_по_оси_У в общем случае
    не равен ПИ_по_оси_Х -> нарушается масштаб -> компенсируем коэффициентами.
    Например имеем карту
    0   1   2       0;0     1;0     2;0
    3   4   5       0;1     1;1     2;1
    6   7   8       0;2     1;2     2;2
    9   10  11      0;3     1;3     ..
    12  13 14
    Тогда х_радиановый будет равен Х/3, у_радиановый = Y/5
    """
    if s == f:
        print(
            '< A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (f % sampled_map.shape[1], f // sampled_map.shape[1])
        if isClosedSurface:
            # return 0  # 0 на торе стабильнее синусов :)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))
            # return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
            #         koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])), abs(
                koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
        else:
            return ((xy_o[0] - xy_i[0]) ** 2 + (xy_o[1] - xy_i[1]) ** 2) ** 0.5
            #return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))

    print('< A* ищет путь среди {} вершин и {} рёбер... >'.format(graph_size,
                                                                  G.number_of_edges()))
    min_label = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes = np.ndarray(shape=(0, 4), dtype=float)
    for i in range(graph_size):
        to_append = [i, float('inf'), huer(i), float('inf')]
        min_label = np.append(min_label, [to_append], axis=0)
        unvisited_nodes = np.append(unvisited_nodes, [to_append], axis=0)
    min_label[s][1] = 0
    unvisited_nodes[s][1] = 0
    pathes = [[s] for i in range(graph_size)]

    k = 1
    cur_node = s
    sampled_map[s // sampled_map.shape[1]][s % sampled_map.shape[1]][2] = 0.5
    while k < graph_size:
        if cur_node == f and len(pathes[f]) > 1:
            break
        to_vist = [n for n in G.neighbors(int(cur_node))]
        target_node = [float('inf'),
                       float('inf')]  # содер. имя и сумму узла, в кот. пойдём
        for i, node in enumerate(to_vist):
            try:
                ind = np.where(unvisited_nodes == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes[np.where(unvisited_nodes == cur_node)[0][0]][
                    1] + \
                G.edges[[cur_node, node]]['weight']
            if label < unvisited_nodes[ind][1]:
                unvisited_nodes[ind][1] = label
                unvisited_nodes[ind][3] = label + unvisited_nodes[ind][2]
                min_label[np.where(min_label == node)[0][0]][1] = label
                min_label[np.where(min_label == node)[0][0]][3] = label + \
                                                                  min_label[
                                                                      np.where(
                                                                          min_label == node)[
                                                                          0][
                                                                          0]][
                                                                      2]
                pathes[node] = pathes[int(cur_node)] + [int(node)]
            if unvisited_nodes[ind][2] == float('inf'):
                unvisited_nodes[ind][2] = \
                    min_label[np.where(min_label == node)[0][0]][2] = huer(
                    node)
            node_info = unvisited_nodes[ind]
            if node_info[3] < target_node[1]:
                target_node[0] = node  # или node_info[0]
                target_node[1] = node_info[3]

            draw_node(node, 0.25)

        unvisited_nodes = np.delete(unvisited_nodes,
                                    np.where(unvisited_nodes == cur_node)[0][
                                        0], axis=0)

        if target_node[0] == float('inf'):
            unvisited_nodes = unvisited_nodes[
                unvisited_nodes[:, 3].argsort()[::1]]
            target_node[0] = unvisited_nodes[0][0]
            # print(target_node)
            # if target_node[0] == float('inf'):
            #     print('< Путь не существует! >')
            #     break
        cur_node = target_node[0]

        draw_node(cur_node, 0.5)
        k += 1
    # print('< Посещено узлов: {}, время работы: {} с >'.format(k,
    #                                                           time.time() - init_time))
    print('< Изучено вершин: {} >'.format(k))
    path = pathes[f]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(
            round(min_label[np.where(min_label == f)[0][0]][1], 3),
            len(path) - 1))
    # elif cur_node == f:
    #     path = []
    #     print(
    #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
    else:
        # print(path)
        path = []
        print('Путь не существует!')
    # print(pathes)
    return path


def bidirect_a_star(s, f):
    if s == f:
        print(
            '< Двунаправленный A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node, target):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (target % sampled_map.shape[1], target // sampled_map.shape[1])
        if isClosedSurface:
            # return 0  # 0 на торе стабильнее синусов :)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))
            # return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
            #         koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])), abs(
                koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
        else:
            # return ((xy_o[0] - xy_i[0]) ** 2 + (xy_o[1] - xy_i[1]) ** 2) ** 0.5
            return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))

    print(
        '< Двунаправленный A* ищет путь среди {} вершин и {} рёбер... >'.format(
            graph_size, G.number_of_edges()))
    min_label_s = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes_s = np.ndarray(shape=(0, 4), dtype=float)
    min_label_f = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes_f = np.ndarray(shape=(0, 4), dtype=float)
    for i in range(graph_size):
        to_append_s = [i, float('inf'), huer(i, f), float('inf')]
        min_label_s = np.append(min_label_s, [to_append_s], axis=0)
        unvisited_nodes_s = np.append(unvisited_nodes_s, [to_append_s], axis=0)

        to_append_f = [i, float('inf'), huer(i, s), float('inf')]
        min_label_f = np.append(min_label_f, [to_append_f], axis=0)
        unvisited_nodes_f = np.append(unvisited_nodes_f, [to_append_f], axis=0)

    min_label_s[s][1] = 0
    unvisited_nodes_s[s][1] = 0

    min_label_f[f][1] = 0
    unvisited_nodes_f[f][1] = 0
    pathes_s = [[s] for i in range(graph_size)]
    pathes_f = [[f] for i in range(graph_size)]

    k = 1
    cur_node_s = s
    cur_node_f = f
    sampled_map[s // sampled_map.shape[1]][s % sampled_map.shape[1]][2] = 0.5
    sampled_map[f // sampled_map.shape[1]][f % sampled_map.shape[1]][2] = 0.5

    def check_neighbors(cur_node, to_visit, target_node, unvisited_nodes,
                        min_label, pathes, to_node):
        for i, node in enumerate(to_visit):
            try:
                ind = np.where(unvisited_nodes == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes[
                    np.where(unvisited_nodes == cur_node)[0][0]][
                    1] + \
                G.edges[[cur_node, node]]['weight']
            if label < unvisited_nodes[ind][1]:
                unvisited_nodes[ind][1] = label
                unvisited_nodes[ind][3] = label + unvisited_nodes[ind][2]
                min_label[np.where(min_label == node)[0][0]][1] = label
                min_label[np.where(min_label == node)[0][0]][3] = label + \
                                                                  min_label[
                                                                      np.where(
                                                                          min_label == node)[
                                                                          0][
                                                                          0]][
                                                                      2]
                pathes[node] = pathes[int(cur_node)] + [int(node)]
            if unvisited_nodes[ind][2] == float('inf'):
                unvisited_nodes[ind][2] = \
                    min_label[np.where(min_label == node)[0][0]][2] = huer(
                    node, to_node)
            node_info = unvisited_nodes[ind]
            if node_info[3] < target_node[1]:
                target_node[0] = node  # или node_info[0]
                target_node[1] = node_info[3]

        unvisited_nodes = np.delete(unvisited_nodes,
                                    np.where(unvisited_nodes == cur_node)[
                                        0][
                                        0], axis=0)

        if target_node[0] == float('inf'):
            unvisited_nodes = unvisited_nodes[
                unvisited_nodes[:, 3].argsort()[::1]]
            target_node[0] = unvisited_nodes[0][0]
            # print(target_node)
            # if target_node[0] == float('inf'):
            #     print('< Путь не существует! >')
            #     break
        # cur_node = target_node[0]

        if not sampled_map[int(cur_node) // sampled_map.shape[1]][
            int(cur_node) % sampled_map.shape[1]][2]:
            sampled_map[int(cur_node) // sampled_map.shape[1]][
                int(cur_node) % sampled_map.shape[1]][2] = 0.5
        return target_node[0]

    isAssembled = False
    while k < graph_size:
        if cur_node_s == f and len(pathes_s[f]) > 1:
            break
        if cur_node_f == s and len(pathes_f[s]) > 1:
            break

        if cur_node_s == cur_node_f:
            isAssembled = True
            # print('ura')

        to_visit_s = [n for n in G.neighbors(int(cur_node_s))]
        to_visit_f = [n for n in G.neighbors(int(cur_node_f))]
        target_node_s = [float('inf'),
                         float('inf')]  # сод. имя и сумму узла, в кот. пойдём
        target_node_f = [float('inf'),
                         float('inf')]  # сод. имя и сумму узла, в кот. пойдём

        # cur_node_s = check_neighbors(cur_node_s, to_visit_s, target_node_s,
        #                              unvisited_nodes_s, min_label_s, pathes_s,f)

        ########## для варианта от старта
        for i, node in enumerate(to_visit_s):
            try:
                ind = np.where(unvisited_nodes_s == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes_s[
                    np.where(unvisited_nodes_s == cur_node_s)[0][0]][
                    1] + \
                G.edges[[cur_node_s, node]]['weight']
            if label < unvisited_nodes_s[ind][1]:
                unvisited_nodes_s[ind][1] = label
                unvisited_nodes_s[ind][3] = label + unvisited_nodes_s[ind][2]
                min_label_s[np.where(min_label_s == node)[0][0]][1] = label
                min_label_s[np.where(min_label_s == node)[0][0]][3] = label + \
                                                                      min_label_s[
                                                                          np.where(
                                                                              min_label_s == node)[
                                                                              0][
                                                                              0]][
                                                                          2]
                pathes_s[node] = pathes_s[int(cur_node_s)] + [int(node)]
            if unvisited_nodes_s[ind][2] == float('inf'):
                unvisited_nodes_s[ind][2] = \
                    min_label_s[np.where(min_label_s == node)[0][0]][2] = huer(
                    node, f)
            node_info = unvisited_nodes_s[ind]
            if node_info[3] < target_node_s[1]:
                target_node_s[0] = node  # или node_info[0]
                target_node_s[1] = node_info[3]

            draw_node(node, 0.25)

        unvisited_nodes_s = np.delete(unvisited_nodes_s,
                                      np.where(
                                          unvisited_nodes_s == cur_node_s)[0][
                                          0], axis=0)

        if target_node_s[0] == float('inf'):
            unvisited_nodes_s = unvisited_nodes_s[
                unvisited_nodes_s[:, 3].argsort()[::1]]
            target_node_s[0] = unvisited_nodes_s[0][0]
            # print(target_node)
            # if target_node[0] == float('inf'):
            #     print('< Путь не существует! >')
            #     break
        cur_node_s = target_node_s[0]

        draw_node(cur_node_s, 0.5)
        ####### для от старта кончился

        ######для варинта от финиша
        for i, node in enumerate(to_visit_f):
            try:
                ind = np.where(unvisited_nodes_f == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes_f[
                    np.where(unvisited_nodes_f == cur_node_f)[0][0]][
                    1] + \
                G.edges[[cur_node_f, node]]['weight']
            if label < unvisited_nodes_f[ind][1]:
                unvisited_nodes_f[ind][1] = label
                unvisited_nodes_f[ind][3] = label + unvisited_nodes_f[ind][2]
                min_label_f[np.where(min_label_f == node)[0][0]][1] = label
                min_label_f[np.where(min_label_f == node)[0][0]][3] = label + \
                                                                      min_label_f[
                                                                          np.where(
                                                                              min_label_f == node)[
                                                                              0][
                                                                              0]][
                                                                          2]
                pathes_f[node] = pathes_f[int(cur_node_f)] + [int(node)]
            if unvisited_nodes_f[ind][2] == float('inf'):
                unvisited_nodes_f[ind][2] = \
                    min_label_f[np.where(min_label_f == node)[0][0]][2] = huer(
                    node, s)
            node_info = unvisited_nodes_f[ind]
            if node_info[3] < target_node_f[1]:
                target_node_f[0] = node  # или node_info[0]
                target_node_f[1] = node_info[3]

            draw_node(node, 0.25)

        unvisited_nodes_f = np.delete(unvisited_nodes_f,
                                      np.where(
                                          unvisited_nodes_f == cur_node_f)[
                                          0][
                                          0], axis=0)

        if target_node_f[0] == float('inf'):
            unvisited_nodes_f = unvisited_nodes_f[
                unvisited_nodes_f[:, 3].argsort()[::1]]
            target_node_f[0] = unvisited_nodes_f[0][0]
            # print(target_node)
            # if target_node[0] == float('inf'):
            #     print('< Путь не существует! >')
            #     break
        cur_node_f = target_node_f[0]

        draw_node(cur_node_f, 0.5)
        ###### для от финиша кончился

        k += 1

    # print('< Посещено узлов: {}, время работы: {} с >'.format(k,
    #                                                           time.time() - init_time))
    # print('< Изучено вершин: {} >'.format(k))
    if min_label_s[np.where(min_label_s == f)[0][0]][1] < \
            min_label_f[np.where(min_label_f == s)[0][0]][1]:
        path = pathes_s[f]
        if len(path) > 1:
            print('Длина пути: {} м, шагов: {}'.format(
                round(min_label_s[np.where(min_label_s == f)[0][0]][1], 3),
                len(path) - 1))
        # elif cur_node == f:
        #     path = []
        #     print(
        #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
        else:
            # print(path)
            path = []
            print('Путь не существует!')
        # print(pathes)
    else:
        path = list(reversed(pathes_f[s]))
        if len(path) > 1:
            print('Длина пути: {} м, шагов: {}'.format(
                round(min_label_f[np.where(min_label_f == s)[0][0]][1], 3),
                len(path) - 1))
        # elif cur_node == f:
        #     path = []
        #     print(
        #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
        else:
            # print(path)
            path = []
            print('Путь не существует!')
        # print(pathes)
    return path


def massive_a_star(s, f):
    if s == f:
        print(
            '< Массивный A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (f % sampled_map.shape[1], f // sampled_map.shape[1])
        if isClosedSurface:
            # return 0  # 0 на торе стабильнее синусов :)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))
            # return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
            #         koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])), abs(
                koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
        else:
            # return ((xy_o[0] - xy_i[0]) ** 2 + (xy_o[1] - xy_i[1]) ** 2) ** 0.5
            return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))

    print('< Массивный A* ищет путь среди {} вершин и {} рёбер... >'.format(
        graph_size,
        G.number_of_edges()))
    min_label = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes = np.ndarray(shape=(0, 4), dtype=float)
    for i in range(graph_size):
        to_append = [i, float('inf'), huer(i), float('inf')]
        min_label = np.append(min_label, [to_append], axis=0)
        unvisited_nodes = np.append(unvisited_nodes, [to_append], axis=0)
    min_label[s][1] = 0
    unvisited_nodes[s][1] = 0
    pathes = [[s] for i in range(graph_size)]

    k = 1
    cur_node = s
    sampled_map[s // sampled_map.shape[1]][s % sampled_map.shape[1]][2] = 0.5
    while k < graph_size:
        if cur_node == f and len(pathes[f]) > 1:
            break
        to_vist = [n for n in G.neighbors(int(cur_node))]
        target_node = [float('inf'),
                       float('inf')]  # содер. имя и сумму узла, в кот. пойдём
        for i, node in enumerate(to_vist):
            try:
                ind = np.where(unvisited_nodes == node)[0][0]
            except IndexError:
                continue

            label = \
                unvisited_nodes[np.where(unvisited_nodes == cur_node)[0][0]][
                    1] + \
                G.edges[[cur_node, node]]['weight']
            if label < unvisited_nodes[ind][1]:
                unvisited_nodes[ind][1] = label
                unvisited_nodes[ind][3] = label + unvisited_nodes[ind][2]
                min_label[np.where(min_label == node)[0][0]][1] = label
                min_label[np.where(min_label == node)[0][0]][3] = label + \
                                                                  min_label[
                                                                      np.where(
                                                                          min_label == node)[
                                                                          0][
                                                                          0]][
                                                                      2]
                pathes[node] = pathes[int(cur_node)] + [int(node)]
            if unvisited_nodes[ind][2] == float('inf'):
                unvisited_nodes[ind][2] = \
                    min_label[np.where(min_label == node)[0][0]][2] = huer(
                    node)
            node_info = unvisited_nodes[ind]
            if node_info[3] < target_node[1]:
                target_node[0] = node  # или node_info[0]
                target_node[1] = node_info[3]

            draw_node(node, 0.25)

            to_visit_child = [n for n in G.neighbors(int(node))]
            # проверяем "округу округи"
            for j, node_ch in enumerate(to_visit_child):
                try:
                    ind_ch = np.where(unvisited_nodes == node_ch)[0][0]
                except IndexError:
                    continue
                lbl_ch = \
                    unvisited_nodes[np.where(unvisited_nodes == node)[0][0]][
                        1] + \
                    G.edges[[node, node_ch]]['weight']
                if lbl_ch < unvisited_nodes[ind_ch][1]:
                    unvisited_nodes[ind_ch][1] = lbl_ch
                    unvisited_nodes[ind_ch][3] = lbl_ch + \
                                                 unvisited_nodes[ind_ch][2]
                    min_label[np.where(min_label == node_ch)[0][0]][1] = lbl_ch
                    min_label[np.where(min_label == node_ch)[0][0]][
                        3] = lbl_ch + \
                             min_label[np.where(min_label == node_ch)[0][0]][2]
                    pathes[node_ch] = pathes[int(node)] + [int(node_ch)]
                if unvisited_nodes[ind_ch][2] == float('inf'):
                    unvisited_nodes[ind_ch][2] = \
                        min_label[np.where(min_label == node_ch)[0][0]][
                            2] = huer(node_ch)
                draw_node(node_ch, 0.25)

                # if not sampled_map[int(node) // sampled_map.shape[1]][
                #     int(node) % sampled_map.shape[1]][2]:
                #     sampled_map[int(node) // sampled_map.shape[1]][
                #         int(node) % sampled_map.shape[1]][2] = 0.5

        # звершн. основ. цикл
        unvisited_nodes = np.delete(unvisited_nodes,
                                    np.where(unvisited_nodes == cur_node)[0][
                                        0], axis=0)

        if target_node[0] == float('inf'):
            unvisited_nodes = unvisited_nodes[
                unvisited_nodes[:, 3].argsort()[::1]]
            target_node[0] = unvisited_nodes[0][0]
        cur_node = target_node[0]

        draw_node(cur_node, 0.5)
        k += 1
    # print('< Посещено узлов: {}, время работы: {} с >'.format(k,
    #                                                           time.time() - init_time))
    # print('< Изучено вершин: {} >'.format(k))
    path = pathes[f]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(
            round(min_label[np.where(min_label == f)[0][0]][1], 3),
            len(path) - 1))
    # elif cur_node == f:
    #     path = []
    #     print(
    #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
    else:
        # print(path)
        path = []
        print('Путь не существует!')
    # print(pathes)
    return path


def uber_a_star(s, f):
    if s == f:
        print(
            '< Улучшенный A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node, target):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (target % sampled_map.shape[1], target // sampled_map.shape[1])
        if isClosedSurface:
            # return 0  # 0 на торе стабильнее синусов :)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))
            # return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
            #         koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])), abs(
                koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
        else:
            # return ((xy_o[0] - xy_i[0]) ** 2 + (xy_o[1] - xy_i[1]) ** 2) ** 0.5
            return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))

    print(
        '< Улучшенный A* ищет путь среди {} вершин и {} рёбер... >'.format(
            graph_size, G.number_of_edges()))
    min_label_s = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes_s = np.ndarray(shape=(0, 4), dtype=float)
    min_label_f = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes_f = np.ndarray(shape=(0, 4), dtype=float)
    for i in range(graph_size):
        to_append_s = [i, float('inf'), huer(i, f), float('inf')]
        min_label_s = np.append(min_label_s, [to_append_s], axis=0)
        unvisited_nodes_s = np.append(unvisited_nodes_s, [to_append_s], axis=0)

        to_append_f = [i, float('inf'), huer(i, s), float('inf')]
        min_label_f = np.append(min_label_f, [to_append_f], axis=0)
        unvisited_nodes_f = np.append(unvisited_nodes_f, [to_append_f], axis=0)

    min_label_s[s][1] = 0
    unvisited_nodes_s[s][1] = 0

    min_label_f[f][1] = 0
    unvisited_nodes_f[f][1] = 0
    pathes_s = [[s] for i in range(graph_size)]
    pathes_f = [[f] for i in range(graph_size)]

    k = 1
    cur_node_s = s
    cur_node_f = f
    sampled_map[s // sampled_map.shape[1]][s % sampled_map.shape[1]][2] = 0.5
    sampled_map[f // sampled_map.shape[1]][f % sampled_map.shape[1]][2] = 0.5
    n_count = 2
    isAssembled = False
    while k < graph_size:
        if cur_node_s == f and len(pathes_s[f]) > 1:
            break
        if cur_node_f == s and len(pathes_f[s]) > 1:
            break

        if cur_node_s == cur_node_f:
            isAssembled = True
            # print('ura')

        to_visit_s = [n for n in G.neighbors(int(cur_node_s))]
        to_visit_f = [n for n in G.neighbors(int(cur_node_f))]
        target_node_s = [float('inf'),
                         float('inf')]  # сод. имя и сумму узла, в кот. пойдём
        target_node_f = [float('inf'),
                         float('inf')]  # сод. имя и сумму узла, в кот. пойдём

        # cur_node_s = check_neighbors(cur_node_s, to_visit_s, target_node_s,
        #                              unvisited_nodes_s, min_label_s, pathes_s,f)

        ########## для варианта от старта
        for i, node in enumerate(to_visit_s):
            try:
                ind = np.where(unvisited_nodes_s == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes_s[
                    np.where(unvisited_nodes_s == cur_node_s)[0][0]][
                    1] + \
                G.edges[[cur_node_s, node]]['weight']
            if label < unvisited_nodes_s[ind][1]:
                unvisited_nodes_s[ind][1] = label
                unvisited_nodes_s[ind][3] = label + unvisited_nodes_s[ind][2]
                min_label_s[np.where(min_label_s == node)[0][0]][1] = label
                min_label_s[np.where(min_label_s == node)[0][0]][3] = label + \
                                                                      min_label_s[
                                                                          np.where(
                                                                              min_label_s == node)[
                                                                              0][
                                                                              0]][
                                                                          2]
                pathes_s[node] = pathes_s[int(cur_node_s)] + [int(node)]
            if unvisited_nodes_s[ind][2] == float('inf'):
                unvisited_nodes_s[ind][2] = \
                    min_label_s[np.where(min_label_s == node)[0][0]][2] = huer(
                    node, f)
            node_info = unvisited_nodes_s[ind]
            if node_info[3] < target_node_s[1]:
                target_node_s[0] = node  # или node_info[0]
                target_node_s[1] = node_info[3]

            to_visit_child_s = [n for n in G.neighbors(int(node))]
            # проверяем "округу округи" старта
            for j, node_ch in enumerate(to_visit_child_s):
                try:
                    ind_ch = np.where(unvisited_nodes_s == node_ch)[0][0]
                except IndexError:
                    continue
                lbl_ch = \
                    unvisited_nodes_s[
                        np.where(unvisited_nodes_s == node)[0][0]][
                        1] + \
                    G.edges[[node, node_ch]]['weight']
                if lbl_ch < unvisited_nodes_s[ind_ch][1]:
                    unvisited_nodes_s[ind_ch][1] = lbl_ch
                    unvisited_nodes_s[ind_ch][3] = lbl_ch + \
                                                   unvisited_nodes_s[ind_ch][2]
                    min_label_s[np.where(min_label_s == node_ch)[0][0]][
                        1] = lbl_ch
                    min_label_s[np.where(min_label_s == node_ch)[0][0]][
                        3] = lbl_ch + \
                             min_label_s[
                                 np.where(min_label_s == node_ch)[0][0]][2]
                    pathes_s[node_ch] = pathes_s[int(node)] + [int(node_ch)]
                if unvisited_nodes_s[ind_ch][2] == float('inf'):
                    unvisited_nodes_s[ind_ch][2] = \
                        min_label_s[
                            np.where(min_label_s == node_ch)[0][0]][
                            2] = huer(node_ch)

                if not sampled_map[int(node_ch) // sampled_map.shape[1]][
                    int(node_ch) % sampled_map.shape[1]][2]:
                    sampled_map[int(node_ch) // sampled_map.shape[1]][
                        int(node_ch) % sampled_map.shape[1]][2] = 0.25

        unvisited_nodes_s = np.delete(unvisited_nodes_s,
                                      np.where(
                                          unvisited_nodes_s == cur_node_s)[0][
                                          0], axis=0)

        if target_node_s[0] == float('inf'):
            unvisited_nodes_s = unvisited_nodes_s[
                unvisited_nodes_s[:, 3].argsort()[::1]]
            target_node_s[0] = unvisited_nodes_s[0][0]
            # print(target_node)
            # if target_node[0] == float('inf'):
            #     print('< Путь не существует! >')
            #     break
        cur_node_s = target_node_s[0]

        if not sampled_map[int(cur_node_s) // sampled_map.shape[1]][
            int(cur_node_s) % sampled_map.shape[1]][2]:
            sampled_map[int(cur_node_s) // sampled_map.shape[1]][
                int(cur_node_s) % sampled_map.shape[1]][2] = 0.5
        ####### для от старта кончился

        ######для варинта от финиша
        for i, node in enumerate(to_visit_f):
            try:
                ind = np.where(unvisited_nodes_f == node)[0][0]
            except IndexError:
                continue
            label = \
                unvisited_nodes_f[
                    np.where(unvisited_nodes_f == cur_node_f)[0][0]][
                    1] + \
                G.edges[[cur_node_f, node]]['weight']
            if label < unvisited_nodes_f[ind][1]:
                unvisited_nodes_f[ind][1] = label
                unvisited_nodes_f[ind][3] = label + unvisited_nodes_f[ind][2]
                min_label_f[np.where(min_label_f == node)[0][0]][1] = label
                min_label_f[np.where(min_label_f == node)[0][0]][3] = label + \
                                                                      min_label_f[
                                                                          np.where(
                                                                              min_label_f == node)[
                                                                              0][
                                                                              0]][
                                                                          2]
                pathes_f[node] = pathes_f[int(cur_node_f)] + [int(node)]
            if unvisited_nodes_f[ind][2] == float('inf'):
                unvisited_nodes_f[ind][2] = \
                    min_label_f[np.where(min_label_f == node)[0][0]][2] = huer(
                    node, s)
            node_info = unvisited_nodes_f[ind]
            if node_info[3] < target_node_f[1]:
                target_node_f[0] = node  # или node_info[0]
                target_node_f[1] = node_info[3]

            to_visit_child_f = [n for n in G.neighbors(int(node))]
            # проверяем "округу округи" финиш
            for k, node_ch in enumerate(to_visit_child_f):
                try:
                    ind_ch = np.where(unvisited_nodes_f == node_ch)[0][0]
                except IndexError:
                    continue
                lbl_ch = \
                    unvisited_nodes_f[
                        np.where(unvisited_nodes_f == node)[0][0]][
                        1] + \
                    G.edges[[node, node_ch]]['weight']
                if lbl_ch < unvisited_nodes_f[ind_ch][1]:
                    unvisited_nodes_f[ind_ch][1] = lbl_ch
                    unvisited_nodes_f[ind_ch][3] = lbl_ch + \
                                                   unvisited_nodes_f[ind_ch][2]
                    min_label_f[np.where(min_label_f == node_ch)[0][0]][
                        1] = lbl_ch
                    min_label_f[np.where(min_label_f == node_ch)[0][0]][
                        3] = lbl_ch + \
                             min_label_f[
                                 np.where(min_label_f == node_ch)[0][0]][2]
                    pathes_f[node_ch] = pathes_f[int(node)] + [int(node_ch)]
                    if unvisited_nodes_f[ind_ch][2] == float('inf'):
                        unvisited_nodes_f[ind_ch][2] = \
                            min_label_f[
                                np.where(min_label_f == node_ch)[0][0]][
                                2] = huer(node_ch)

                    if not sampled_map[int(node) // sampled_map.shape[1]][
                        int(node) % sampled_map.shape[1]][2]:
                        sampled_map[int(node) // sampled_map.shape[1]][
                            int(node) % sampled_map.shape[1]][2] = 0.5

        unvisited_nodes_f = np.delete(unvisited_nodes_f,
                                      np.where(
                                          unvisited_nodes_f == cur_node_f)[
                                          0][
                                          0], axis=0)

        if target_node_f[0] == float('inf'):
            unvisited_nodes_f = unvisited_nodes_f[
                unvisited_nodes_f[:, 3].argsort()[::1]]
            target_node_f[0] = unvisited_nodes_f[0][0]
            # print(target_node)
            # if target_node[0] == float('inf'):
            #     print('< Путь не существует! >')
            #     break
        cur_node_f = target_node_f[0]

        if not sampled_map[int(cur_node_f) // sampled_map.shape[1]][
            int(cur_node_f) % sampled_map.shape[1]][2]:
            sampled_map[int(cur_node_f) // sampled_map.shape[1]][
                int(cur_node_f) % sampled_map.shape[1]][2] = 0.5
        ###### для от финиша кончился

        k += 1

    # print('< Посещено узлов: {}, время работы: {} с >'.format(k,
    #                                                           time.time() - init_time))
    # print('< Изучено вершин: {} >'.format(k))
    if min_label_s[np.where(min_label_s == f)[0][0]][1] < \
            min_label_f[np.where(min_label_f == s)[0][0]][1]:
        path = pathes_s[f]
        if len(path) > 1:
            print('Длина пути: {} м, шагов: {}'.format(
                round(min_label_s[np.where(min_label_s == f)[0][0]][1], 3),
                len(path) - 1))
        # elif cur_node == f:
        #     path = []
        #     print(
        #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
        else:
            # print(path)
            path = []
            print('Путь не существует!')
        # print(pathes)
    else:
        path = list(reversed(pathes_f[s]))
        if len(path) > 1:
            print('Длина пути: {} м, шагов: {}'.format(
                round(min_label_f[np.where(min_label_f == s)[0][0]][1], 3),
                len(path) - 1))
        # elif cur_node == f:
        #     path = []
        #     print(
        #         'Путь не существует! <завершили досрочно, так как случайно ткнулись в финиш, до которого нет пути>')
        else:
            # print(path)
            path = []
            print('Путь не существует!')
        # print(pathes)
    return path


# print(sampled_map.shape)


def rrt():
    print('<start RRT>')
    pass


G = nx.Graph()
for n in range(sampled_map.shape[0] * sampled_map.shape[1]):
    G.add_node(n)
graph_size = G.number_of_nodes()


def check_up(name, i, j):
    if not sampled_map[i - 1][j][2]:  # верх
        G.add_edge(name,
                   (name - sampled_map.shape[1] + graph_size) % graph_size,
                   weight=w_adj)


def check_right(name, i, j):
    if not sampled_map[i][(j + 1) % len(sampled_map[i])][2]:  # право
        if j == len(sampled_map[i]) - 1:
            G.add_edge(name, name - sampled_map.shape[1] + 1, weight=w_adj)
        else:
            G.add_edge(name, name + 1, weight=w_adj)


def check_down(name, i, j):
    if not sampled_map[i + 1][j][2]:  # низ
        G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)


def check_left(name, i, j):
    if not sampled_map[i][j - 1][2]:  # лево
        if j == 0:
            G.add_edge(name, name - 1 + sampled_map.shape[1], weight=w_adj)
        else:
            G.add_edge(name, name - 1, weight=w_adj)


def check_ru(name, i, j):
    if not sampled_map[i - 1][j][2] and not \
            sampled_map[i][(j + 1) % len(sampled_map[i])][2] and not \
            sampled_map[i - 1][(j + 1) % len(sampled_map[i])][2]:  # пв
        if j == len(sampled_map[i]) - 1:
            G.add_edge(name, (name - sampled_map.shape[
                1] + graph_size) % graph_size - sampled_map.shape[1] + 1,
                       weight=w_diag)
        else:
            G.add_edge(name, (name - sampled_map.shape[
                1] + graph_size) % graph_size + 1, weight=w_diag)


def check_rd(name, i, j):
    if not sampled_map[i][j + 1][2] and not sampled_map[i + 1][j][2] and not \
            sampled_map[i + 1][j + 1][2]:  # пн
        G.add_edge(name, name + sampled_map.shape[1] + 1, weight=w_diag)


def check_ld(name, i, j):
    if not sampled_map[i + 1][j][2] and not sampled_map[i][j - 1][2] and not \
            sampled_map[i + 1][j - 1][2]:  # лн
        G.add_edge(name, name + sampled_map.shape[1] - 1, weight=w_diag)


def check_lu(name, i, j):
    if not sampled_map[i][j - 1][2] and not sampled_map[i - 1][j][2] and not \
            sampled_map[i - 1][j - 1][2]:  # лв
        if j == 0:
            G.add_edge(name,
                       (name - 1 + graph_size) % graph_size, weight=w_diag)
        else:
            G.add_edge(name, (name - sampled_map.shape[
                1] + graph_size) % graph_size - 1, weight=w_diag)


# проверка по часовой (сверху/ пв или ближайший к этим)
for i in range(len(sampled_map)):
    for j in range(len(sampled_map[i])):
        if not sampled_map[i][j][2]:
            name = sampled_map.shape[1] * i + j
            try:
                check_up(name, i, j)
            except IndexError:
                pass
            try:
                check_right(name, i, j)
            except IndexError:
                pass
            try:
                check_down(name, i, j)
            except IndexError:
                pass
            try:
                check_left(name, i, j)
            except IndexError:
                pass
            try:
                check_ru(name, i, j)
            except IndexError:
                pass
            try:
                check_rd(name, i, j)
            except IndexError:
                pass
            try:
                check_ld(name, i, j)
            except IndexError:
                pass
            try:
                check_lu(name, i, j)
            except IndexError:
                pass
# for i in range(len(sampled_map)):
#     for j in range(len(sampled_map[i])):
#         if not sampled_map[i][j][2]:
#             name = sampled_map.shape[1] * i + j
#             if i == 0:  # если верхняя строка
#                 if j == 0:  # если левый столбей
#                     # name = 0
#                     check_right(name, i, j)
#                     check_down(name, i, j)
#                     # diag
#                     check_rd(name, i, j)
#                 elif j == sampled_map.shape[1] - 1:  # если правый столбец
#                     # name = sampled_map.shape[1] - 1
#                     check_down(name, i, j)
#                     check_left(name, i, j)
#                     # diag
#                     check_ld(name, i, j)
#                 else:  # если остальное (не пров. верх. и пв и лв)
#                     # name = j
#                     check_right(name, i, j)
#                     check_down(name, i, j)
#                     check_left(name, i, j)
#                     # диаг
#                     check_rd(name, i, j)
#                     check_ld(name, i, j)
#
#             elif i == sampled_map.shape[0] - 1:  # если нижняя строка
#                 if j == 0:  # если левый столбей
#                     check_up(name, i, j)
#                     check_right(name, i, j)
#                     # diag
#                     check_ru(name, i, j)
#                 elif j == sampled_map.shape[1] - 1:  # если правый столбец
#                     check_up(name, i, j)
#                     check_left(name, i, j)
#                     # diag
#                     check_lu(name, i, j)
#                 else:  # если остальное (кроме низ, пн, лн)
#                     check_up(name, i, j)
#                     check_right(name, i, j)
#                     check_left(name, i, j)
#                     # diag
#                     check_ru(name, i, j)
#                     check_lu(name, i, j)
#
#             else:  # любая другая строка
#                 if j == 0:  # если левый столбей (кроме лево, лн, лв)
#                     check_up(name, i, j)
#                     check_right(name, i, j)
#                     check_down(name, i, j)
#                     # diag
#                     check_ru(name, i, j)
#                     check_rd(name, i, j)
#                 elif j == sampled_map.shape[
#                     1] - 1:  # if прав.стл. (кр. право, пв, пн)
#                     check_up(name, i, j)
#                     check_down(name, i, j)
#                     check_left(name, i, j)
#                     # diag
#                     check_ld(name, i, j)
#                     check_lu(name, i, j)
#                 else:  # если остальное
#                     # name = sampled_map.shape[1] * i + j
#                     # проверяем по часовой сверху, сначала прямые, потом диагонали
#                     check_up(name, i, j)
#                     check_right(name, i, j)
#                     check_down(name, i, j)
#                     check_left(name, i, j)
#
#                     # диагональные проверяем: (пв, пн, лн, лв)
#                     check_ru(name, i, j)
#                     check_rd(name, i, j)
#                     check_ld(name, i, j)
#                     check_lu(name, i, j)
# print(len(G.nodes()), G.nodes())
# print(len(G.edges()), G.edges())
# print(nx.dijkstra_path(G, 0, 20))

# print([n for n in G.neighbors(0)])
# print(G.edges[[0, 14]])

red, blue = (255, 0, 0), (0, 0, 255)
green_dark, green_light = (0, 160, 0), (128, 255, 128)
gray, gray_1, gray_2 = (128, 128, 128), (128, 128, 192), (192, 128, 128)
color_f, color_s = (0, 0, 0), (255, 255, 255)
yellow = (255, 153, 0)
index_bot = [0, 0]
index_start = [0, 0]
index_finish = [0, 0]
isInProgress = False
isSetObj = [False, False]  # start/bot, finish
isNewEnvir = False
clock = pygame.time.Clock()
path_track = []
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.MOUSEBUTTONDOWN and not isInProgress:
            mouse_xy = pygame.mouse.get_pos()
            # print(mouse_xy)
            # print(index_bot)
            indexes = [(mouse_xy[0] - offset[0]) // (dl + margin),
                       (mouse_xy[1] - offset[1]) // (dl + margin)]
            if not sampled_map[indexes[1]][indexes[0]][2] // 1:
                redraw_map(sampled_map)
                isNewEnvir = True
                if event.button == 1:
                    index_start = indexes
                    print(
                        'Старт в {} м'.format(
                            [round(x * real_dl + real_dl / 2, 2) for x in
                             indexes]))
                    index_bot = indexes
                    isSetObj[0] = True
                elif event.button == 3:
                    index_finish = indexes
                    print('Финиш в {} м'.format(
                        [round(x * real_dl + real_dl / 2, 2) for x in
                         indexes]))
                    index_bot = index_start
                    isSetObj[1] = True
        elif event.type == pygame.KEYUP and not isInProgress:
            if isSetObj[0] * isSetObj[1]:
                if event.key == pygame.K_d:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = dijkstra(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key==pygame.K_f:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = fru_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key == pygame.K_a:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key == pygame.K_b:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = bidirect_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key == pygame.K_m:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = massive_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key == pygame.K_u:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = uber_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key == pygame.K_r:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    rrt()
                    path = []
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif event.key == pygame.K_l:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    # print('s', index_start)
                    # print('f', index_finish)
                    path = d_et(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move

    if not isSetObj[0]:
        if bot_isUnder:
            draw_bot(index_bot[0], index_bot[1])
        draw_start(index_start[0], index_start[1])
    if not isSetObj[1]:
        draw_finish(index_finish[0], index_finish[1])
    if not bot_isUnder and not isSetObj[0]:
        draw_bot(index_bot[0], index_bot[1])
    if len(path_track) and not isNewEnvir:
        for t in path_track:
            draw_track(t % sampled_map.shape[1], t // sampled_map.shape[1])

    screen.fill((0, 0, 0))
    for i in range(len(sampled_map)):
        for j in range(len(sampled_map[i])):
            # x = sampled_map[i][j][0] - 0.5 ** 0.5 * dl + (
            #         dl + margin) * j + offset[0]
            # y = -sampled_map[i][j][1] - 0.5 ** 0.5 * dl + (
            #         dl + margin) * i + offset[1]
            x = int(sampled_map[i][j][0] - sampled_map[0][0][0] + (
                    dl + margin) * j + offset[0])
            y = int(-sampled_map[i][j][1] + sampled_map[0][0][1] + (
                    dl + margin) * i + offset[1])
            if sampled_map[i][j][2] == 0:
                color = blue
            elif sampled_map[i][j][2] == 1:
                color = red
            elif sampled_map[i][j][2] == 0.5:
                # color = (0, 168, 192)
                color = green_dark
            elif sampled_map[i][j][2] == 0.25:
                color = green_light
            else:
                color = blue
                # color = gray
                color = (112, 64, 255)
            pygame.draw.rect(screen, color, (x, y, dl, dl))

    if isSetObj[0]:
        if bot_isUnder:
            draw_bot(index_bot[0], index_bot[1])
        draw_start(index_start[0], index_start[1])
    if isSetObj[1]:
        draw_finish(index_finish[0], index_finish[1])
    if not bot_isUnder and isSetObj[0]:
        draw_bot(index_bot[0], index_bot[1])
    if len(path_track) and not isNewEnvir:
        for t in path_track:
            draw_track(t % sampled_map.shape[1], t // sampled_map.shape[1])

    if isInProgress:
        if len(path):
            cell = path.pop(0)
            path_track.append(cell)
            # print(cell)
            index_bot = [cell % sampled_map.shape[1],
                         cell // sampled_map.shape[1]]
        else:
            time.sleep(1 / fps_move)
            fps = fps_static
            print('< Завершено >\n')
            isInProgress = False

    # pos = nx.spring_layout(G)  # pos = nx.nx_agraph.graphviz_layout(G)
    # nx.draw_networkx(G, pos)
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.show()
    clock.tick(fps)
    pygame.display.update()
