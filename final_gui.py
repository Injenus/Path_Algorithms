"""
Изменёно создание пространсва на диагонали
Исправлен А*
+ затемненение для крайних вершин дерева
+ rrt*
+ сумма Минковского для rect бота
"""

import os
import networkx as nx
import numpy as np
import time
import sys
import math
import random

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

"""
Робот круглый с диаметром - n_dl - кол-во дискреток (можно не целое)
Если робот больше одной дискретки, 
то препятсвия раздуваются на delta/2 с каждой стороны

Карту можно сделать замкнутой (поверхность тора) - параметр isClosedSurface
"""
isClosedSurface = False
isMinkowski = True
r_w, r_h = 0.35, 0.60  # габариты робота в метрах
if isMinkowski:
    r_bot = ((r_w) ** 2 + (r_h) ** 2) ** 0.5  # диаметр в метрах
else:
    r_bot = 0.65
speed = 10  # скорость движения робота в м/с

pygame.init()
display_info = pygame.display.Info()
prelim_size = (
    int(display_info.current_w * 0.85), int(display_info.current_h * 0.85))

sampled_map = np.load('Sampled_map_as_center_0125.npy')
colmns, rows = sampled_map.shape[1], sampled_map.shape[0]

margin = int(1)
offset = (0, 0)
dl = min(prelim_size[0] // colmns - margin,
         (prelim_size[1] - 2 * offset[1]) // rows - margin)
# print(dl)
size = ((dl + margin) * colmns, (dl + margin) * rows)
koef_obj = 1.  # коэффициент размера объектов в клетке
real_dl = sampled_map[0][1][0] - sampled_map[0][0][0]
print('Размер ячейки: {} м'.format(round(real_dl, 3)))
n_dl = r_bot / real_dl  # размер робота в дискретах
d_r_w = r_w / real_dl * dl
d_r_l = r_h / real_dl * dl
bot_isUnder = False
if n_dl > 1.18:
    bot_isUnder = True
print('Габаритные размеры робота: {} x {} м'.format(r_h, r_w))
print('Габаритный диаметр робота: {} м'.format(round(r_bot, 2)))
w_adj, w_diag = real_dl, 2 ** 0.5 * real_dl  # веса рёбер для смеж. и диаг. вершин
fps_static = 60
fps_move = speed / real_dl
fps = fps_static
# print(sampled_map[:, :, 2])
quan_free_nodes = np.sum(sampled_map[:, :, 2] == 0)

max_d_rrt = real_dl * 4.5
max_samples_rrt = 3000
r_find = 1.5 * max_d_rrt
if r_find > math.floor(0.95 * r_bot / real_dl) * real_dl:
    r_find = math.floor(0.95 * r_bot / real_dl) * real_dl


def clear(sampled_map):
    for i in range(sampled_map.shape[0]):
        for j in range(sampled_map.shape[1]):
            if sampled_map[i][j][2]:
                sampled_map[i][j][2] = 0
    return sampled_map


# sampled_map = clear(sampled_map)

if not isClosedSurface:
    for i in range(len(sampled_map)):
        for j in range(len(sampled_map[i])):
            if i == 0 or i == len(sampled_map) - 1 or j == 0 or j == len(
                    sampled_map[i]) - 1:
                sampled_map[i][j][2] = 1


# рашсирение препятсвий для корректного конфигурационного пространства
# такие математические препятсвия имеют цвет >1 (+2)
def increase_barrier(arr, base_color=1, set_color=2, isD=1):
    # увеличивает препт. на 1 дискр. с каждой стороны
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j][2] == base_color:
                try:
                    if not arr[i - 1][j][2]:
                        arr[i - 1][j][2] += set_color  # |^
                except IndexError:
                    pass
                try:
                    if not arr[i - 1][j + 1][2] and isD:
                        arr[i - 1][j + 1][2] += set_color  # /^
                except IndexError:
                    pass
                try:
                    if not arr[i][j + 1][2]:
                        arr[i][j + 1][2] += set_color  # ->
                except IndexError:
                    pass
                try:
                    if not arr[i + 1][j + 1][2] and isD:
                        arr[i + 1][j + 1][2] += set_color  # \.
                except IndexError:
                    pass
                try:
                    if not arr[i + 1][j][2]:
                        arr[i + 1][j][2] += set_color  # |.
                except IndexError:
                    pass
                try:
                    if not arr[i + 1][j - 1][2] and isD:
                        arr[i + 1][j - 1][2] += set_color  # /.
                except IndexError:
                    pass
                try:
                    if not arr[i][j - 1][2]:
                        arr[i][j - 1][2] += set_color  # <-
                except IndexError:
                    pass
                try:
                    if not arr[i - 1][j - 1][2] and isD:
                        arr[i - 1][j - 1][2] += set_color  # ^\
                except IndexError:
                    pass
    return arr


h_count = math.floor(r_h / 2 / real_dl)
w_count = math.floor(r_w / 2 / real_dl)
# d_count = math.floor((h_count ** 2 + w_count ** 2) ** 0.5)
# d_count = math.floor((h_count + w_count) / 2)
d_count = max(min(h_count, w_count) - 1, 0)


def minkowski_space(arr, set_color=2):
    base_color = 2
    set_color = 2
    for h in range(1, h_count + 1):
        set_color = 2 + h / 10
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j][2] == 1 or arr[i][j][2] == base_color:
                    try:
                        if not arr[i - 1][j][2]:
                            arr[i - 1][j][2] = set_color  # |^
                    except IndexError:
                        pass
                    try:
                        if not arr[i + 1][j][2]:
                            arr[i + 1][j][2] = set_color  # |.
                    except IndexError:
                        pass
        base_color = 2 + h / 10
    base_color = 3
    set_color = 3
    for w in range(1, w_count + 1):
        set_color = 3 + w / 10
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j][2] == 1 or arr[i][j][2] == base_color or \
                        2 <= arr[i][j][2] < 3:
                    try:
                        if not arr[i][j + 1][2]:
                            arr[i][j + 1][2] = set_color  # ->
                    except IndexError:
                        pass
                    try:
                        if not arr[i][j - 1][2]:
                            arr[i][j - 1][2] = set_color  # <-
                    except IndexError:
                        pass
        base_color = 3 + w / 10

    base_color = 4
    set_color = 4
    for d in range(1, d_count + 1):
        set_color = 4 + d / 10
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j][2] == 1 or arr[i][j][2] == base_color or 2 <= \
                        arr[i][j][2] < 4:
                    try:
                        if not arr[i - 1][j + 1][2]:
                            arr[i - 1][j + 1][2] = set_color  # /^
                    except IndexError:
                        pass
                    try:
                        if not arr[i + 1][j + 1][2]:
                            arr[i + 1][j + 1][2] = set_color  # \.
                    except IndexError:
                        pass
                    try:
                        if not arr[i + 1][j - 1][2]:
                            arr[i + 1][j - 1][2] += set_color  # /.
                    except IndexError:
                        pass
                    try:
                        if not arr[i - 1][j - 1][2]:
                            arr[i - 1][j - 1][2] += set_color  # ^\
                    except IndexError:
                        pass
        base_color = 4 + d / 10
    return arr


last_col = 1
if isMinkowski:
    sampled_map = minkowski_space(sampled_map)
else:
    for i in range(math.ceil((n_dl - 1) / 2)):
        last_col = 2 ** (i + 1)
        sampled_map = increase_barrier(sampled_map, 2 ** i, last_col)

# sampled_map = increase_barrier(sampled_map, last_col, 0.8, 0)

# print(size)
# print(dl)
# print(sampled_map.shape)

os.environ['SDL_VIDEO_WINDOW_POS'] = str(
    int(display_info.current_w - size[0] - 21)) + "," + str(42)
screen = pygame.display.set_mode(size)

pygame.display.set_caption(
    "'LMB' - set Start, 'RMB' - set Finish, 'D' - Dijkstra, 'A'+'E' - A*E, 'A'+'C' - A*C 'A'+'M' - A*M, ~bA* ('B'), 'R'+'NUM' - RRT (tlr=NUM), ~RRT* ('T')")


def draw_bot(ind_x, ind_y, w=0):  # аналог j и i
    x_r = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0] + dl / 2
    y_r = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1] + dl / 2
    if not isMinkowski:
        pygame.draw.circle(screen, yellow, (x_r, y_r),
                           max(((dl + margin) * n_dl) / 2, dl / 2))
    else:
        pygame.draw.rect(screen, yellow,
                         (x_r - d_r_w / 2, y_r - d_r_l / 2, d_r_w, d_r_l))


def draw_track(ind_x, ind_y):
    x_t = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0] + dl / 2
    y_t = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1] + dl / 2
    radius = dl / 2 * 0.5
    if radius > ((dl + margin) * n_dl) / 2:
        radius = ((dl + margin) * n_dl) / 2
    pygame.draw.circle(screen, (255, 255, 0), (x_t, y_t), radius)


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
            if arr[i][j][2] < 1 and arr[i][j][2] != 0.8:
                arr[i][j][2] = 0
    return arr


def draw_node(node, value):
    if value == 0.25:
        if not sampled_map[int(node) // sampled_map.shape[1]][
            int(node) % sampled_map.shape[1]][2]:
            sampled_map[int(node) // sampled_map.shape[1]][
                int(node) % sampled_map.shape[1]][2] = value
            return True
        else:
            return False
    elif value == 0.5:
        if not sampled_map[int(node) // sampled_map.shape[1]][
            int(node) % sampled_map.shape[1]][2] or \
                sampled_map[int(node) // sampled_map.shape[1]][
                    int(node) % sampled_map.shape[1]][2] == 0.25:
            sampled_map[int(node) // sampled_map.shape[1]][
                int(node) % sampled_map.shape[1]][2] = 0.5
            return True
        else:
            return False
    elif value == 0.75:
        if sampled_map[int(node) // sampled_map.shape[1]][
            int(node) % sampled_map.shape[1]][2] < 1:
            sampled_map[int(node) // sampled_map.shape[1]][
                int(node) % sampled_map.shape[1]][2] = 0.75


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
    node_counter = 1

    k = 0
    while k < graph_size:
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
                    if draw_node(node, 0.25):
                        node_counter += 1
            unvisited_nodes = np.delete(unvisited_nodes, 0, axis=0)
            if draw_node(curr_node[0], 0.5):
                node_counter += 1
        k += 1

    print('< Изучено вершин: {} >'.format(node_counter))
    path = pathes[f]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(
            round(min_label[np.where(min_label == f)[0][0]][1], 3),
            len(path) - 1))
    else:
        path = []
        print('Путь не существует!')
    return path


def a_star(s, f, h_type, isMain=1):
    if s == f:
        print(
            '< A*({}) даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                h_type, graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (f % sampled_map.shape[1], f // sampled_map.shape[1])
        if isClosedSurface:
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))
            if h_type == 'e':
                return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
                        koef_xy[1] * (
                        xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            elif h_type == 'c':
                return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])),
                           abs(koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
            elif h_type == 'm':
                return abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) + abs(
                    koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1]))
        else:
            if h_type == 'e':
                return ((xy_o[0] - xy_i[0]) ** 2 + (
                        xy_o[1] - xy_i[1]) ** 2) ** 0.5
            elif h_type == 'c':
                return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))
            elif h_type == 'm':
                return abs(xy_o[0] - xy_i[0]) + abs(xy_o[1] - xy_i[1])

    print(
        '< A*({}) ищет путь среди {} вершин и {} рёбер... >'.format(h_type,
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
    node_counter = 0
    main_node_counter = 1

    cur_node = s
    draw_node(s, 0.5)
    while unvisited_nodes.shape[0] > 0:
        if cur_node == f and len(pathes[f]) > 1:
            # print(cur_node,unvisited_nodes[0],unvisited_nodes[1])
            break
        elif cur_node == f and unvisited_nodes.shape[0] > 1:
            cur_node = unvisited_nodes[1][0]
            # print("eeee", unvisited_nodes.shape)
            unvisited_nodes[np.where(unvisited_nodes == f)[0][0]][1] = float(
                'inf')
            unvisited_nodes[np.where(unvisited_nodes == f)[0][0]][3] = float(
                'inf')
        elif cur_node == f:
            # print("fffff",unvisited_nodes.shape)
            # print(' Путь не удалось найти')
            break

        to_vist = [n for n in G.neighbors(int(cur_node))]
        for i, node in enumerate(to_vist):
            try:
                ind = np.where(unvisited_nodes == node)[0][0]
            except IndexError:
                continue
            label = \
                min_label[np.where(min_label == cur_node)[0][0]][1] + \
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
            if isMain:
                if draw_node(node, 0.25):
                    node_counter += 1

        unvisited_nodes = np.delete(unvisited_nodes,
                                    np.where(unvisited_nodes == cur_node)[0][
                                        0], axis=0)

        if unvisited_nodes.shape[0] > 0:
            unvisited_nodes = unvisited_nodes[
                unvisited_nodes[:, 3].argsort()[::1]]
            cur_node = unvisited_nodes[0][0]
        # if draw_node(cur_node, 0.5):
        #     node_counter += 1
        if isMain:
            draw_node(cur_node, 0.5)
        main_node_counter += 1

    print('< Увидено вершин: {} >'.format(node_counter))
    print('< Изучено вершин: {} >'.format(main_node_counter))
    path = pathes[f]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(
            round(min_label[np.where(min_label == f)[0][0]][1], 3),
            len(path) - 1))
    else:
        path = []
        print('Путь не существует!')
    return path


def strange_a_star(s, f, h_type='e'):
    if s == f:
        print(
            '< Странный {}A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                h_type, graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]

    def huer(currnet_node):
        global isClosedSurface
        xy_o = (currnet_node % sampled_map.shape[1],
                currnet_node // sampled_map.shape[1])
        xy_i = (f % sampled_map.shape[1], f // sampled_map.shape[1])
        if isClosedSurface:  # для тора
            # return 0  # 0 на торе стабильнее синусов :)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_o_sin = (math.sin(xy_o[0] / sampled_map.shape[1]),
                        math.sin(xy_o[1] / sampled_map.shape[0]))
            xy_i_sin = (math.sin(xy_i[0] / sampled_map.shape[1]),
                        math.sin(xy_i[1] / sampled_map.shape[0]))

            if h_type == 'e':
                return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
                        koef_xy[1] * (
                        xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            elif h_type == 'c':
                return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])),
                           abs(koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
            elif h_type == 'm':
                return abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) + abs(
                    koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1]))
        else:
            if h_type == 'e':
                return ((xy_o[0] - xy_i[0]) ** 2 + (
                        xy_o[1] - xy_i[1]) ** 2) ** 0.5
            elif h_type == 'c':
                return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))
            elif h_type == 'm':
                return abs(xy_o[0] - xy_i[0]) + abs(xy_o[1] - xy_i[1])

    print(
        '< Странный {}A* ищет путь среди {} вершин и {} рёбер... >'.format(
            h_type,
            graph_size,
            G.number_of_edges()))
    min_label = np.ndarray(shape=(0, 4), dtype=float)
    need_to_consider = np.ndarray(shape=(0, 4), dtype=float)
    considered = np.ndarray(shape=(0, 4), dtype=float)
    considered = np.append(considered, [
        [float('inf'), float('inf'), float('inf'), float('inf')]], axis=0)
    for i in range(graph_size):
        to_append = [i, float('inf'), huer(i), float('inf')]
        min_label = np.append(min_label, [to_append], axis=0)
    min_label[s][1] = 0
    min_label[s][3] = huer(s) + 0
    need_to_consider = np.append(need_to_consider,
                                 [[s, 0, huer(s), huer(s) + 0]], axis=0)
    pathes = [[s] for i in range(graph_size)]
    k = 1
    cur_node = s  # ни на что не влияет, просто иниц.
    while len(need_to_consider) > 0:
        need_to_consider = need_to_consider[
            need_to_consider[:, 3].argsort()[::1]]
        cur_node = need_to_consider[0][0]
        if cur_node == f:
            break
        to_vist = [n for n in G.neighbors(int(cur_node))]

        b = need_to_consider[np.where(need_to_consider == cur_node)[0][0]][1]
        c = need_to_consider[np.where(need_to_consider == cur_node)[0][0]][2]
        d = need_to_consider[np.where(need_to_consider == cur_node)[0][0]][3]
        cons_arr = [cur_node, b, c, d]

        considered = np.append(considered, [cons_arr], axis=0)

        for i, node in enumerate(to_vist):
            label = considered[np.where(considered == cur_node)[0][0]][1] + \
                    G.edges[[cur_node, node]]['weight']

            if label >= min_label[np.where(min_label == node)[0][0]][1] and \
                    np.extract(considered == node, considered).shape[0]:
                continue
            draw_node(node, 0.25)

            if label < min_label[np.where(min_label == node)[0][0]][1] or not \
                    np.extract(considered == node, considered).shape[0]:
                min_label[np.where(min_label == node)[0][0]][1] = label
                min_label[np.where(min_label == node)[0][0]][3] = label + \
                                                                  min_label[
                                                                      np.where(
                                                                          min_label == node)[
                                                                          0][
                                                                          0]][
                                                                      2]
                pathes[node] = pathes[int(cur_node)] + [int(node)]
                if not \
                        np.extract(need_to_consider == node,
                                   need_to_consider).shape[
                            0]:
                    need_to_consider = np.append(need_to_consider, [
                        [node, min_label[np.where(min_label == node)[0][0]][1],
                         min_label[np.where(min_label == node)[0][0]][2],
                         min_label[np.where(min_label == node)[0][0]][3]]],
                                                 axis=0)
        need_to_consider = np.delete(need_to_consider,
                                     np.where(need_to_consider == cur_node)[0][
                                         0], axis=0)
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


def old_a_star(s, f):
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
            '< олд A* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
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
        '< олд A* ищет путь среди {} вершин и {} рёбер... >'.format(graph_size,
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


def bidirect_a_star(s, f, h_type):
    if s == f:
        print(
            '< Двунаправленный A*({}) даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                h_type, graph_size, G.number_of_edges()))
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
            if h_type == 'e':
                return ((koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) ** 2 + (
                        koef_xy[1] * (
                        xy_o_sin[1] - xy_i_sin[1])) ** 2) ** 0.5
            elif h_type == 'c':
                return max(abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])),
                           abs(koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1])))
            elif h_type == 'm':
                return abs(koef_xy[0] * (xy_o_sin[0] - xy_i_sin[0])) + abs(
                    koef_xy[1] * (xy_o_sin[1] - xy_i_sin[1]))
        else:
            if h_type == 'e':
                return ((xy_o[0] - xy_i[0]) ** 2 + (
                        xy_o[1] - xy_i[1]) ** 2) ** 0.5
            elif h_type == 'c':
                return max(abs(xy_o[0] - xy_i[0]), abs(xy_o[1] - xy_i[1]))
            elif h_type == 'm':
                return abs(xy_o[0] - xy_i[0]) + abs(xy_o[1] - xy_i[1])

    print(
        '< Двунаправленный A*({}) ищет путь среди {} вершин и {} рёбер... >'.format(
            h_type, graph_size, G.number_of_edges()))
    min_label_s = np.ndarray(shape=(0, 4), dtype=float)
    min_label_f = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes_s = np.ndarray(shape=(0, 4), dtype=float)
    unvisited_nodes_f = np.ndarray(shape=(0, 4), dtype=float)
    for i in range(graph_size):
        to_append_s = [i, float('inf'), huer(i, f), float('inf')]
        to_append_f = [i, float('inf'), huer(i, s), float('inf')]
        min_label_s = np.append(min_label_s, [to_append_s], axis=0)
        min_label_f = np.append(min_label_f, [to_append_f], axis=0)
        unvisited_nodes_s = np.append(unvisited_nodes_s, [to_append_s], axis=0)
        unvisited_nodes_f = np.append(unvisited_nodes_f, [to_append_f], axis=0)
    min_label_s[s][1] = 0
    min_label_f[f][1] = 0
    unvisited_nodes_s[s][1] = 0
    unvisited_nodes_f[f][1] = 0
    pathes_s = [[s] for i in range(graph_size)]
    pathes_f = [[f] for i in range(graph_size)]
    node_counter_s = 0
    node_counter_f = 0
    main_node_coun_s = 1
    main_node_coun_f = 1

    cur_node_s = s
    cur_node_f = f
    draw_node(s, 0.5)
    draw_node(f, 0.5)

    def search(min_label, unvisited_nodes, pathes, cur_node, s, f,
               node_counter, main_node_counter):
        while unvisited_nodes.shape[0] > 0:
            if cur_node == f and len(pathes[f]) > 1:
                # print(cur_node,unvisited_nodes[0],unvisited_nodes[1])
                break
            elif cur_node == f and unvisited_nodes.shape[0] > 1:
                cur_node = unvisited_nodes[1][0]
                # print("eeee", unvisited_nodes.shape)
                unvisited_nodes[np.where(unvisited_nodes == f)[0][0]][
                    1] = float(
                    'inf')
                unvisited_nodes[np.where(unvisited_nodes == f)[0][0]][
                    3] = float(
                    'inf')
            elif cur_node == f:
                # print("fffff",unvisited_nodes.shape)
                # print(' Путь не удалось найти')
                break

            to_vist = [n for n in G.neighbors(int(cur_node))]
            for i, node in enumerate(to_vist):
                try:
                    ind = np.where(unvisited_nodes == node)[0][0]
                except IndexError:
                    continue
                label = \
                    min_label[np.where(min_label == cur_node)[0][0]][1] + \
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
                if draw_node(node, 0.25):
                    node_counter += 1

            unvisited_nodes = np.delete(unvisited_nodes,
                                        np.where(unvisited_nodes == cur_node)[
                                            0][
                                            0], axis=0)
            if unvisited_nodes.shape[0] > 0:
                unvisited_nodes = unvisited_nodes[
                    unvisited_nodes[:, 3].argsort()[::1]]
                cur_node = unvisited_nodes[0][0]
            draw_node(cur_node, 0.5)
            main_node_counter += 1
        return [min_label[np.where(min_label == f)[0][0]][1], pathes[f],
                node_counter, main_node_counter]

    s_to_f = search(min_label_s, unvisited_nodes_s, pathes_s, cur_node_s, s, f,
                    node_counter_s, main_node_coun_s)
    f_to_s = search(min_label_f, unvisited_nodes_f, pathes_f, cur_node_f, f, s,
                    node_counter_f, main_node_coun_f)
    # node_counter = min(s_to_f[2] + f_to_s[2], quan_free_nodes)
    print('< Увидено вершин: {} >'.format(s_to_f[2] + f_to_s[2]))
    print('< Изучено вершин: {} >'.format(s_to_f[3] + f_to_s[3]))
    if s_to_f[0] > f_to_s[0]:
        s_to_f = f_to_s
        path = list(reversed(s_to_f[1]))
    else:
        path = s_to_f[1]
    if len(path) > 1:
        print('Длина пути: {} м, шагов: {}'.format(round(s_to_f[0], 3),
                                                   len(path) - 1))
    else:
        path = []
        print('Путь не существует!')
    return path


def rrt(s, f, dist_f, max_d=max_d_rrt):
    max_samples = int(max(0.02 * graph_size, max_samples_rrt))
    if s == f:
        print(
            '< RRT даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]
    print(
        '< RRT (tlr = {} м) ищет путь среди {} вершин и {} рёбер... >'.format(
            round(dist_f, 3), graph_size, G.number_of_edges()))

    # dist_f = min(dist_f, r_bot)

    def coord_by_node(node):  # имя узла по центру клетки
        x_ind = node % int(sampled_map.shape[1])
        y_ind = node // int(sampled_map.shape[1])
        x = x_ind * real_dl + real_dl / 2
        y = y_ind * real_dl + real_dl / 2
        return [x, y]

    def node_by_coord(x, y):  # центр клетки по имени узла
        x_ind = (x - real_dl / 2) // real_dl
        y_ind = (y - real_dl / 2) // real_dl
        return int(y_ind * sampled_map.shape[1] + x_ind)

    def nearest_node_in_graph(node, Gr):
        distance = float('inf')
        n_node = None
        xy_targ = coord_by_node(node)
        for n in Gr:
            xy_cur = coord_by_node(n)
            cur_dist = ((xy_targ[0] - xy_cur[0]) ** 2 + (
                    xy_targ[1] - xy_cur[1]) ** 2) ** 0.5
            if cur_dist < distance:
                n_node = n
                distance = cur_dist
        return n_node

    def find_child(parent_node, xy_parent, ang):
        if math.radians(-22.5) <= ang < math.radians(22.5):  # right
            if parent_node % int(sampled_map.shape[1]) == sampled_map.shape[
                1] - 1:
                child_node = parent_node - sampled_map.shape[1] + 1
                # print('hur')
            else:
                child_node = parent_node + 1
        elif math.radians(22.5) <= ang < math.radians(67.5):  # rd
            if parent_node % int(sampled_map.shape[1]) == sampled_map.shape[
                1] - 1:
                child_node = (parent_node + 1 + graph_size) % graph_size
                # print('hur')
            else:
                child_node = (parent_node + sampled_map.shape[
                    1] + 1) % graph_size
            # child_node = parent_node + sampled_map.shape[1] + 1
        elif math.radians(67.5) <= ang < math.radians(112.5):  # d
            child_node = (parent_node + sampled_map.shape[1]) % graph_size
        elif math.radians(112.5) <= ang < math.radians(157.5):  # ld
            if not parent_node % int(sampled_map.shape[1]):
                child_node = (parent_node + 2 * sampled_map.shape[
                    1] - 1) % graph_size
                # print('hur')
            else:
                child_node = (parent_node + sampled_map.shape[
                    1] - 1) % graph_size
            # child_node = parent_node + sampled_map.shape[1] - 1
        elif math.radians(157.5) <= ang <= math.radians(180) or math.radians(
                -180) <= ang < math.radians(-157.5):  # l
            if not parent_node % int(sampled_map.shape[1]):
                child_node = parent_node - 1 + sampled_map.shape[1]
                # print('hur')
            else:
                child_node = parent_node - 1
        elif math.radians(-157.5) <= ang < math.radians(-112.5):  # lu
            if not parent_node % int(sampled_map.shape[1]):
                child_node = (parent_node - 1 + graph_size) % graph_size
                # print('hur')
            else:
                child_node = (parent_node - sampled_map.shape[
                    1] - 1 + graph_size) % graph_size
        elif math.radians(-112.5) <= ang < math.radians(-67.5):  # u
            child_node = (parent_node - sampled_map.shape[
                1] + graph_size) % graph_size
        elif math.radians(-67.5) <= ang < math.radians(-22.5):  # ru
            if parent_node % int(sampled_map.shape[1]) == sampled_map.shape[
                1] - 1:
                child_node = (parent_node - sampled_map.shape[
                    1] + graph_size) % graph_size - sampled_map.shape[1] + 1
                # print('hur')
            else:
                child_node = (parent_node - sampled_map.shape[
                    1] + graph_size) % graph_size + 1
        return child_node

    def distance_bt_nodes(a, b):
        xy_a = coord_by_node(a)
        xy_b = coord_by_node(b)
        return ((xy_a[0] - xy_b[0]) ** 2 + (xy_a[1] - xy_b[1]) ** 2) ** 0.5

    T = nx.Graph()
    T.add_node(s)
    # пути до каждой вершины дерева (Т) из старта
    pathes = [[s] for i in range(G.number_of_nodes())]
    draw_node(s, 0.75)

    node_counter = 0
    rand_node_coun = 0
    end_node_coun = 1
    isNotInToler = True

    def search_path(max_d=max_d):
        nonlocal pathes, node_counter, rand_node_coun, end_node_coun, isNotInToler
        while isNotInToler and rand_node_coun < max_samples:
            # выбираем узел вне дерева
            xy_rand = coord_by_node(s)
            rand_node = s
            while rand_node in T.nodes() or sampled_map[
                min(rand_node // sampled_map.shape[1],
                    sampled_map.shape[0] - 1)][
                min(rand_node % sampled_map.shape[1],
                    sampled_map.shape[1] - 1)][2] >= 1:
                xy_rand[0] = random.random() * sampled_map.shape[1] * real_dl
                xy_rand[1] = random.random() * sampled_map.shape[0] * real_dl
                rand_node = node_by_coord(xy_rand[0], xy_rand[1])
            draw_node(rand_node, 0.25)
            rand_node_coun += 1
            # находим ближайший в дереве
            parent_node = nearest_node_in_graph(rand_node, T)
            xy_parent = coord_by_node(parent_node)
            # r = ((xy_rand[0] - xy_parent[0]) ** 2 + (
            #         xy_rand[1] - xy_parent[1]) ** 2) ** 0.5
            # if r == 0:
            #     ang = float('inf')
            # else:
            #     ang = (xy_rand[0] - xy_parent[0]) / r

            xy_off = [xy_rand[0] - xy_parent[0], xy_rand[1] - xy_parent[1]]
            ang = math.atan2(xy_off[1], xy_off[0])
            # print('ccor', xy_rand, xy_parent)
            # print(ang)

            if isClosedSurface:
                direc_dist = distance_bt_nodes(rand_node, parent_node)
                koef_xy = (
                    sampled_map.shape[1] / 2 * real_dl,
                    sampled_map.shape[0] / 2 * real_dl)
                xy_rand_sin = [
                    koef_xy[0] * math.sin(xy_rand[0] / sampled_map.shape[1]),
                    koef_xy[1] * math.sin(xy_rand[1] / sampled_map.shape[0])]

                mirr_rand_node = node_by_coord(xy_rand_sin[0], xy_rand_sin[1])
                reverse_dist = distance_bt_nodes(mirr_rand_node, parent_node)
                # print(coord_by_node(mirr_rand_node))
                # print(direc_dist, reverse_dist)
                if reverse_dist <= direc_dist:
                    if ang >= 0:
                        ang -= math.pi
                    else:
                        ang += math.pi
            # print(ang)

            child_node = find_child(parent_node, xy_parent, ang)
            isGoing = True
            while isGoing:  # пока недалеко или не дойдём до выборки или препятствия
                j_x = min(child_node % sampled_map.shape[1],
                          sampled_map.shape[1] - 1)
                i_y = min(child_node // sampled_map.shape[1],
                          sampled_map.shape[0] - 1)

                xy_child = coord_by_node(child_node)
                if ((xy_child[0] - xy_parent[0]) ** 2 + (
                        xy_child[1] - xy_parent[1]) ** 2) ** 0.5 > max_d:
                    isGoing = False
                    draw_node(parent_node, 0.75)
                    # print('далеко ушли', child_node)

                elif sampled_map[i_y][j_x][2] < 1:
                    # if child_node not in T.nodes():
                    T.add_node(child_node)
                    T.add_edge(parent_node, child_node,
                               weight=G.edges[[parent_node, child_node]][
                                   'weight'])
                    draw_node(child_node, 0.5)
                    node_counter += 1
                    pathes[child_node] = pathes[parent_node] + [child_node]
                    if child_node == rand_node:
                        # print('дошли до таргета')
                        node_counter -= 1
                        draw_node(child_node, 0.75)
                        isGoing = False

                    if distance_bt_nodes(child_node, f) < dist_f:
                        isGoing = False
                        isNotInToler = False
                        # print('на финише')
                        # pathes[f] = pathes[parent_node] + [child_node] + [f]
                        pathes[f] = pathes[parent_node] + [child_node]
                        draw_node(child_node, 0.75)
                        if child_node != rand_node:
                            node_counter -= 1
                    else:
                        parent_node = child_node
                        child_node = find_child(child_node,
                                                coord_by_node(child_node), ang)

                else:
                    isGoing = False
                    draw_node(child_node, 0.75)
                    # print('gрепятсвие')
            end_node_coun += 1
        else:
            if isNotInToler:
                f_near_node = nearest_node_in_graph(f, T)
                pathes[f] = pathes[f_near_node]

    search_path()

    if rand_node_coun == max_samples:
        print('< Достигли макисмума случайных выборок: {} >'.format(
            rand_node_coun))
    else:
        print('< Cлучайных выборок: {} >'.format(rand_node_coun))
    print('< Соединительных вершин: {} >'.format(node_counter))
    print('< Концевых вершин: {} >'.format(end_node_coun))
    path = pathes[f]
    len_path = 0
    for i in range(1, len(path)):
        try:
            len_path += T.edges[[path[i - 1], path[i]]]['weight']
        except KeyError:
            continue

    if len(path) > 1:
        if not isNotInToler:
            print('Робот доехал до цели с допуском {} м (ds = {} м)'.format(
                round(dist_f, 3), round(distance_bt_nodes(path[-1], f), 3)))
        else:
            print('< Не дошли до окрестности, дальше ножками! >')
        print('Длина пути: {} м, шагов: {}'.format(round(len_path, 3),
                                                   len(path) - 1))
    else:
        path = []
        print('Путь не существует!')
    return path


def rrt_star(s, f, dist_f, max_d=max_d_rrt):
    max_samples = int(max(0.02 * graph_size, max_samples_rrt))
    if s == f:
        print(
            '< RRT* даже не искал путь среди {} вершин и {} рёбер! Не балуйтесь симулятором... >'.format(
                graph_size, G.number_of_edges()))
        print('Вершины не изучались, длина пути 0 м, шагов пройдено тоже 0')
        return [f]
    print(
        '< RRT* (tlr = {} м) ищет путь среди {} вершин и {} рёбер... >'.format(
            round(dist_f, 3), graph_size, G.number_of_edges()))
    # dist_f = min(dist_f, r_bot)

    T = nx.Graph()
    T.add_node(s)
    # пути до каждой вершины дерева (Т) из старта
    pathes = [[s] for i in range(G.number_of_nodes())]

    def coord_by_node(node):  # имя узла по центру клетки
        x_ind = node % int(sampled_map.shape[1])
        y_ind = node // int(sampled_map.shape[1])
        x = x_ind * real_dl + real_dl / 2
        y = y_ind * real_dl + real_dl / 2
        return [x, y]

    def node_by_coord(x, y):  # центр клетки по имени узла
        x_ind = (x - real_dl / 2) // real_dl
        y_ind = (y - real_dl / 2) // real_dl
        return int(y_ind * sampled_map.shape[1] + x_ind)

    def nearest_node_in_graph(node, Gr):
        distance = float('inf')
        n_node = None
        xy_targ = coord_by_node(node)
        for n in T:
            xy_cur = coord_by_node(n)
            cur_dist = ((xy_targ[0] - xy_cur[0]) ** 2 + (
                    xy_targ[1] - xy_cur[1]) ** 2) ** 0.5
            if cur_dist < distance:
                n_node = n
                distance = cur_dist
        return n_node

    def find_child(parent_node, xy_parent, ang):
        if math.radians(-22.5) <= ang < math.radians(22.5):  # right
            if parent_node % int(sampled_map.shape[1]) == sampled_map.shape[
                1] - 1:
                child_node = parent_node - sampled_map.shape[1] + 1
                # print('hur')
            else:
                child_node = parent_node + 1
        elif math.radians(22.5) <= ang < math.radians(67.5):  # rd
            if parent_node % int(sampled_map.shape[1]) == sampled_map.shape[
                1] - 1:
                child_node = (parent_node + 1 + graph_size) % graph_size
                # print('hur')
            else:
                child_node = (parent_node + sampled_map.shape[
                    1] + 1) % graph_size
            # child_node = parent_node + sampled_map.shape[1] + 1
        elif math.radians(67.5) <= ang < math.radians(112.5):  # d
            child_node = (parent_node + sampled_map.shape[1]) % graph_size
        elif math.radians(112.5) <= ang < math.radians(157.5):  # ld
            if not parent_node % int(sampled_map.shape[1]):
                child_node = (parent_node + 2 * sampled_map.shape[
                    1] - 1) % graph_size
                # print('hur')
            else:
                child_node = (parent_node + sampled_map.shape[
                    1] - 1) % graph_size
            # child_node = parent_node + sampled_map.shape[1] - 1
        elif math.radians(157.5) <= ang <= math.radians(180) or math.radians(
                -180) <= ang < math.radians(-157.5):  # l
            if not parent_node % int(sampled_map.shape[1]):
                child_node = parent_node - 1 + sampled_map.shape[1]
                # print('hur')
            else:
                child_node = parent_node - 1
        elif math.radians(-157.5) <= ang < math.radians(-112.5):  # lu
            if not parent_node % int(sampled_map.shape[1]):
                child_node = (parent_node - 1 + graph_size) % graph_size
                # print('hur')
            else:
                child_node = (parent_node - sampled_map.shape[
                    1] - 1 + graph_size) % graph_size
        elif math.radians(-112.5) <= ang < math.radians(-67.5):  # u
            child_node = (parent_node - sampled_map.shape[
                1] + graph_size) % graph_size
        elif math.radians(-67.5) <= ang < math.radians(-22.5):  # ru
            if parent_node % int(sampled_map.shape[1]) == sampled_map.shape[
                1] - 1:
                child_node = (parent_node - sampled_map.shape[
                    1] + graph_size) % graph_size - sampled_map.shape[1] + 1
                # print('hur')
            else:
                child_node = (parent_node - sampled_map.shape[
                    1] + graph_size) % graph_size + 1
        try:
            return child_node
        except UnboundLocalError:
            print(ang)

    def distance_bt_nodes(a, b):
        xy_a = coord_by_node(a)
        xy_b = coord_by_node(b)
        return ((xy_a[0] - xy_b[0]) ** 2 + (xy_a[1] - xy_b[1]) ** 2) ** 0.5

    def cost_to_node(node):
        nonlocal pathes
        path = pathes[node]
        cost_path = 0
        for i in range(1, len(path)):
            try:
                cost_path += T.edges[[path[i - 1], path[i]]]['weight']
            except KeyError:
                # print('cost attention')
                continue
        return cost_path

    def get_neighbs(node, radius):
        lst_neignbs = []
        for n in T.nodes():
            if distance_bt_nodes(node, n) <= radius and n != node:
                lst_neignbs.append(n)
        return lst_neignbs

    def is_direct_path(st, tr):
        isNotObstacle = True
        l = distance_bt_nodes(st, tr)
        xy_st = coord_by_node(st)
        xy_tr = coord_by_node(tr)
        path_sttr = []

        def find_ij(node):
            j_x = min(node % sampled_map.shape[1],
                      sampled_map.shape[1] - 1)
            i_y = min(node // sampled_map.shape[1],
                      sampled_map.shape[0] - 1)
            return [i_y, j_x]

        # def is_free_area(node):
        #     is_free = True
        #     n_list = [n for n in G.neighbors(int(node))]
        #     for n in n_list:
        #         ij = find_ij(n)
        #         if sampled_map[ij[0]][ij[1]][2] >= 1:
        #             is_free = False
        #     return is_free

        xy_off = [xy_st[0] - xy_tr[0], xy_st[1] - xy_tr[1]]
        ang_d = math.atan2(xy_off[1], xy_off[0])
        if isClosedSurface:
            direc_dist = distance_bt_nodes(st, tr)
            koef_xy = (
                sampled_map.shape[1] / 2 * real_dl,
                sampled_map.shape[0] / 2 * real_dl)
            xy_rand_sin = [
                koef_xy[0] * math.sin(xy_st[0] / sampled_map.shape[1]),
                koef_xy[1] * math.sin(xy_st[1] / sampled_map.shape[0])]

            mirr_st = node_by_coord(xy_rand_sin[0], xy_rand_sin[1])
            reverse_dist = distance_bt_nodes(mirr_st, tr)
            # print(coord_by_node(mirr_rand_node))
            # print(direc_dist, reverse_dist)
            if reverse_dist <= direc_dist:
                if ang_d >= 0:
                    ang_d -= math.pi
                else:
                    ang_d += math.pi

        child_node = find_child(st, xy_st, ang_d)
        ij_cn = find_ij(child_node)
        if sampled_map[ij_cn[0]][ij_cn[1]][2] < 1:
            path_sttr.append(child_node)
        else:
            # print('fff')
            isNotObstacle = False
        parent_node = st

        # print('bfw\n')
        while child_node != tr and isNotObstacle and distance_bt_nodes(
                child_node, tr) < l:
            is_free = True
            n_list = [n for n in G.neighbors(int(child_node))]
            # print(len(n_list))
            if len(n_list) <= 8:
                isNotObstacle = False
            # else:
            #     n_list.append(child_node)
            for n in n_list:
                ij = find_ij(n)
                if sampled_map[ij[0]][ij[1]][2] >= 1:
                    is_free = False

            if is_free:
                # T.add_node(child_node)
                # T.add_edge(parent_node, child_node,
                #            weight=G.edges[[parent_node, child_node]][
                #                'weight'])
                path_sttr.append(child_node)
                parent_node = child_node
                child_node = find_child(parent_node,
                                        coord_by_node(parent_node),
                                        ang_d)
            else:
                isNotObstacle = False
                break
        # else:
        #     isNotObstacle = False
        return [isNotObstacle, path_sttr, child_node]

    draw_node(s, 0.75)

    end_node_coun = 1
    node_counter = 0
    rand_node_coun = 0
    isNotInToler = True
    diffr = 0

    def search_path(max_d=max_d):
        nonlocal pathes, node_counter, rand_node_coun, end_node_coun, isNotInToler, diffr
        while isNotInToler and rand_node_coun < max_samples:
            # if rand_node_coun % int(0.01 * max_samples) == 0:
            #     print('~{}%'.format(
            #         round(rand_node_coun / max_samples * 100, 3)))
            # выбираем узел вне дерева
            xy_rand = coord_by_node(s)
            rand_node = s
            while rand_node in T.nodes():
                #     or sampled_map[
                # min(rand_node // sampled_map.shape[1],
                #     sampled_map.shape[0] - 1)][
                # min(rand_node % sampled_map.shape[1],
                #     sampled_map.shape[1] - 1)][2] >= 1:
                xy_rand[0] = random.random() * sampled_map.shape[1] * real_dl
                xy_rand[1] = random.random() * sampled_map.shape[0] * real_dl
                rand_node = node_by_coord(xy_rand[0], xy_rand[1])
            draw_node(rand_node, 0.25)
            rand_node_coun += 1
            # находим ближайший в дереве
            parent_node = nearest_node_in_graph(rand_node, T)
            xy_parent = coord_by_node(parent_node)

            xy_off = [xy_rand[0] - xy_parent[0], xy_rand[1] - xy_parent[1]]
            ang = math.atan2(xy_off[1], xy_off[0])
            # print('ccor', xy_rand, xy_parent)
            # print(ang)

            if isClosedSurface:
                direc_dist = distance_bt_nodes(rand_node, parent_node)
                koef_xy = (
                    sampled_map.shape[1] / 2 * real_dl,
                    sampled_map.shape[0] / 2 * real_dl)
                xy_rand_sin = [
                    koef_xy[0] * math.sin(xy_rand[0] / sampled_map.shape[1]),
                    koef_xy[1] * math.sin(xy_rand[1] / sampled_map.shape[0])]

                mirr_rand_node = node_by_coord(xy_rand_sin[0], xy_rand_sin[1])
                reverse_dist = distance_bt_nodes(mirr_rand_node, parent_node)
                # print(coord_by_node(mirr_rand_node))
                # print(direc_dist, reverse_dist)
                if reverse_dist <= direc_dist:
                    if ang >= 0:
                        ang -= math.pi
                    else:
                        ang += math.pi
            # print(ang)

            child_node = find_child(parent_node, xy_parent, ang)
            isGoing = True
            while isGoing:  # пока недалеко или не дойдём до выборки или препятствия
                j_x = min(child_node % sampled_map.shape[1],
                          sampled_map.shape[1] - 1)
                i_y = min(child_node // sampled_map.shape[1],
                          sampled_map.shape[0] - 1)

                xy_child = coord_by_node(child_node)
                if ((xy_child[0] - xy_parent[0]) ** 2 + (
                        xy_child[1] - xy_parent[1]) ** 2) ** 0.5 > max_d:
                    isGoing = False
                    draw_node(parent_node, 0.75)
                    # print('далеко ушли', child_node)

                elif sampled_map[i_y][j_x][2] < 1:  # уже не важна принадл. к Т

                    child_neighbs = get_neighbs(child_node, r_find)
                    cost_min = cost_to_node(
                        parent_node) + distance_bt_nodes(parent_node,
                                                         child_node)
                    p = parent_node  # для счётяика (првоерка)
                    # print('old p', parent_node)
                    # print(len(child_neighbs))
                    dop_path = []
                    real_child = child_node
                    for neighb in child_neighbs:
                        # print('is')
                        is_n_path = is_direct_path(neighb, child_node)

                        if is_n_path[0]:
                            # print('TRUE')
                            # neighb_cost = cost_to_node(neighb) + \
                            #               G.edges[[child_node, neighb]][
                            #                   'weight']
                            neighb_cost = cost_to_node(
                                neighb) + distance_bt_nodes(neighb,
                                                            child_node)
                            if neighb_cost < cost_min:
                                # print("hue")
                                cost_min = neighb_cost
                                parent_node = neighb
                                dop_path = is_n_path[1]
                                real_child = is_n_path[2]
                    if p != parent_node:  # для счётчика (проверки)
                        diffr += 1
                    # print('new p', parent_node, '\n')

                    # print('d', distance_bt_nodes(parent_node, child_node))
                    T.add_node(child_node)
                    T.add_edge(parent_node, child_node,
                               weight=distance_bt_nodes(parent_node,
                                                        child_node))
                    # G.add_edge(parent_node, child_node,
                    #            weight=distance_bt_nodes(parent_node,
                    #                                     child_node))
                    if draw_node(child_node, 0.5):
                        node_counter += 1
                    pathes[child_node] = pathes[parent_node] + [child_node]

                    # if p != parent_node:
                    #     addit_path =dop_path
                    #     pathes[child_node] = pathes[
                    #                              parent_node] + addit_path

                    # if p == parent_node:
                    #     pathes[child_node] = pathes[p] + [child_node]
                    # else:
                    #     pathes[child_node] = pathes[parent_node] + is_n_path[1]

                    if child_node == rand_node:
                        draw_node(child_node, 0.75)
                        node_counter -= 1
                        # print('дошли до таргета')
                        isGoing = False

                    if distance_bt_nodes(child_node, f) < dist_f:
                        isGoing = False
                        isNotInToler = False
                        # print('на финише')
                        # pathes[f] = pathes[parent_node] + [child_node] + [f]
                        pathes[f] = pathes[parent_node] + [child_node]
                        draw_node(child_node, 0.75)
                        if child_node != rand_node:
                            node_counter -= 1
                    else:
                        parent_node = child_node
                        child_node = find_child(parent_node,
                                                coord_by_node(parent_node),
                                                ang)

                else:
                    draw_node(parent_node, 0.75)
                    isGoing = False
                    # print('gрепятсвие')
            end_node_coun += 1

        else:
            if isNotInToler:
                f_near_node = nearest_node_in_graph(f, T)
                pathes[f] = pathes[f_near_node]
                # pathes[f] = pathes[parent_node] + [child_node]

    search_path()

    # print(diffr)
    if rand_node_coun == max_samples:
        print('< Достигли макисмума случайных выборок: {} >'.format(
            rand_node_coun))
    else:
        print('< Cлучайных выборок: {} >'.format(rand_node_coun))

    print('< Соединительных вершин : {} >'.format(node_counter))
    print('< Концевых вершин: {} >'.format(end_node_coun))
    path = pathes[f]
    len_path = 0
    for i in range(1, len(path)):
        try:
            len_path += T.edges[[path[i - 1], path[i]]]['weight']
        except KeyError:
            T.add_edge(path[i - 1], path[i],
                       weight=distance_bt_nodes(path[i - 1], path[i]))
            # print('q')
            continue

    if len(path) > 1:
        if not isNotInToler:
            print('Робот доехал до цели с допуском {} м (ds = {} м)'.format(
                round(dist_f, 3), round(distance_bt_nodes(path[-1], f), 3)))
        else:
            print('< Не дошли до окрестности, дальше ножками! >')
        print('Длина пути: {} м, шагов: {}'.format(round(len_path, 3),
                                                   len(path) - 1))
    else:
        path = []
        print('Путь не существует!')
    return path


G = nx.Graph()
for n in range(sampled_map.shape[0] * sampled_map.shape[1]):
    G.add_node(n)
graph_size = G.number_of_nodes()


def check_up(name, i, j):
    if not sampled_map[i - 1][j][2] // 1:  # верх
        G.add_edge(name,
                   (name - sampled_map.shape[1] + graph_size) % graph_size,
                   weight=w_adj)


def check_right(name, i, j):
    if not sampled_map[i][(j + 1) % len(sampled_map[i])][2] // 1:  # право
        if j == len(sampled_map[i]) - 1:
            G.add_edge(name, name - sampled_map.shape[1] + 1, weight=w_adj)
        else:
            G.add_edge(name, name + 1, weight=w_adj)


def check_down(name, i, j):
    if not sampled_map[i + 1][j][2] // 1:  # низ
        G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)


def check_left(name, i, j):
    if not sampled_map[i][j - 1][2] // 1:  # лево
        if j == 0:
            G.add_edge(name, name - 1 + sampled_map.shape[1], weight=w_adj)
        else:
            G.add_edge(name, name - 1, weight=w_adj)


def check_ru(name, i, j):
    if not sampled_map[i - 1][(j + 1) % len(sampled_map[i])][2] // 1:  # пв
        if j == len(sampled_map[i]) - 1:
            G.add_edge(name, (name - sampled_map.shape[
                1] + graph_size) % graph_size - sampled_map.shape[1] + 1,
                       weight=w_diag)
        else:
            G.add_edge(name, (name - sampled_map.shape[
                1] + graph_size) % graph_size + 1, weight=w_diag)


def check_rd(name, i, j):
    if not sampled_map[i + 1][j + 1][2] // 1:  # пн
        G.add_edge(name, name + sampled_map.shape[1] + 1, weight=w_diag)


def check_ld(name, i, j):
    if not sampled_map[i + 1][j - 1][2] // 1:  # лн
        G.add_edge(name, name + sampled_map.shape[1] - 1, weight=w_diag)


def check_lu(name, i, j):
    if not sampled_map[i - 1][j - 1][2] // 1:  # лв
        if j == 0:
            G.add_edge(name,
                       (name - 1 + graph_size) % graph_size, weight=w_diag)
        else:
            G.add_edge(name, (name - sampled_map.shape[
                1] + graph_size) % graph_size - 1, weight=w_diag)


# проверка по часовой (сверху/ пв или ближайший к этим)
for i in range(len(sampled_map)):
    for j in range(len(sampled_map[i])):
        if not sampled_map[i][j][2] // 1:
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
green_dark, green_light = (0, 156, 0), (0, 210, 0)
gray, gray_1, gray_2 = (128, 128, 128), (128, 128, 192), (192, 128, 128)
color_f, color_s = (0, 0, 0), (224, 224, 224)
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
        elif event.type == pygame.KEYDOWN and not isInProgress:
            keys = pygame.key.get_pressed()
            if isSetObj[0] * isSetObj[1]:
                if keys[pygame.K_d]:
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
                elif keys[pygame.K_a] and keys[pygame.K_e]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], 'e')
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_a] and keys[pygame.K_m]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], 'm')
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_a] and keys[pygame.K_c]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], 'c')
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_a] and keys[pygame.K_o]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = old_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_a] and keys[pygame.K_s]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = strange_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0])
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_b] and keys[pygame.K_e]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = bidirect_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], 'e')
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_b] and keys[pygame.K_c]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = bidirect_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], 'c')
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_b] and keys[pygame.K_m]:
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = bidirect_a_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], 'm')
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_0]:
                    toler = 0
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_1]:
                    toler = 1 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_2]:
                    toler = 2 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_3]:
                    toler = 3 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_4]:
                    toler = 4 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_5]:
                    toler = 5 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_6]:
                    toler = 6 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_7]:
                    toler = 7 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_8]:
                    toler = 8 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_r] and keys[pygame.K_9]:
                    toler = 9 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_0]:
                    toler = 0
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_1]:
                    toler = 1 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_2]:
                    toler = 2 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_3]:
                    toler = 3 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_4]:
                    toler = 4 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_5]:
                    toler = 5 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_6]:
                    toler = 6 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_7]:
                    toler = 7 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_8]:
                    toler = 8 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
                    path_track = []
                    isNewEnvir = False
                    fps = fps_move
                elif keys[pygame.K_t] and keys[pygame.K_9]:
                    toler = 9 * real_dl
                    redraw_map(sampled_map)
                    isNewEnvir = True
                    isInProgress = True
                    path = rrt_star(
                        index_start[1] * sampled_map.shape[1] + index_start[0],
                        index_finish[1] * sampled_map.shape[1] + index_finish[
                            0], toler)
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
            elif sampled_map[i][j][2] == 0.75:
                color = (0, 96, 0)
            elif sampled_map[i][j][2] == 0.8:
                color = gray
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
