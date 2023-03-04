import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygame

sampled_map = np.load('Sampled_map.npy')
colmns, rows = sampled_map.shape[1], sampled_map.shape[0]
prelim_size = (1280, 760)
margin = int(3)
offset = (0, 0)
dl = min(prelim_size[0] // colmns - margin,
         (prelim_size[1] - 2 * offset[1]) // rows - margin)
size = ((dl + margin) * colmns, (dl + margin) * rows)
koef_obj = 0.85  # коэффициент размера объектов в клетке
w_adj, w_diag = 1, round(2 ** 0.5,
                         1)  # веса рёбер для соседних и диагональных вершин
print(size)
print(dl)
print(sampled_map.shape)

pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption(
    "Click 'LMB' to set Start position, 'RMB' - Finish position, then press 'D', 'A', 'R' to start algorithm of Dijkstra, A* or RRT respectively.")


def draw_bot(ind_x, ind_y, w=0):  # аналог j и i
    x_r = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0] + dl / 2
    y_r = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1] + dl / 2
    pygame.draw.circle(screen, yellow, (x_r, y_r), dl / 2 * 0.95)


def draw_finish(ind_x, ind_y):
    ds = dl * koef_obj
    x_t = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0]
    y_t = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1]
    pygame.draw.rect(screen, green,
                     (x_t + (dl - ds) / 2, y_t + (dl - ds) / 2, ds, ds))


def draw_start(ind_x, ind_y):
    ds = dl * koef_obj
    x_t = sampled_map[ind_y][ind_x][0] - sampled_map[0][0][0] + (
            dl + margin) * ind_x + offset[0]
    y_t = -sampled_map[ind_y][ind_x][1] + sampled_map[0][0][1] + (
            dl + margin) * ind_y + offset[1]
    pygame.draw.rect(screen, pink,
                     (x_t + (dl - ds) / 2, y_t + (dl - ds) / 2, ds, ds))


def dijkstra():
    print('start D')
    pass


def a_star():
    print('start A*')
    pass


def rrt():
    print('start RRT')
    pass


G = nx.Graph()
for n in range(sampled_map.shape[0] * sampled_map.shape[1]):
    G.add_node(n)

# проверка по часовой (сверху/ пв или ближайший к этим)
for i in range(len(sampled_map)):
    for j in range(len(sampled_map[i])):
        name = sampled_map.shape[1] * i + j
        if i == 0:  # если верхняя строка
            if j == 0:  # если левый столбей
                # name = 0
                if not sampled_map[i][j + 1][2]:
                    G.add_edge(name, name + 1, weight=w_adj)
                if not sampled_map[i + 1][j][2]:
                    G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)
                # diag
                if not sampled_map[i][j + 1][2] and not sampled_map[i + 1][j][
                    2] and not sampled_map[i + 1][j + 1][2]:
                    G.add_edge(name, name + sampled_map.shape[1] + 1,
                               weight=w_diag)
            elif j == sampled_map.shape[1] - 1:  # если правый столбец
                # name = sampled_map.shape[1] - 1
                if not sampled_map[i + 1][j][2]:
                    G.add_edge(name, name + sampled_map.shape[1],
                               weight=w_adj)
                if not sampled_map[i][j - 1][2]:
                    G.add_edge(name, name - 1, weight=w_adj)
                # diag
                if not sampled_map[i + 1][j][2] and not sampled_map[i][j - 1][
                    2] and not sampled_map[i + 1][j - 1][2]:
                    G.add_edge(name, name + sampled_map.shape[1] - 1,
                               weight=w_diag)
            else:  # если остальное (не пров. верх. и пв и лв)
                # name = j
                if not sampled_map[i][j + 1][2]:  # право
                    G.add_edge(name, name + 1, weight=w_adj)
                if not sampled_map[i + 1][j][2]:  # низ
                    G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j - 1][2]:  # лево
                    G.add_edge(name, name - 1, weight=w_adj)
                # диаг
                if not sampled_map[i][j + 1][2] and not sampled_map[i + 1][j][
                    2] and not sampled_map[i + 1][j + 1][2]:  # пн
                    G.add_edge(name, name + sampled_map.shape[1] + 1,
                               weight=w_diag)
                if not sampled_map[i + 1][j][2] and not sampled_map[i][j - 1][
                    2] and not sampled_map[i + 1][j - 1][2]:  # лн
                    G.add_edge(name, name + sampled_map.shape[1] - 1,
                               weight=w_diag)


        elif i == sampled_map.shape[0] - 1:  # если нижняя строка
            if j == 0:  # если левый столбей
                if not sampled_map[i - 1][j][2]:  # верх
                    G.add_edge(name, name - sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j + 1][2]:  # право
                    G.add_edge(name, name + 1, weight=w_adj)
                # diag
                if not sampled_map[i - 1][j][2] and not sampled_map[i][j + 1][
                    2] and not sampled_map[i - 1][j + 1][2]:  # пв
                    G.add_edge(name, name - sampled_map.shape[1] + 1,
                               weight=w_diag)
            elif j == sampled_map.shape[1] - 1:  # если правый столбец
                if not sampled_map[i - 1][j][2]:  # верх
                    G.add_edge(name, name - sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j - 1][2]:  # лево
                    G.add_edge(name, name - 1, weight=w_adj)
                # diag
                if not sampled_map[i][j - 1][2] and not sampled_map[i - 1][j][
                    2] and not sampled_map[i - 1][j - 1][2]:  # лв
                    G.add_edge(name, name - sampled_map.shape[1] - 1)

            else:  # если остальное (кроме низ, пн, лн)
                if not sampled_map[i - 1][j][2]:  # верх
                    G.add_edge(name, name - sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j + 1][2]:  # право
                    G.add_edge(name, name + 1, weight=w_adj)
                if not sampled_map[i][j - 1][2]:  # лево
                    G.add_edge(name, name - 1, weight=w_adj)
                # diag
                if not sampled_map[i - 1][j][2] and not sampled_map[i][j + 1][
                    2] and not sampled_map[i - 1][j + 1][2]:  # пв
                    G.add_edge(name, name - sampled_map.shape[1] + 1,
                               weight=w_diag)
                if not sampled_map[i][j - 1][2] and not sampled_map[i - 1][j][
                    2] and not sampled_map[i - 1][j - 1][2]:  # лв
                    G.add_edge(name, name - sampled_map.shape[1] - 1)

        else:  # любая другая строка
            if j == 0:  # если левый столбей (кроме лево, лн, лв)
                if not sampled_map[i - 1][j][2]:  # верх
                    G.add_edge(name, name - sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j + 1][2]:  # право
                    G.add_edge(name, name + 1, weight=w_adj)
                if not sampled_map[i + 1][j][2]:  # низ
                    G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)
                # diag
                if not sampled_map[i - 1][j][2] and not sampled_map[i][j + 1][
                    2] and not sampled_map[i - 1][j + 1][2]:  # пв
                    G.add_edge(name, name - sampled_map.shape[1] + 1,
                               weight=w_diag)
                if not sampled_map[i][j + 1][2] and not sampled_map[i + 1][j][
                    2] and not sampled_map[i + 1][j + 1][2]:  # пн
                    G.add_edge(name, name + sampled_map.shape[1] + 1,
                               weight=w_diag)
            elif j == sampled_map.shape[
                1] - 1:  # if прав.стл. (кр. право, пв, пн)
                if not sampled_map[i - 1][j][2]:  # верх
                    G.add_edge(name, name - sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i + 1][j][2]:  # низ
                    G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j - 1][2]:  # лево
                    G.add_edge(name, name - 1, weight=w_adj)
                # diag
                if not sampled_map[i + 1][j][2] and not sampled_map[i][j - 1][
                    2] and not sampled_map[i + 1][j - 1][2]:  # лн
                    G.add_edge(name, name + sampled_map.shape[1] - 1,
                               weight=w_diag)
                if not sampled_map[i][j - 1][2] and not sampled_map[i - 1][j][
                    2] and not sampled_map[i - 1][j - 1][2]:  # лв
                    G.add_edge(name, name - sampled_map.shape[1] - 1)
            else:  # если остальное
                # name = sampled_map.shape[1] * i + j
                # проверяем по часовой сверху, сначала прямые, потом диагонали
                if not sampled_map[i - 1][j][2]:  # верх
                    G.add_edge(name, name - sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j + 1][2]:  # право
                    G.add_edge(name, name + 1, weight=w_adj)
                if not sampled_map[i + 1][j][2]:  # низ
                    G.add_edge(name, name + sampled_map.shape[1], weight=w_adj)
                if not sampled_map[i][j - 1][2]:  # лево
                    G.add_edge(name, name - 1, weight=w_adj)

                # диагональные проверяем: (пв, пн, лн, лв)
                if not sampled_map[i - 1][j][2] and not sampled_map[i][j + 1][
                    2] and not sampled_map[i - 1][j + 1][2]:  # пв
                    G.add_edge(name, name - sampled_map.shape[1] + 1,
                               weight=w_diag)
                if not sampled_map[i][j + 1][2] and not sampled_map[i + 1][j][
                    2] and not sampled_map[i + 1][j + 1][2]:  # пн
                    G.add_edge(name, name + sampled_map.shape[1] + 1,
                               weight=w_diag)
                if not sampled_map[i + 1][j][2] and not sampled_map[i][j - 1][
                    2] and not sampled_map[i + 1][j - 1][2]:  # лн
                    G.add_edge(name, name + sampled_map.shape[1] - 1,
                               weight=w_diag)
                if not sampled_map[i][j - 1][2] and not sampled_map[i - 1][j][
                    2] and not sampled_map[i - 1][j - 1][2]:  # лв
                    G.add_edge(name, name - sampled_map.shape[1] - 1)
print(G.nodes())

red, blue = (255, 0, 0), (0, 0, 255)
green, pink = (51, 204, 51), (255, 0, 255)
yellow = (255, 153, 0)
index_bot = [-10, -10]
index_start = [-10, -10]
index_finish = [-10, -10]
isInProgress = False
isSetObj = [False, False]  # start/bot, finish
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_xy = pygame.mouse.get_pos()
            print(mouse_xy)
            print(index_bot)
            indexes = [(mouse_xy[0] - offset[0]) // (dl + margin),
                       (mouse_xy[1] - offset[1]) // (dl + margin)]
            if not isInProgress:
                if not sampled_map[indexes[1]][indexes[0]][2]:
                    if event.button == 1:
                        index_start = indexes
                        index_bot = indexes
                        isSetObj[0] = True
                    elif event.button == 3:
                        index_finish = indexes
                        isSetObj[1] = True
        elif event.type == pygame.KEYUP:
            if isSetObj[0] * isSetObj[1]:
                if event.key == pygame.K_d:
                    isInProgress = True
                    dijkstra()
                    isInProgress = False
                elif event.key == pygame.K_a:
                    isInProgress = True
                    a_star()
                    isInProgress = False
                elif event.key == pygame.K_r:
                    isInProgress = True
                    rrt()
                    isInProgress = False

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
            color = red if sampled_map[i][j][2] else blue
            pygame.draw.rect(screen, color, (x, y, dl, dl))

    draw_start(index_start[0], index_start[1])
    draw_finish(index_finish[0], index_finish[1])
    draw_bot(index_bot[0], index_bot[1])

    # pos = nx.spring_layout(G)  # pos = nx.nx_agraph.graphviz_layout(G)
    # nx.draw_networkx(G, pos)
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.show()

    pygame.display.update()
