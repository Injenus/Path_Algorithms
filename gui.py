import numpy as np
import pygame
import sys

colmns, rows = 113, 89
blank_map = np.load('blank_map_0(test).npy')
koef = 7
margin = 1
size = (colmns * koef, rows * koef)
pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Map')

free = blank_map[np.where(blank_map[:, 2].astype(int) == 0)]
barrier = blank_map[np.where(blank_map[:, 2].astype(int) == 1)]
offset_xy = (np.amin(blank_map, axis=0)[0], np.amax(blank_map, axis=0)[1])

red = (255, 0, 0)
blue = (0, 0, 255)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
    for ind_point in range(len(blank_map)):
        if blank_map[ind_point][2]:
            pygame.draw.rect(screen, red, (
            blank_map[ind_point][0] - 0.5 * 2 ** 0.5 * koef,
            blank_map[ind_point][1] + 0.5 * 2 ** 0.5 * koef, koef, koef))
        else:
            pass
    # for col in range(colmns):
    #     x = col * koef
    #     pygame.draw.rect(screen, red, (x, 0, koef, koef))

    pygame.display.update()
