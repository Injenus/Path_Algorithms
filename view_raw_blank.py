import numpy as np
import matplotlib.pyplot as plt
blank_map = np.load('blank_map_0(test).npy')
free = blank_map[np.where(blank_map[:, 2].astype(int) == 0)]
barrier = blank_map[np.where(blank_map[:, 2].astype(int) == 1)]
plt.scatter(free[:,0],free[:,1], color='b')
plt.scatter(barrier[:,0],barrier[:,1], color='r')
plt.show()