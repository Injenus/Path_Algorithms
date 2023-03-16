import numpy as np
import matplotlib.pyplot as plt

sampled_map = np.load('Sampled_map_as_center_0125.npy')
for i in range(len(sampled_map)):
    free = sampled_map[np.where(sampled_map[:, :, 2].astype(int) == 0)]
    barrier = sampled_map[np.where(sampled_map[:, :, 2].astype(int) == 1)]
    plt.scatter(free[:, 0], free[:, 1], color='b')
    plt.scatter(barrier[:, 0], barrier[:, 1], color='r')
plt.show()
