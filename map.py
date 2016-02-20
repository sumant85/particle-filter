__author__ = 'sumant'

import math

import numpy as np


class Map(object):
    def __init__(self, x_size, y_size):
        # Prob map stores the probability that a particular map cell is unoccupied
        # Thus, prob 1 => Unoccupied; prob 0 => occupied/unknown
        self.prob_map = None
        self.x_size = x_size
        self.y_size = y_size
        pass

    @classmethod
    def from_file(cls, filename, x_size, y_size):
        # TODO: hard coding constants for now
        mp = cls(x_size, y_size)
        prob_map = [[] for _ in range(x_size)]
        x = 0
        with open(filename, 'r') as fp:
            cnt = 0
            for line in fp:
                if cnt < 7:
                    cnt += 1
                    continue
                prob_map[x] = [0 for _y in range(y_size)]

                for i, ele in enumerate(line.strip().split(' ')):
                    ele = float(ele)
                    if ele == -1:
                        prob_map[x][i] = 0
                    else:
                        prob_map[x][i] = ele
                x += 1
        mp.prob_map = np.array(prob_map)
        return mp


def visualize_map(m):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca()
    plt.imshow(m.prob_map.T)
    ax.set_xticks(range(0, 800, 20))
    ax.set_yticks(range(0, 800, 20))

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    m = Map.from_file('map/wean.dat', 800, 800)
    visualize_map(m)
    # m = Map.from_file('map/test.dat', 5, 5)