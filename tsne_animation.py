import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class tsne_animation:

    def __init__(self, ims, lbls):
        self._ims = ims
        self._tags = lbls
        self._frames = []
        self._colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

    def anim(self):

        fig = plt.figure()

        for q in range(len(self._ims)):

            # im = plt.scatter(self._ims[i][:,0], self._ims[i][:,1], animated=True)
            # self._frames.append([im])
            im = plt.scatter(self._ims[q][self._tags[q] == 0, 0],
                             self._ims[q][self._tags[q] == 0, 1],
                             c=self._colors[0],
                             label=0,
                             animated=True)

            for c, label in zip(self._colors[1:], np.arange(1, 10)):
                im = plt.scatter(self._ims[q][self._tags[q] == label, 0],
                            self._ims[q][self._tags[q] == label, 1],
                            c=c,
                        label=label,
                        animated=True)

            self._frames.append([im])


        ani = animation.ArtistAnimation(fig, self._frames, interval=300, blit=True,
                                        repeat=True)

        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        plt.show()



if __name__ == '__main__':

    ims = []
    lbls = []
    for i in range(10):
        t_arr = np.random.rand(50, 2)
        lables = np.random.randint(0, 10, size=50)
        ims.append(t_arr)
        lbls.append(lables)

    a = tsne_animation(ims, lbls)
    a.anim()
