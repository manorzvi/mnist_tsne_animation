import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tsne_animation_utils import create_dir_result
from pprint import pprint
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from moviepy.editor import *

import matplotlib.patheffects as PathEffects


class tsne_animation:

    def __init__(self, dir_results, from_pickel=True):
        self._colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'black', 'orange', 'purple'
        self._labels = list(range(10))
        self._dir_results = dir_results

    def _generate_arrays(self):
        self._ims  = []
        self._tags = []
        self._ims_srcs = []
        self._tags_srcs = []
        for filename in os.listdir(self._dir_results):
            if filename.endswith("proj.npy"):
                self._ims.append(np.load(self._dir_results + '\\' + filename))
                self._ims_srcs.append(filename)
            elif filename.endswith("TAGS.npy"):
                self._tags.append(np.load(self._dir_results + '\\' + filename))
                self._tags_srcs.append(filename)

        #print(len(self._ims))
        #print(self._ims[-1].shape)
        #print(len(self._tags))
        #print(self._tags[-1].shape)

    def _scatter(self, index):

        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 10))
        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(self._ims[index][:, 0], self._ims[index][:, 1], lw=0, s=40,
                        c=palette[self._tags[index].astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(self._ims[index][self._tags[index] == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        return f, ax, sc, txts

    def plotting(self):

        self._generate_arrays()

        self._dir_scatter_results = 'scatter_plots_results'
        create_dir_result(self._dir_scatter_results)

        for name,i in zip(self._ims_srcs,range(len(self._ims))):

            self._scatter(i)
            plt.savefig(self._dir_scatter_results + '\\' + name[:-9], dpi=120)

    def get_list_scatter_plots(self):
        self._scatter_plots_names = []
        for filename in os.listdir(self._dir_scatter_results):
            if filename.endswith("TSNE.png"):
                self._scatter_plots_names.append(self._dir_scatter_results + '\\' + filename)

        return self._scatter_plots_names

    def anim(self):

        self._generate_arrays()
        self._ims_iteration = np.dstack(im.reshape(-1, 2) for im in self._ims)
        self._tags_iteration = np.dstack(tag.reshape(-1, 2) for tag in self._tags)










        #fig = plt.figure()

        #for q in range(len(self._ims)):
        #    pass
            # im = plt.scatter(self._ims[i][:,0], self._ims[i][:,1], animated=True)
            # self._frames.append([im])

            #taggedX0 = self._ims[q][self._tags[q] == 0, 0]
            #taggedX1 = self._ims[q][self._tags[q] == 0, 1]

            #im = plt.scatter(taggedX0,
            #                 taggedX1,
            #                 c=self._colors[0],
            #                 label=0,
            #                 animated=True)

            #for c, label in zip(self._colors[1:], np.arange(1, 10)):
            #    taggedX0 = self._ims[q][self._tags[q] == label, 0]
            #    taggedX1 = self._ims[q][self._tags[q] == label, 1]
            #    im=plt.scatter(taggedX0,
            #                taggedX1,
            #                c=c,
            #                label=label,
            #                animated=True)

            #self._frames.append([im])




        #ani = animation.ArtistAnimation(fig, self._frames, interval=300, blit=True,
        #                                repeat=True)
        #plt.show(ani)



if __name__ == '__main__':

    dir_results = 'tsne_results'

    a = tsne_animation(dir_results, from_pickel=True)
    a.plotting()
    #a.anim()
    #f, ax, sc, txts = a._scatter(-1)

    #animation = mpy.VideoClip(a._make_frame_mpl,
    #                          duration=a._ims_iteration.shape[2] / 40.)

    #animation.write_gif("tsne_animation.gif", fps=20)

    scatter_plots_list = a.get_list_scatter_plots()
    print(scatter_plots_list)

    clip = ImageSequenceClip(scatter_plots_list, fps=8)
    clip.write_gif('tsne_gif.gif', fps=8, loop=1)






