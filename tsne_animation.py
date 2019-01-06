import numpy as np
import matplotlib.pyplot as plt
from tsne_animation_utils import create_dir_result
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from moviepy.editor import *

import matplotlib.patheffects as PathEffects


class tsne_animation:

    def __init__(self, dir_results, from_pickel=True, arrow_anim=True):
        self._colors = np.array(['red', 'gold', 'seagreen', 'deepskyblue',
                                 'blue', 'mediumvioletred', 'grey', 'yellow',
                                 'lime', 'purple'])
        self._labels = list(range(10))
        self._dir_results = dir_results
        self._arrow_anim = arrow_anim

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

        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(self._ims[index][:, 0], self._ims[index][:, 1], lw=0, s=10,
                        c=self._colors[self._tags[index].astype(np.int)].tolist())

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        arrow_corners = np.zeros(shape=(2, 10))

        for i in range(10):
            xtext, ytext = np.mean(self._ims[index][self._tags[index] == i, :], axis=0)
            arrow_corners[:, i] = np.array([xtext, ytext])
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([PathEffects.Stroke(linewidth=5,
                                                     foreground="w"),
                                  PathEffects.Normal()])

        if self._arrow_anim:
            self._ims_arrow_corners.append(arrow_corners)
            if index > 0:
                prev_arrow_corners = self._ims_arrow_corners[0]
                for arrow_corners in self._ims_arrow_corners[1:]:
                    deltas = arrow_corners - prev_arrow_corners
                    for i in range(10):
                        deltaX = deltas[0, i]
                        deltaY = deltas[1, i]
                        arrow = ax.arrow(prev_arrow_corners[0, i], prev_arrow_corners[1, i], deltaX, deltaY, fc=self._colors[i], ec=self._colors[i])

                    prev_arrow_corners = arrow_corners

    def _digit_scatter(self, index, digit):

        # We create a scatter plot.
        f = plt.figure(figsize=(20, 20))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(self._ims[index][np.where(self._tags[index] == digit), 0],
                        self._ims[index][np.where(self._tags[index] == digit), 1], lw=0, s=10,
                        c=self._colors[self._tags[index].astype(np.int)].tolist()[digit])

        ax.set_xlim(-100,100)
        ax.set_ylim(-100, 100)

        xtext, ytext = np.mean(self._ims[index][self._tags[index] == digit, :], axis=0)
        arrow_corners = np.array([xtext, ytext])

        txt = ax.text(xtext, ytext, str(digit), fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

        loctxt = ax.text(-100, -100, str((xtext, ytext)), fontsize=24)
        loctxt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

        if self._arrow_anim:
            self._ims_arrow_corners.append(arrow_corners)
            if index > 0:
                prev_arrow_corners = self._ims_arrow_corners[0]

                for arrow_corners in self._ims_arrow_corners[1:]:
                    deltas = arrow_corners - prev_arrow_corners

                    deltaX = deltas[0]
                    deltaY = deltas[1]
                    arrow = ax.arrow(prev_arrow_corners[0], prev_arrow_corners[1], deltaX, deltaY, fc=self._colors[digit], ec=self._colors[digit])

                    prev_arrow_corners = arrow_corners

    def plotting(self):
        self._plotting = True

        self._generate_arrays()

        self._dir_scatter_results = 'scatter_plots_results'
        create_dir_result(self._dir_scatter_results)

        self._ims_arrow_corners = []

        for name, index in zip(self._ims_srcs, range(len(self._ims))):
            self._scatter(index)

            plt.savefig(self._dir_scatter_results + '\\' + name[:-9], dpi=120)

    def plotting_one_digit(self, digit=0):
        self._plotting_one_digit = True
        self._generate_arrays()

        self._dir_scatter_results = 'digit_{}_scatter_plots_results'.format(digit)
        create_dir_result(self._dir_scatter_results)

        self._ims_arrow_corners = []

        for name, index in zip(self._ims_srcs, range(len(self._ims))):
            self._digit_scatter(index, digit)

            plt.savefig(self._dir_scatter_results + '\\' + name[:-9], dpi=120)

    def get_list_scatter_plots(self):
        self._scatter_plots_names = []
        for filename in os.listdir(self._dir_scatter_results):
            if filename.endswith("TSNE.png"):
                self._scatter_plots_names.append(self._dir_scatter_results + '\\' + filename)

        return self._scatter_plots_names

    def anim_one_digit(self, digit=0):

        self.plotting_one_digit(digit)
        scatter_plots_list = a.get_list_scatter_plots()
        clip = ImageSequenceClip(scatter_plots_list, fps=5)
        clip.write_gif('digit_{}_tsne_gif.gif'.format(digit), fps=5, loop=1)

    def anim(self):

        self.plotting()
        scatter_plots_list = a.get_list_scatter_plots()
        clip = ImageSequenceClip(scatter_plots_list, fps=5)
        clip.write_gif('tsne_gif.gif', fps=5, loop=1)




if __name__ == '__main__':

    dir_results = 'tsne_results'

    a = tsne_animation(dir_results, from_pickel=True, arrow_anim=False)
    #a.plotting()
    #a.plotting_one_digit(digit=0)

    a.anim_one_digit(digit=0)
    a.anim()
