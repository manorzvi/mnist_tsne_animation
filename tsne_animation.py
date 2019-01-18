import numpy as np
import matplotlib.pyplot as plt
from tsne_animation_utils import create_dir_result
import seaborn as sns
import sys
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from moviepy.editor import *

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

import matplotlib.patheffects as PathEffects


class tsne_animation:

    def __init__(self, dir_results, from_pickel=True, arrow_anim=False):
        self._colors = np.array(['red', 'gold', 'seagreen', 'deepskyblue',
                                 'blue', 'mediumvioletred', 'grey', 'yellow',
                                 'lime', 'purple'])
        self._labels = list(range(10))
        self._dir_results = dir_results
        self._arrow_anim = arrow_anim

        self._ploted = False
        self._generate_arrays()

    def _generate_arrays(self):

        self._ims  = []
        self._tags = []
        self._ims_srcs = []
        self._tags_srcs = []

        for filename in os.listdir(self._dir_results):
            if filename.endswith("proj.npy"):
                self._ims.append(np.load(self._dir_results + '\\' + filename))
                self._ims_srcs.append(filename)
            elif filename.endswith("tags.npy"):
                self._tags.append(np.load(self._dir_results + '\\' + filename))
                self._tags_srcs.append(filename)

        self._ims_iter = np.dstack(im for im in self._ims)

        #print('ims_iter', end=' : ');
        #print(self._ims_iter.shape)
        #print('ims_iter[...,-1]', end=' : ');
        #print(self._ims_iter[..., -1].shape)

    def _scatter(self, x, index):

        # We create a scatter plot.
        self._f = plt.figure(figsize=(8, 8))
        self._ax = plt.subplot(aspect='equal')
        self._sc = self._ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                                    c=self._colors[self._tags[-1].astype(np.int)].tolist())

        self._ax.set_xlim(-25, 25)
        self._ax.set_ylim(-25, 25)
        self._ax.axis('off')
        self._ax.axis('tight')

        # We add the labels for each digit.
        self._txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.mean(x[self._tags[-1] == i, :], axis=0)
            txt = self._ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            self._txts.append(txt)

        return self._f, self._ax, self._sc, self._txts

    def plotting(self):

        self._dir_scatter_results = 'scatter_plots_results'

        create_dir_result(self._dir_scatter_results)

        self._ims_arrow_corners = []

        for x, name in zip(self._ims_iter, self._ims_srcs):
            self._scatter()
            plt.savefig(self._dir_scatter_results + '\\' + name[:-9], dpi=120)

    def _make_frame_mpl(self, t):

        i = int(t*10)
        print(t)
        print(i)
        input()
        x = self._ims_iter[..., i]
        self._sc.set_offsets(x)
        for j, txt in zip(range(10), self._txts):
            xtext, ytext = np.mean(x[self._tags[-1] == i, :], axis=0)
            txt.set_x(xtext)
            txt.set_y(ytext)
        return mplfig_to_npimage(self._f)

    def anim(self):

        self._scatter(self._ims_iter[..., -1])

        animation = mpy.VideoClip(self._make_frame_mpl,
                                  duration=self._ims_iter.shape[2]/10)

        animation.write_gif(os.path.join(os.getcwd(), 'mnist_tsne_activations.gif'), fps=10)




if __name__ == '__main__':

    dir_results = 'tsne_results'

    a = tsne_animation(dir_results, from_pickel=True, arrow_anim=False)

    a.plotting()
    a.anim()
