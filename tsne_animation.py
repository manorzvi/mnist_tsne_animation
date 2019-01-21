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
#from tSNE_prior import *
from tSNE_prior import TSNE
from tSNE_prior import POSITIONS
from tSNE_prior import RS

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

import matplotlib.patheffects as PathEffects


class tsne_animation:

    def __init__(self, dir_results):
        self._colors = np.array(['red', 'gold', 'seagreen', 'deepskyblue',
                                 'blue', 'mediumvioletred', 'grey', 'yellow',
                                 'lime', 'purple'])
        self._labels = list(range(10))
        self._dir_results = dir_results
        self._cwd = os.getcwd()


    def _generate_tSNE(self, metric='regular', location='tSNE_results', alpha=0):

        self._activations       = []
        self._activations_names = []

        for filename in os.listdir(self._dir_results):
            if filename.endswith("activation.npy"):
                self._activations.append(np.load(self._dir_results + '\\' + filename))
                self._activations_names.append(filename)

        self._dir_tSNE_results = location
        create_dir_result(location)
        os.chdir(location)

        if metric == 'regular':
            tsne = TSNE(random_state=RS, verbose=True)
            for activation, name in zip(self._activations, self._activations_names):
                X_proj  = tsne.fit_transform(activation)
                np.save(name[:-14] + 'projection', X_proj)

        if metric == 'init':
            tsne = TSNE(random_state=RS, verbose=True)
            X_proj = tsne.fit_transform(self._activations[0])
            np.save(self._activations_names[0][:-14] + 'projection', X_proj)
            for activation, name in zip(self._activations[1:], self._activations_names[1:]):
                tsne = TSNE(random_state=RS, verbose=True, init=X_proj)
                X_proj  = tsne.fit_transform(activation)
                np.save(name[:-14] + 'projection', X_proj)

        if metric == 'prior':
            tsne = TSNE(random_state=RS, verbose=True, method='prior_tsne', alpha=alpha)
            count = 0
            for activation, name in zip(self._activations, self._activations_names):
                X_proj  = tsne.fit_transform(activation)
                np.save(name[:-14] + 'projection', X_proj)
                count += 1
                if count >= 30:
                    break

        os.chdir(self._cwd)

    def _generate_arrays(self, location='tSNE_result'):

        self._ims       = []
        self._ims_names = []

        if not hasattr(self, '_dir_tSNE_results'):
            self._dir_tSNE_results = location

        for filename in os.listdir(self._dir_tSNE_results):
            if filename.endswith("projection.npy"):
                self._ims.append(np.load(self._dir_tSNE_results + '\\' + filename))
                self._ims_names.append(filename)
        for filename in os.listdir(self._dir_results):
            if filename.endswith("tags.npy"):
                self._tags = (filename, np.load(self._dir_results + '\\' + filename))
                break


        self._ims_iter = np.dstack(im.reshape(-1,2) for im in self._ims)

        #print('ims_iter', end=' : ');
        #print(self._ims_iter.shape)
        #print('ims_iter[...,-1]', end=' : ');
        #print(self._ims_iter[..., -1].shape)

    def _scatter(self, x):

        # We create a scatter plot.
        self._f = plt.figure(figsize=(8, 8))
        self._ax = plt.subplot(aspect='equal')
        self._sc = self._ax.scatter(x[:, 0], x[:, 1], lw=0, s=10,
                                    c=self._colors[self._tags[1].astype(np.int)].tolist())

        #self._ax.set_xlim(-100, 100)
        #self._ax.set_ylim(-100, 100)
        #self._ax.axis('off')
        #self._ax.axis('tight')

        # We add the labels for each digit.
        self._txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(x[self._tags[1] == i, :], axis=0)
            txt = self._ax.text(xtext, ytext, str(i), fontsize=12)
            txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"),
                                  PathEffects.Normal()])
            self._txts.append(txt)

        return self._f, self._ax, self._sc, self._txts

    def _make_frame_mpl(self, t):

        i = int(t*10)
        print(i)
        x = self._ims_iter[..., i]
        self._sc.set_offsets(x)
        for j, txt in zip(range(10), self._txts):
            xtext, ytext = np.mean(x[self._tags[1] == j, :], axis=0)
            txt.set_x(xtext)
            txt.set_y(ytext)
        return mplfig_to_npimage(self._f)

    def anim(self, gif='mnist_tsne_activations.gif'):

        self._scatter(self._ims_iter[..., -1])

        animation = mpy.VideoClip(self._make_frame_mpl,
                                  duration=self._ims_iter.shape[2]/10)

        animation.write_gif(os.path.join(os.getcwd(), gif), fps=10)


if __name__ == '__main__':

    dir_results = 'mnist_activations_results'

    a = tsne_animation(dir_results)
    a._generate_tSNE(metric='prior', location='prior_alpha09_tSNE_result', alpha=0.9)
    a._generate_arrays(location='prior_alpha02_tSNE_result')
    a.anim(gif='mnist_activations_prior_alpha02_tSNE.gif')

