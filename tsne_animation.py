import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import sys
import shutil

from moviepy.editor             import *
from tSNE_prior                 import TSNE
from tSNE_prior                 import RS
from moviepy.video.io.bindings  import mplfig_to_npimage
from pprint                     import pprint
import moviepy.editor           as mpy
import matplotlib.patheffects   as PathEffects

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def create_dir_result(directory):
    current_directory = os.getcwd()
    print(current_directory)
    final_directory   = os.path.join(current_directory, directory)
    print(final_directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        print(directory + ' created')
    else: #directory exist
        shutil.rmtree(final_directory)
        print(directory + ' removed')
        os.makedirs(final_directory)
        print(directory + ' created again')


class tsne_animation:

    def __init__(self, dir_results):
        self._colors = np.array(['red', 'gold', 'seagreen', 'deepskyblue',
                                 'blue', 'mediumvioletred', 'grey', 'yellow',
                                 'lime', 'purple'])
        self._labels = list(range(10))
        self._dir_results = dir_results
        self._cwd = os.getcwd()


        os.chdir(self._dir_results)
        for filename in os.listdir():
            if filename.endswith("tags.npy"):
                self._tags = (filename, np.load(filename))
                break
        os.chdir(self._cwd)



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
                X_proj  = tsne.fit_transform(activation, self._intermediate_positions)
                np.save(name[:-14] + 'projection', X_proj)

        if metric == 'init':
            tsne = TSNE(random_state=RS, verbose=True)
            X_proj = tsne.fit_transform(self._activations[0], self._intermediate_positions)
            np.save(self._activations_names[0][:-14] + 'projection', X_proj)
            for activation, name in zip(self._activations[1:], self._activations_names[1:]):
                tsne = TSNE(random_state=RS, verbose=True, init=X_proj)
                X_proj  = tsne.fit_transform(activation, self._intermediate_positions)
                np.save(name[:-14] + 'projection', X_proj)

        if metric == 'prior':
            tsne = TSNE(random_state=RS, verbose=True, method='prior_tsne', alpha=alpha)
            count = 0 #Prior run takes long time. We allow it to run only for 30 activations.
            for activation, name in zip(self._activations, self._activations_names):
                X_proj  = tsne.fit_transform(activation, self._intermediate_positions)
                np.save(name[:-14] + 'projection', X_proj)
                count += 1
                if count >= 30:
                    break
        print('[tSNE Animation] generate_tSNE Done.')
        os.chdir(self._cwd)

    def _generate_intermediate_positions(self, metric=[], origin=[], alpha=None):
        """"
        Run tSNE algorithm and record intermediate points positions meanwhile.
        Save those positions into attribute: self._intermediate_positions (dict of list).
        Do it for each metric in <metric> and for each origin file in <origin>.
        :parameter:
        :metric:   [str] (list of strings), type of tSNE algorithem. options: regular, init, prior
        :origin:   [str] (list of strings), relative paths to origin <activations>.npy files
        :return:
        None
        """
        if not isinstance(origin, list) or not isinstance(metric, list):
            raise ValueError('Do me a favor, give me metric and origin as lists (even of 1 element long)')

        for o in origin:
            if not o.endswith('.npy'):
                raise ValueError('All origin files must be Numpy Pickel file (*.npy)')

        if ('init' in metric or 'prior' in metric) and len(origin)<2:
            raise ValueError('If metric chosen is \'init\' or \'prior\', one must provide at least 2 activation files')

        if 'prior' in metric and alpha is None:
            raise ValueError('If metric chosen is \'prior\', a valid alpha value must be provided')

        cwd = os.getcwd()
        os.chdir(self._dir_results)

        # A dictionary to hold tSNE intermediate results per each metric
        self._intermediate_positions = {}
        # Hold the origins for naming conventions purposes.
        self._origin                 = origin

        if 'regular' in metric:
            print('[tSNE Animation] Start regular tSNE calculation')
            # that list is just temporary
            self._intermediate_positions['regular'] = []
            # One tSNE object to rule them all
            tsne = TSNE(random_state=RS, verbose=True)
            # Per each activation file
            for activationFile in origin:
                # load one file
                activation = np.load(activationFile)
                # tSNE the hell out of it
                tsne.fit_transform(activation, self._intermediate_positions['regular'])

            # Convert each list of intermediate positions into iterator for animation purposes
            self._intermediate_positions['regular'] = np.dstack(pos.reshape(-1, 2)
                                                                for pos
                                                                in self._intermediate_positions['regular'])
        if 'init' in metric:
            print('[tSNE Animation] Start init tSNE calculation')
            self._intermediate_positions['init'] = []
            tsne = TSNE(random_state=RS, verbose=True)
            activation = np.load(origin[0])
            X_proj = tsne.fit_transform(activation, self._intermediate_positions['init'])
            for activationFile in origin[1:]:
                activation = np.load(activationFile)
                tsne = TSNE(random_state=RS, verbose=True, init=X_proj)
                X_proj = tsne.fit_transform(activation, self._intermediate_positions['init'])

            self._intermediate_positions['init'] = np.dstack(pos.reshape(-1, 2)
                                                             for pos
                                                             in self._intermediate_positions['init'])
        if 'prior' in metric:
            print('[tSNE Animation] Start prior tSNE calculation')
            self._intermediate_positions['prior'] = []
            tsne = TSNE(random_state=RS, verbose=True, method='prior_tsne', alpha=alpha)
            for activationFile in origin:
                activation = np.load(activationFile)
                tsne.fit_transform(activation, self._intermediate_positions['prior'])

            self._intermediate_positions['prior'] = np.dstack(pos.reshape(-1, 2)
                                                              for pos
                                                              in self._intermediate_positions['prior'])

        os.chdir(cwd)
        np.save('intermediate_positions.npy', self._intermediate_positions)
        np.save('intermediate_positions_origin.npy', self._origin)
        print('[tSNE Animation] generate_intermediate_positions Done')



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
        self._ims_iter = np.dstack(im.reshape(-1, 2) for im in self._ims)

    def _scatter(self, x):
        """
        Scatter plot x
        :param x: 2D ndarray of data-points
        :return:  None
        :Note:    Method also make use of self._colors, self._tags which initialized at: self._generate arrays and constructor
        """

        # We create a scatter plot.
        self._f = plt.figure(figsize=(8, 8))
        self._ax = plt.subplot(aspect='equal')
        self._sc = self._ax.scatter(x[:, 0], x[:, 1], lw=0, s=10,
                                    c=self._colors[self._tags[1].astype(np.int)].tolist())

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
        """
        Method goes over self._ims_iter (an iterator of final tSNE results), and set offsets of scatter plot
        which handled by attribute self._sc.
        :param t: time parameter, necessary to MoviePy package
        :return: A MoviePy object
        """

        i = int(t*10)
        print(i)
        x = self._ims_iter[..., i]
        self._sc.set_offsets(x)
        for j, txt in zip(range(10), self._txts):
            xtext, ytext = np.mean(x[self._tags[1] == j, :], axis=0)
            txt.set_x(xtext)
            txt.set_y(ytext)
        return mplfig_to_npimage(self._f)

    def _make_frame_mpl2(self, t):
        """
        Method goes over self._results (an iterator of intermediate tSNE positions), and set offsets of scatter plot
        which handled by attribute self._sc.
        Note: same as _make_frame_mpl, but refers to self._results, instead of self._ims_iter
        :param t: time parameter, necessary to MoviePy package
        :return: A MoviePy object
        """

        i = int(t*10)
        print(i)
        x = self._results[..., i]
        self._sc.set_offsets(x)
        for j, txt in zip(range(10), self._txts):
            xtext, ytext = np.mean(x[self._tags[1] == j, :], axis=0)
            txt.set_x(xtext)
            txt.set_y(ytext)
        return mplfig_to_npimage(self._f)

    def anim(self, gif='mnist_tsne_activations.gif'):
        """
        Animate final tSNE results.
        :param gif: name.
        :return: None
        """

        self._scatter(self._ims_iter[..., -1])

        animation = mpy.VideoClip(self._make_frame_mpl,
                                  duration=self._ims_iter.shape[2]/10)

        animation.write_gif(os.path.join(os.getcwd(), gif), fps=10, loop=0)

    def _animate_intemediate_positions(self):
        """
        This method animate each intermediate series of tSNE results (as recorded in _generate_intermediate_positions),
        and save it as *.gif file.
        :prerequisites:
        :Run _generate_intermediate_positions before running this method.
        :parameter:
        :None
        :return:
        :None
        """
        if not hasattr(self,'_intermediate_positions'):
            self._intermediate_positions = np.load('intermediate_positions.npy').item()
        if not hasattr(self, '_origin'):
            self._origin = np.load('intermediate_positions_origin.npy').item()

        for metric, results in self._intermediate_positions.items():

            self._results = results

            self._scatter(self._results[..., -1])

            animation = mpy.VideoClip(self._make_frame_mpl2,
                                      duration=self._results.shape[2] / 10)
            animation.write_gif(os.path.join(os.getcwd(), '{}-{}_{}.gif'.format(self._origin[0][:-20],
                                                                                self._origin[-1][13:-4],
                                                                                metric)),
                                fps=10,
                                loop=0)
            del self._results


if __name__ == '__main__':

    dir_results = 'mnist_activations_results'

    a = tsne_animation(dir_results)

    a._generate_intermediate_positions(metric=['regular', 'init', 'prior'],
                                       origin=['Epoch000Batch099_TSNE_activation.npy',
                                               'Epoch000Batch100_TSNE_activation.npy'], alpha=0.9)

    a._animate_intemediate_positions()

    #a._generate_tSNE(metric='regular', location='regular_tSNE_result')
    #a._generate_arrays(location='regular_tSNE_result')
    #a.anim(gif='TSNE_activations_regular.gif')

