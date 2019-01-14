import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import sys
import os
import shutil
import seaborn as sns
from mnistModel import mnistModel
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


class MnistTsne:

    def __init__(self,
                 dir_results      = 'tsne_results',
                 train_csv        = r'C:\Users\Manor\Desktop\projectB\mnist_train.csv',
                 test_csv         = r'C:\Users\Manor\Desktop\projectB\mnist_test.csv',
                 train_batch_size = 32,
                 num_of_epoches   = 1,
                 optimizer        = 'GradientDescentOptimizer'):

        self._model = mnistModel()
        self._dir_results      = dir_results
        self._current_dir      = os.getcwd()
        self._train_csv        = train_csv
        self._test_csv         = test_csv
        self._train_batch_size = train_batch_size
        self._num_of_epoches   = num_of_epoches
        if optimizer == 'GradientDescentOptimizer':
            self._optimizer    = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self._train_loss_results     = []
        self._train_accuracy_results = []
        self._activations            = []

        self._RS = 11092001

        self.generate_dataset()

    def _loss_sparse_softmax_cross_entropy(self, x, y):

        self._last_prediction = self._model(x)
        self._last_loss       = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                                       logits=self._last_prediction)
        # Track progress
        self._train_loss_results.append(self._last_loss.numpy())

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            self._loss_sparse_softmax_cross_entropy(inputs, targets)

        self._last_gradient = tape.gradient(self._last_loss, self._model.trainable_variables)

    @staticmethod
    def _batch(trainset, labels, start, end):

        yield trainset[start:end, :], labels[start:end]

    def mini_batch_train(self):

        # Perform multiple epoches on the whole data set
        for epoch in range(self._num_of_epoches):

            epoch_accuracy = []
            epoch_loss     = []

            # Perform Stratified K fold cross-validation on the data set for each epoch.
            for train_index, validation_index in StratifiedShuffleSplit(n_splits=10,
                                                                        test_size=0.1).split(self._train_features,
                                                                                             self._train_labels):
                _validation_features = self._train_features[validation_index]
                _validation_labels = self._train_labels[validation_index]

                _train_features = self._train_features[train_index]
                _train_labels   = self._train_labels[train_index]

                # create permutation of the remain train-set
                permutation = np.random.permutation(len(_train_labels))
                _train_labels = _train_labels[permutation]
                _train_features = _train_features[permutation]

                # get the number of batches to go through
                num_batches = int(_train_labels.shape[0]/self._train_batch_size)+1

                batch_counter = 0
                # generate batches of the train-set
                for batch_data, batch_label in zip(np.array_split(_train_features, num_batches),
                                                   np.array_split(_train_labels, num_batches)):

                    # Optimize the model
                    self.grad(batch_data, batch_label)
                    self._optimizer.apply_gradients(zip(self._last_gradient, self._model.variables))

                epoch_accuracy.append(self.compute_accuracy(_validation_features,
                                                            _validation_labels))
                epoch_loss.append(self._loss_sparse_softmax_cross_entropy(_validation_features,
                                                                          _validation_labels))

                ####################################################################
                ##################### MANOR 14/1/19 : NOT DONE!!! ##################
                ####################################################################

    def tsne_compare(self):
        print("Let's check if TSNE Algo produce diffrent output for 2 same inputs (with the same RS)")
        lla_tsne_comparisons = TSNE(n_components=2, random_state=self._RS)

        lla_proj = [lla_tsne_comparisons.fit_transform(self._activations[-1].numpy()),
                    lla_tsne_comparisons.fit_transform(self._activations[-1].numpy())]

        palette = np.array(sns.color_palette("hls", 10))
        fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')

        i = 0
        for a in lla_proj:
            ax[i].scatter(a[:, 0], a[:, 1], lw=0, s=10, c=palette[self._ytsne.astype(np.int)])
            i += 1

        plt.savefig('TwoTSNE_same')

    def lla_tsne_projection(self, epoch, batch):

        lla = self._model(self._xtsne, a2out=True)
        self._activations.append(lla)

        lla_tsne_obj = TSNE(n_components=2, random_state=self._RS)
        lla_tsne_proj = lla_tsne_obj.fit_transform(lla.numpy())

        np.save(self._dir_results + "\Epoch{:03d}Batch{:03d}TSNE_proj".format(epoch, batch),
                lla_tsne_proj)
        if not os.path.isfile(self._dir_results + "\TSNE_tags".format(epoch, batch)):
            np.save(self._dir_results + "\TSNE_tags".format(epoch, batch),
                    self._ytsne)

    def generate_dataset(self):

        _train_df = pd.read_csv(self._train_csv)
        _test_df = pd.read_csv(self._test_csv)

        self._train = _train_df.values
        self._test  = _test_df.values

        self._train_features = self._train[:, :-1]/255.0
        self._train_labels   = self._train[:, -1]

        self._test_features  = self._test[:, :-1]/255.0
        self._test_labels    = self._test[:, -1]

        self._tsne_features  = self._test_features[:1000]
        self._tsne_labels    = self._test_labels[:1000]

    def describe_data_set(self):
        self._train_df = pd.read_csv(self._train_csv)
        self._test_df = pd.read_csv(self._test_csv)
        print('-----------------------------\n'
              '-Train data set Labels Count-\n'
              '-----------------------------')
        print(self._train_df[self._label_name].value_counts())
        print('----------------------------\n'
              '-Test data set Labels Count-\n'
              '----------------------------')
        print(self._test_df[self._label_name].value_counts())

        plt.figure()
        self._train_df[self._label_name].hist()
        self._test_df[self._label_name].hist()
        plt.show()

    def compute_accuracy(self, labels):
        predictions = tf.argmax(self._last_prediction, axis=1, output_type=tf.int64)
        labels = tf.cast(labels, tf.int64)
        batch_size = int(self._last_prediction.shape[0])

        self._train_accuracy_results.append((tf.reduce_sum(tf.cast(tf.equal(predictions,
                                                                            labels),
                                                                   dtype=tf.float32)) / batch_size).numpy())

    def _create_dir_result(self):
        self._final_dir_result = os.path.join(self._current_dir, self._dir_results)
        if not os.path.exists(self._final_dir_result):
            os.makedirs(self._final_dir_result)
            print(self._final_dir_result + ' created')
        else:  # directory exist
            shutil.rmtree(self._final_dir_result)
            print(self._final_dir_result + ' removed')
            os.makedirs(self._final_dir_result)
            print(self._final_dir_result + ' created again')

    @staticmethod
    def _pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    def plot_progress(self, metrics='loss'):

        plt.figure(figsize=[16, 16])

        if metrics == 'loss':
            plt.plot(range(len(self._train_loss_results)),
                     self._train_loss_results,
                     label='train loss')
            plt.plot(list(range(len(self._train_loss_results)))[::500],
                     self._train_loss_results[::500], '.')

            for i, j in zip(list(range(len(self._train_loss_results)))[::500],
                            self._train_loss_results[::500]):
                plt.annotate(str("{:.03f}".format(j)), xy=(i, j))

            plt.plot(len(self._train_loss_results) - 1, self._train_loss_results[-1], 'ro')
            plt.annotate(str("{:.03f}".format(self._train_loss_results[-1])),
                         xy=(len(self._train_loss_results) - 1,
                             self._train_loss_results[-1]))
            plt.title('Train loss as function of iterations')
            plt.ylabel('Loss')
        if metrics == 'accuracy':
            plt.plot(range(len(self._train_accuracy_results)),
                     self._train_accuracy_results,
                     label='train accuracy')
            plt.plot(list(range(len(self._train_accuracy_results)))[::100],
                     self._train_accuracy_results[::100], '.')

            for i, j in zip(list(range(len(self._train_accuracy_results)))[::100],
                            self._train_accuracy_results[::100]):
                plt.annotate(str("{:.03f}".format(j)), xy=(i, j))

            plt.plot(len(self._train_accuracy_results) - 1, self._train_accuracy_results[-1], 'ro')
            plt.annotate(str("{:.03f}".format(self._train_accuracy_results[-1])),
                         xy=(len(self._train_accuracy_results) - 1,
                             self._train_accuracy_results[-1]))
            plt.title('Train accuracy as function of iterations')
            plt.ylabel('Accuracy')

        plt.legend()
        if metrics == 'loss':
            plt.savefig('mnist_train_loss_iteration_graph')
        if metrics == 'accuracy':
            plt.savefig('mnist_train_accuracy_iteration_graph')
        plt.show()


def main():

    mnist_tsne_obj = MnistTsne()

    #mnist_tsne_obj.describe_data_set()

    mnist_tsne_obj.mini_batch_train()
    sys.exit()
    input('Enter for tsne_compare')
    mnist_tsne_obj.tsne_compare()

    mnist_tsne_obj.plot_progress()


if __name__ == '__main__':
    sys.exit(main())
