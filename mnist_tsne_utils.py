import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_) , y_


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value, y_ = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables), y_


def feed4tsne(model, x):

    return model(x, a2out=True)


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


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.cast(labels, tf.int64)
  batch_size = int(logits.shape[0])
  return tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def plot_progress(list_results, metrics='loss'):

    fig = plt.figure(figsize=[16,16])
    if metrics == 'loss':     plt.plot(range(len(list_results)), list_results, label='train loss')
    if metrics == 'accuracy': plt.plot(range(len(list_results)), list_results, label='train accuracy')
    plt.plot(list(range(len(list_results)))[::500], list_results[::500], '.')
    for i, j in zip(list(range(len(list_results)))[::500], list_results[::500]):
        plt.annotate(str("{:.03f}".format(j)), xy=(i, j))
    plt.plot(len(list_results) - 1, list_results[-1], 'ro')
    plt.annotate(str("{:.03f}".format(list_results[-1])), xy=(len(list_results) - 1, list_results[-1]))

    if metrics == 'loss':     plt.title('Train loss as function of iterations')
    if metrics == 'accuracy': plt.title('Train accuracy as function of iterations')
    plt.xlabel('Iterations')
    if metrics == 'loss':   plt.ylabel('Loss')
    if metrics == 'accuracy':   plt.ylabel('Accuracy')
    plt.legend()
    if metrics == 'loss':     plt.savefig('mnist_train_loss_iteration_graph')
    if metrics == 'accuracy': plt.savefig('mnist_train_accuracy_iteration_graph')
    plt.show()


def mini_batch_train(num_epochs, model, optimizer, global_step, mnist_train, xtest, ytest, xtsne, ytsne, dir_results):

    train_loss_results = []
    train_accuracy_results = []
    activations = []

    for epoch in range(num_epochs):

        batch = 0
        for x, y in mnist_train:

            x = tf.to_float(x) / 255.0

            # Optimize the model
            loss_value, grads, predictions = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step)

            # Track progress
            train_loss_results.append(loss_value.numpy())
            train_accuracy_results.append(compute_accuracy(predictions, y).numpy())

            if batch % 100 == 0:

                print("Epoch {:03d}, Batch {:03d}:      Loss: {:.3f}, Accuracy: {:.3f}".format(epoch,
                                                                                               batch,
                                                                                               train_loss_results[-1],
                                                                                               train_accuracy_results[-1]))

            # For mnistCNNmodel:
            # ------------------
            #print("Epoch {:03d}, Batch {:03d}:      Loss: {:.3f}, Accuracy: {:.3f}".format(epoch,
            #                                                                               batch,
            #                                                                               train_loss_results[-1],
            #                                                                               train_accuracy_results[-1]))


            if epoch == num_epochs-1 and batch < 59900 and batch > 59700:
                #if batch % 10 == 0:
                    a2out = feed4tsne(model, xtsne)
                    activations.append(a2out)
                    a2out_tsne = TSNE(n_components=2, random_state=20181712)
                    a2out_proj = a2out_tsne.fit_transform(a2out.numpy())
                    np.save(dir_results + "\Epoch{:03d}Batch{:03d}TSNE_proj".format(epoch, batch), a2out_proj)
                    np.save(dir_results + "\Epoch{:03d}Batch{:03d}TAGS".format(epoch, batch), ytsne)

            batch += 1

        print("-------------------------------------------------------")
        print("At End Of Epoch {:03d}:    Accuracy on Test Set:    {:.4f}".format(epoch, compute_accuracy(model(xtest), ytest).numpy()))
        print("-------------------------------------------------------")

        # end epoch

    return train_loss_results, train_accuracy_results, activations





