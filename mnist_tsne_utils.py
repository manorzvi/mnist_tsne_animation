import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np



def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


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





