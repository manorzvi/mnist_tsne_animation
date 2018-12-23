import tensorflow as tf

class mnistModel(tf.keras.Model):
    def __init__(self):
        super(mnistModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(900, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False, a1out=False, a2out=False):
        a1 = self.dense1(inputs)
        a2 = self.dense2(a1)
        if a1out and a2out:
            raise ValueError('both flags are True!')
        elif a1out:
            return a1
        elif a2out:
            return a2
        else:
            return self.dense3(a2)