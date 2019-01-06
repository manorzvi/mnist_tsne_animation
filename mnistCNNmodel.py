import tensorflow as tf

class mnistCNNmodel(tf.keras.Model):
    def __init__(self):
        super(mnistCNNmodel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation=tf.nn.relu, input_shape=(28,28,1))
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation=tf.nn.relu)
        self.maxPool2D1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation=tf.nn.relu)
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation=tf.nn.relu)
        self.maxPool2D2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False, a1out=False, a2out=False):

        inputs_np = inputs.numpy()

        inputs_np2 = inputs_np.reshape(inputs_np.shape[0], 28, 28, 1)

        inputs_tf2 = tf.convert_to_tensor(inputs_np2)

        a1 = self.conv1(inputs_tf2)
        a2 = self.conv2(a1)
        a3 = self.maxPool2D1(a2)
        a4 = self.dropout1(a3)
        a5 = self.conv3(a4)
        a6 = self.conv4(a5)
        a7 = self.maxPool2D2(a6)
        a8 = self.dropout2(a7)
        a9 = self.flatten(a8)
        a10 = self.dense1(a9)
        a11 = self.dropout3(a10)
        a12 = self.dense2(a11)
        a13 = self.dense3(a12)

        if a1out and a2out:
            raise ValueError('both flags are True!')
        elif a1out:
            return a11
        elif a2out:
            return a12
        else:
            return a13
