import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ANN:
    def __init__(self, xdim, layer_sizes, activations, graph):
        assert len(layer_sizes) == len(
            activations
        ), "The length of layer_sizes and activations must equal!"

        self.xdim = xdim
        self.nlayer = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activations = activations

        n_param_per_layer = np.zeros(self.nlayer, dtype=int)

        n_param_per_layer[0] = (xdim + 1) * layer_sizes[0]
        for i in range(self.nlayer - 1):
            n_param_per_layer[i + 1] = (layer_sizes[i] + 1) * layer_sizes[i + 1]

        cs_n_param_per_layer = np.cumsum(n_param_per_layer)

        self.n_param_per_layer = n_param_per_layer
        self.n_param = cs_n_param_per_layer[-1]
        self.cs_n_param_per_layer = cs_n_param_per_layer

        self.graph = graph

        with self.graph.as_default():
            self.params_plc = tf.placeholder(
                shape=(1, 1, self.n_param), dtype=tf.float64
            )
            self.x_plc = tf.placeholder(
                shape=(None, None, None, self.xdim), dtype=tf.float64
            )
            self.predicted_y = self.predict(self.x_plc, self.params_plc, None)

    @staticmethod
    def make_activation_layer(x, activation):
        if activation == "relu":
            network = tf.nn.relu(x)
        elif activation == "leaky_relu":
            network = tf.nn.leaky_relu(x, alpha=0.2)
        elif activation == "sigmoid":
            network = tf.nn.sigmoid(x)
        elif activation == "log_sigmoid":
            network = tf.math.log_sigmoid(x)
        elif activation == "linear":
            network = x
        elif activation == "square":
            network = tf.square(x)
        elif activation == "softmax":
            network = tf.nn.softmax(x)
        elif activation == "log_softmax":
            network = tf.nn.log_softmax(x)
        else:
            raise Exception(
                "Only allow relu, leaky_relu, sigmoid, linear. Unknown activation: {}".format(
                    activation
                )
            )

        return network

    def predict_np(self, x, params, config):

        with tf.Session() as sess:
            predicted_y_np = sess.run(
                self.predicted_y, feed_dict={self.x_plc: x, self.params_plc: params}
            )

        return predicted_y_np

    def predict(self, x, params, config):
        # nlayer: scalar
        # layer_sizes: list of size nlayer
        # activations: list of activations
        # x:      (n|1, m, npoint, xdim)
        # params: (n, m|1, n_param)
        """
        nlayer = 2
        layer_size = [2,3]
        network: xdim -> 2 -> 3
        (output_dim = 3)
        returns:
            network
            params: list of tensor of shape:
                [xdim+1,layer_size[0]]
                [layer_size[i]+1, layer_size[i+1]]
        """

        with self.graph.as_default():
            xdim = self.xdim
            nlayer = self.nlayer
            layer_sizes = self.layer_sizes
            activations = self.activations

            cs_n_param_per_layer = self.cs_n_param_per_layer

            param_0 = tf.gather(
                params, indices=list(range(cs_n_param_per_layer[0])), axis=-1
            )
            param_0 = tf.reshape(
                param_0,
                shape=(
                    tf.shape(param_0)[0],
                    tf.shape(param_0)[1],
                    xdim + 1,
                    layer_sizes[0],
                ),
            )

            W = tf.gather(param_0, indices=list(range(xdim)), axis=-2)
            # (n, m|1, xdim, layer_sizes[0])
            b = tf.gather(param_0, indices=[xdim], axis=-2)
            # (n, m|1, 1,    layer_sizes[0])

            net = ANN.make_activation_layer(x @ W + b, activations[0])

            for i in range(nlayer - 1):
                param_i = tf.gather(
                    params,
                    indices=list(
                        range(cs_n_param_per_layer[i], cs_n_param_per_layer[i + 1])
                    ),
                    axis=-1,
                )
                param_i = tf.reshape(
                    param_i,
                    shape=(
                        tf.shape(param_i)[0],
                        tf.shape(param_i)[1],
                        layer_sizes[i] + 1,
                        layer_sizes[i + 1],
                    ),
                )
                W = tf.gather(param_i, indices=list(range(layer_sizes[i])), axis=-2)
                # (n, m|1, layer_sizes[i], layer_sizes[i+1])
                b = tf.gather(param_i, indices=[layer_sizes[i]], axis=-2)
                # (n, m|1, 1,              layer_sizes[i+1])

                net = ANN.make_activation_layer(net @ W + b, activations[i + 1])

        return net
