import numpy as np
import tensorflow as tf
import sys
import time


class VaROpt:
    def __init__(
        self,
        xdim,
        zdim,
        xbound,
        # # n_func
        # n_func,
        # Let us focus on n_func = 1 for easier implementation of
        # searching for the quantile
        f,  # return a vector of size n_func
        nz,  # integer: number of z_values
        z_values,  # (nz,zdim) vector of z values
        z_probs,  # (nz,) vector of probabilities
        # tf.reduce_sum(z_probs) = 1
        # initializers for optimizing x
        n_init_x,
        graph,
        name,
        dtype,
    ):
        """
        xbound[0]: min_x, xbound[1]: max_x

        f(x,z) returns a vector
            f_vals = self.f(
                self.x_samples, # (n_func, n_init_x, n_x_sample, 1, xdim)
                self.z_samples) # (n_z_samples,zdim)
            # (n_func, n_init_x, n_x_sample, n_z_samples)
        n_x_samples = 1 (it is only used for continuous z)
        """
        self.graph = graph

        self.xdim = xdim
        self.zdim = zdim

        # self.n_func = n_func
        self.n_init_x = n_init_x

        self.xbound = xbound

        self.dtype = dtype

        self.f = f
        self.nz = nz
        self.z_values = tf.constant(z_values, dtype=dtype)
        self.z_probs = tf.constant(z_probs, dtype=dtype)

        with self.graph.as_default():
            # self.x should be fed with init_x passed to the optimize method
            self.x = tf.get_variable(
                initializer=tf.zeros(
                    shape=[self.n_init_x, self.xdim], dtype=self.dtype
                ),
                dtype=self.dtype,
                constraint=lambda x: tf.clip_by_value(
                    x, clip_value_min=self.xbound[0], clip_value_max=self.xbound[1]
                ),
                name="{}_x".format(name),
            )

            # OPTIMIZE surrogate
            self.quantile_plc = tf.placeholder(dtype=self.dtype, shape=())

            self.quantile_f_vals = self.find_quantile(self.x)

            self.var_loss = tf.reduce_mean(self.quantile_f_vals)
            self.var_train = tf.train.AdamOptimizer().minimize(
                -self.var_loss, var_list=[self.x]
            )

            max_idx = tf.math.argmax(tf.squeeze(self.quantile_f_vals))
            self.max_quantile_f_val = tf.gather(
                self.quantile_f_vals, indices=max_idx, axis=0
            )
            self.max_x = tf.gather(self.x, indices=max_idx, axis=0)

            # find quantile of x_plc
            self.x_plc = tf.placeholder(dtype=self.dtype, shape=(None, self.xdim))
            self.quantile_f_vals_at_x_plc = self.find_quantile(self.x_plc)

    def find_quantile(self, x):
        # x: (n_init_x,xdim)
        # self.quantile_plc
        f_vals = self.f(
            tf.expand_dims(
                tf.expand_dims(tf.expand_dims(x, axis=0), axis=-2), axis=-2
            ),  # (1, n_init_x, 1, 1, xdim)
            self.z_values,
        )  # (nz,zdim)
        # (1, n_init_x, 1, nz)

        f_vals = tf.squeeze(f_vals, axis=0)
        f_vals = tf.squeeze(f_vals, axis=1)
        # (n_init_x, nz)

        sorted_f_vals_idxs = tf.argsort(f_vals, axis=1)
        # (n_init_x, nz)

        sorted_f_vals_probs = tf.gather(self.z_probs, indices=sorted_f_vals_idxs)
        # (n_init_x, nz)

        sorted_f_vals_cprobs = tf.math.cumsum(sorted_f_vals_probs, axis=-1)
        # (n_init_x, nz)

        quantile_f_cprobs = tf.where(
            sorted_f_vals_cprobs > self.quantile_plc,
            sorted_f_vals_cprobs,
            tf.ones_like(sorted_f_vals_cprobs, dtype=self.dtype)
            * tf.cast(100.0, dtype=self.dtype),
        )
        # (n_init_x, nz)

        quantile_f_idxs = tf.expand_dims(
            tf.math.argmin(quantile_f_cprobs, axis=-1), axis=-1
        )
        # (n_init_x,1)

        # inc_idxs = tf.expand_dims(tf.constant(list(range(self.n_init_x)), dtype=tf.int64), axis=-1)
        inc_idxs = tf.expand_dims(tf.range(tf.shape(x)[0]), axis=-1)
        # (n_init_x,1)

        idxs = tf.stop_gradient(
            tf.concat([tf.cast(inc_idxs, dtype=tf.int64), quantile_f_idxs], axis=-1)
        )
        # (n_init_x, 2)

        quantile_z_idxs = tf.expand_dims(
            tf.gather_nd(sorted_f_vals_idxs, indices=idxs), axis=-1
        )
        # (n_init_x, 1)

        quantile_f_idxs = tf.stop_gradient(
            tf.concat([inc_idxs, quantile_z_idxs], axis=-1)
        )

        quantile_f_vals = tf.gather_nd(f_vals, indices=quantile_f_idxs)
        return quantile_f_vals

    def maximize(
        self, init_x, n_x_train, feed_dict={}, verbose=0  # (self.n_init_x, xdim)
    ):

        with self.graph.as_default():
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                self.x.load(init_x, sess)

                t = time.time()

                for ix in range(n_x_train):

                    sess.run(self.var_train, feed_dict=feed_dict)

                    if verbose and ix % verbose == 0:
                        var_loss_np = sess.run(self.var_loss, feed_dict=feed_dict)

                        print(
                            "{}. VaR {} in {:.2f}s.".format(
                                ix, var_loss_np, time.time() - t
                            )
                        )

                        t = time.time()
                        sys.stdout.flush()

                x_np = sess.run(self.x)
                # (n_init_x, xdim)

        return x_np

    def maximize_in_session(
        self, sess, init_x, n_x_train, feed_dict={}, verbose=0  # (self.n_init_x, xdim)
    ):

        with self.graph.as_default():
            self.x.load(init_x, sess)

            t = time.time()

            for ix in range(n_x_train):

                sess.run(self.var_train, feed_dict=feed_dict)

                if verbose and ix % verbose == 0:
                    var_loss_np = sess.run(self.var_loss, feed_dict=feed_dict)

                    print(
                        "{}. VaR {} in {:.2f}s.".format(
                            ix, var_loss_np, time.time() - t
                        )
                    )

                    t = time.time()
                    sys.stdout.flush()

            max_quantile_f_val_np, max_x_np = sess.run(
                [self.max_quantile_f_val, self.max_x], feed_dict
            )

            # max_quantile_f_val_np: scalar
            # max_x_np: (1, xdim)

        return max_x_np, max_quantile_f_val_np

    def find_max_in_set(
        self, sess, xs, feed_dict={}, batchsize=10, verbose=0  # (nx, xdim)
    ):
        # find x with max VaR in xs
        # return x and VaR

        nx = xs.shape[0]
        new_dict = feed_dict.copy()

        max_x = None
        max_var = -1e9

        with self.graph.as_default():

            for i in range(0, nx, batchsize):
                new_dict[self.x_plc] = xs[i : (i + batchsize)]

                var_np = sess.run(self.quantile_f_vals_at_x_plc, feed_dict=new_dict)

                var_np = np.reshape(var_np, (-1,))

                max_idx = np.argmax(var_np)

                if max_var < var_np[max_idx]:
                    max_var = var_np[max_idx]
                    max_x = xs[i : (i + batchsize)][max_idx]

        return max_x, max_var
