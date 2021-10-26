import approximator
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
        zbound,
        # n_func
        n_func,
        f,  # return a vector of size n_func
        z_generator,
        # initializers for optimizing x
        n_init_x,
        # init_x (n_func, n_init_x, xdim) and neighbor_width are arguments
        #   of the optimize method of this class
        # estimate the VaR in a small neighborhood
        # defined by a square of size width, centered at current value of x
        # neighbor_width,
        graph,
        surrogate_config,
        # should we use different surrogate for different initializer or same surrogate?
        # if we use different surrogates, we need n_func * n_init_x surrogate
        # else we need n_func surrogate
        # NOTE: SOLUTION: we use 1 surrogate (1 big neural network) that returns (n_func, n_init_x) values
        name="varopt",
        dtype=tf.float64,
    ):
        """
        xbound[0]: min_x, xbound[1]: max_x
        zbound[0]: min_x, xbound[1]: max_x

        f(x,z) returns a vector
            f_vals = self.f(
                self.x_samples, # (n_func, n_init_x, n_x_sample, 1, xdim)
                self.z_samples) # (n_z_samples,zdim)
            # (n_func, n_init_x, n_x_sample, n_z_samples)

        z_generator(nsample) returns samples of z
            (nsample,zdim)
        surrogate_config: dictionary
            layer_sizes
            activations
        """
        self.graph = graph

        self.xdim = xdim
        self.zdim = zdim

        self.n_func = n_func
        self.n_init_x = n_init_x

        self.xbound = xbound
        self.zbound = zbound

        self.dtype = dtype

        self.f = f

        self.z_generator = z_generator
        self.name = name

        with self.graph.as_default():
            self.neighbor_center_plc = tf.placeholder(
                shape=(self.n_func, self.n_init_x, self.xdim),
                dtype=self.dtype,
                name="{}_neighbor_center_plc".format(self.name),
            )
            self.neighbor_width_plc = tf.placeholder(
                shape=(),
                dtype=self.dtype,
                name="{}_neighbor_width_plc".format(self.name),
            )
            # the neighbor_width will be reduced if the optimization happen in a neighborhood for a long time
            # i.e., refining the search

            # self.x should be fed with init_x passed to the optimize method
            self.x = tf.get_variable(
                initializer=tf.zeros(
                    shape=[self.n_func, self.n_init_x, self.xdim], dtype=self.dtype
                ),
                dtype=self.dtype,
                constraint=lambda x: tf.clip_by_value(
                    x,
                    clip_value_min=tf.math.maximum(
                        self.neighbor_center_plc - self.neighbor_width_plc,
                        self.xbound[0],
                    ),
                    clip_value_max=tf.math.minimum(
                        self.neighbor_center_plc + self.neighbor_width_plc,
                        self.xbound[1],
                    ),
                ),
                name="{}_x".format(self.name),
            )

            # OPTIMIZE surrogate
            self.quantile_plc = tf.placeholder(dtype=self.dtype, shape=())

            # we need an ANN version that can generate n_func independent output!
            self.surrogate = approximator.ANN(
                xdim=self.xdim,
                layer_sizes=surrogate_config["layer_sizes"],
                activations=surrogate_config["activations"],
                graph=tf.get_default_graph(),
            )

            self.surrogate_params = tf.get_variable(
                initializer=tf.random.truncated_normal(
                    shape=[self.n_func, self.n_init_x, self.surrogate.n_param],
                    stddev=0.01,
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
                name="{}_surrogate_param".format(self.name),
            )

            self.n_z_sample_plc = tf.placeholder(
                dtype=tf.int32, shape=(), name="{}_n_z_sample_plc".format(self.name)
            )
            self.z_samples = self.z_generator(self.n_z_sample_plc)
            # (n_z_samples,zdim)

            # for each function, and each initializer
            # we need n_x_sample_plc samples within their neighborhood
            self.n_x_sample_plc = tf.placeholder(
                dtype=tf.int32, shape=(), name="{}_n_x_sample_plc".format(self.name)
            )

            self.x_samples = (
                tf.random.uniform(
                    shape=(self.n_func, self.n_init_x, self.n_x_sample_plc, self.xdim),
                    dtype=self.dtype,
                )
                - 0.5
            ) * 2.0 * self.neighbor_width_plc + tf.expand_dims(
                self.neighbor_center_plc, axis=-2
            )
            # (n_func, n_init_x, n_x_sample, xdim)

            f_vals = self.f(
                tf.expand_dims(
                    self.x_samples, axis=3
                ),  # (n_func, n_init_x, n_x_sample, 1, xdim)
                self.z_samples,
            )  # (n_z_samples,zdim)
            # (n_func, n_init_x, n_x_sample, n_z_samples)

            self.predicted_var = self.surrogate.predict(
                self.x_samples,  # (n_func, n_init_x, n_x_sample, xdim)
                self.surrogate_params,  # (n_func, n_init_x, surrogate.n_param)
                config=None,
            )
            # (n_func, n_init, n_x_sample, 1)

            self.surrogate_loss = tf.reduce_mean(
                self.custom_mae(f_vals - self.predicted_var, self.quantile_plc)
            )

            self.surrogate_train = tf.train.AdamOptimizer().minimize(
                self.surrogate_loss, var_list=[self.surrogate_params]
            )

            # optimize x for maximum surrogate function

            surrogate_at_x = self.surrogate.predict(
                tf.expand_dims(self.x, axis=2),  # (n_func, n_init_x, 1, xdim)
                self.surrogate_params,  # (n_func, n_init_x, surrogate.n_param)
                config=None,
            )
            # (n_func, n_init, 1, 1)

            self.var_loss = tf.reduce_mean(surrogate_at_x)
            self.var_train = tf.train.AdamOptimizer().minimize(
                -self.var_loss, var_list=[self.x]
            )

            self.surrogate_at_x = tf.squeeze(
                tf.squeeze(surrogate_at_x, axis=-1), axis=-1
            )
            # (n_func, n_init)

            max_idx = tf.math.argmax(self.surrogate_at_x, axis=-1)
            # (n_func,)

            inc_idxs = tf.range(self.n_func, dtype=tf.int64)
            idxs = tf.transpose(tf.stack([inc_idxs, max_idx]))
            # (n_func,2)
            self.max_surrogate_at_x = tf.gather_nd(self.surrogate_at_x, indices=idxs)
            # (n_func,)
            self.max_x = tf.gather_nd(self.x, indices=idxs)
            # (n_func,xdim)

            # find quantile of x_plc
            self.x_plc = tf.placeholder(dtype=self.dtype, shape=(1, 50, self.xdim))
            (
                self.quantile_find_quantile_at_x_plc,
                self.loss_find_quantile_at_x_plc,
                self.train_find_quantile_at_x_plc,
            ) = self.find_quantile(
                self.x_plc,
                n_func=1,
                n_x=50,
                name="{}_find_quantile_for_x_plc".format(self.name),
            )

    def find_quantile(self, x, n_func, n_x, name="find_quantile"):
        # x: (n_func, n_x, xdim)
        # quantile value should be fed to self.quantile_plc
        quantile_var = tf.get_variable(
            initializer=tf.zeros(shape=[n_func, n_x, 1], dtype=self.dtype),
            dtype=self.dtype,
            name="{}_{}_quantile_var".format(name, self.name),
        )

        f_vals = self.f(
            tf.expand_dims(
                tf.expand_dims(x, axis=-2), axis=-2
            ),  # (n_func, n_x, 1, 1, xdim)
            self.z_samples,
        )  # (n_z_samples,zdim)
        # (n_func, n_x, 1, n_z_samples)
        print("x:", x)
        print("f_vals 1:", f_vals)

        f_vals = tf.squeeze(f_vals, axis=-2)
        # (n_func, n_x, n_z_samples)
        print("f_vals 2:", f_vals)

        loss = tf.reduce_mean(self.custom_mae(f_vals - quantile_var, self.quantile_plc))

        train = tf.train.AdamOptimizer().minimize(loss, var_list=[quantile_var])

        return tf.squeeze(quantile_var, axis=-1), loss, train

    def custom_mae(self, x, w):

        return tf.where(
            x > tf.cast(0.0, dtype=self.dtype),
            w * x,
            -tf.cast(1 - w, dtype=self.dtype) * x,
        )

    def maximize(
        self,
        quantile,
        n_x_sample,
        n_z_sample,
        neighbor_width,
        init_x,  # (self.n_func, self.n_init_x, xdim)
        n_x_train,
        n_z_train,
        n_iter_force_update_surrogate=1000,  # update the surrogate after 1000 iterations
        verbose=0,
    ):
        """
        neighbor_center_plc, neighbor_width_plc
        quantile_plc
        n_z_sample_plc
        n_x_sample_plc
        """

        with self.graph.as_default():
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                self.x.load(init_x, sess)

                cur_neighbor_center = init_x
                cur_neighbor_width = neighbor_width

                t = time.time()
                last_ix = 0

                for ix in range(n_x_train):
                    # update center and width if necessary
                    cur_x = sess.run(self.x)
                    # (n_func, n_init_x, xdim)

                    # NOTE: this is inefficient
                    # as 1 initializer for 1 function move out of neighborhood
                    # we re-train for all initializers of all functions!
                    if (
                        np.any(
                            np.abs(cur_x - cur_neighbor_center)
                            >= cur_neighbor_width - 1e-5
                        )
                        or ix == 0
                        or ix - last_ix > n_iter_force_update_surrogate
                    ):

                        if ix - last_ix > n_iter_force_update_surrogate and np.all(
                            np.sqrt(
                                np.sum(np.square(cur_x - cur_neighbor_center), axis=-1)
                            )
                            < cur_neighbor_width / 2.0
                        ):
                            cur_neighbor_width /= 2.0
                            print(
                                "Reduce neighbor width to {}".format(cur_neighbor_width)
                            )
                        else:
                            print("Update neighbor center")

                        cur_neighbor_center = cur_x

                        last_ix = ix

                        for iz in range(n_z_train):
                            sess.run(
                                self.surrogate_train,
                                feed_dict={
                                    self.neighbor_center_plc: cur_neighbor_center,
                                    self.neighbor_width_plc: cur_neighbor_width,
                                    self.quantile_plc: quantile,
                                    self.n_z_sample_plc: n_z_sample,
                                    self.n_x_sample_plc: n_x_sample,
                                },
                            )

                    sess.run(
                        self.var_train,
                        feed_dict={
                            self.neighbor_center_plc: cur_neighbor_center,
                            self.neighbor_width_plc: cur_neighbor_width,
                            self.quantile_plc: quantile,
                            self.n_z_sample_plc: n_z_sample,
                            self.n_x_sample_plc: n_x_sample,
                        },
                    )

                    if verbose and ix % verbose == 0:
                        var_loss_np = sess.run(
                            self.var_loss,
                            feed_dict={
                                self.neighbor_center_plc: cur_neighbor_center,
                                self.neighbor_width_plc: cur_neighbor_width,
                                self.quantile_plc: quantile,
                                self.n_z_sample_plc: n_z_sample,
                                self.n_x_sample_plc: n_x_sample,
                            },
                        )

                        print(
                            "{}. Loss {} in {:.2f}s.".format(
                                ix, var_loss_np, time.time() - t
                            )
                        )
                        print("    {}".format(sess.run(self.x)))
                        t = time.time()
                        sys.stdout.flush()

                max_surrogate_at_x_np, max_x_np = sess.run(
                    [self.max_surrogate_at_x, self.max_x],
                    feed_dict={
                        self.neighbor_center_plc: cur_neighbor_center,
                        self.neighbor_width_plc: cur_neighbor_width,
                        self.quantile_plc: quantile,
                        self.n_z_sample_plc: n_z_sample,
                        self.n_x_sample_plc: n_x_sample,
                    },
                )

        return max_x_np, max_surrogate_at_x_np

    def maximize_in_session(
        self,
        sess,  # session
        n_x_train,
        n_z_train,
        n_iter_force_update_surrogate=1000,  # update the surrogate after 1000 iterations
        feed_dict={},
        verbose=0,
    ):
        """
        neighbor_center_plc and neighbor_width_plc are adaptively tuned in this function
        in feed_dict:
            neighbor_width_plc
            neighbor_center_plc (which is the initializer of x)
            quantile_plc
            n_z_sample_plc
            n_x_sample_plc
        """

        with self.graph.as_default():
            cur_neighbor_center = feed_dict[self.neighbor_center_plc]
            cur_neighbor_width = feed_dict[self.neighbor_width_plc]

            self.x.load(cur_neighbor_center, sess)

            t = time.time()
            last_ix = 0

            for ix in range(n_x_train):
                # update center and width if necessary
                cur_x = sess.run(self.x)
                # (n_func, n_init_x, xdim)

                # NOTE: this is inefficient
                # as 1 initializer for 1 function move out of neighborhood
                # we re-train for all initializers of all functions!
                if (
                    np.any(
                        np.abs(cur_x - cur_neighbor_center) >= cur_neighbor_width - 1e-5
                    )
                    or ix == 0
                    or ix - last_ix > n_iter_force_update_surrogate
                ):

                    if ix - last_ix > n_iter_force_update_surrogate and np.all(
                        np.sqrt(np.sum(np.square(cur_x - cur_neighbor_center), axis=-1))
                        < cur_neighbor_width / 2.0
                    ):
                        cur_neighbor_width /= 2.0
                        print("Reduce neighbor width to {}".format(cur_neighbor_width))
                    else:
                        print("Update neighbor center")

                    cur_neighbor_center = cur_x

                    feed_dict[self.neighbor_center_plc] = cur_neighbor_center
                    feed_dict[self.neighbor_width_plc] = cur_neighbor_width

                    last_ix = ix

                    for iz in range(n_z_train):
                        sess.run(self.surrogate_train, feed_dict)

                sess.run(self.var_train, feed_dict)

                if verbose and ix % verbose == 0:
                    var_loss_np = sess.run(self.var_loss, feed_dict)

                    print(
                        "{}. Loss {} in {:.2f}s.".format(
                            ix, var_loss_np, time.time() - t
                        )
                    )
                    # print("    {}".format(sess.run(self.x)))
                    t = time.time()
                    sys.stdout.flush()

            max_surrogate_at_x_np, max_x_np = sess.run(
                [self.max_surrogate_at_x, self.max_x], feed_dict
            )

        return max_x_np, max_surrogate_at_x_np

    def find_max_in_set(
        self, sess, xs, feed_dict={}, ntrain=1000, verbose=0  # session  # (nx,xdim)
    ):
        """
        neighbor_center_plc and neighbor_width_plc are adaptively tuned in this function
        in feed_dict:
            neighbor_width_plc
            neighbor_center_plc (which is the initializer of x)
            quantile_plc
            n_z_sample_plc
            n_x_sample_plc
        """
        nx = xs.shape[0]
        new_dict = feed_dict.copy()

        max_var = -1e9
        max_x = None
        batchsize = 50  # self.x_plc is created with nx = 50

        with self.graph.as_default():

            for i in range(0, nx, batchsize):
                xs_i = xs[i : (i + batchsize)]

                while xs_i.shape[0] < batchsize:
                    xs_i = np.concatenate([xs_i, xs_i], axis=0)
                xs_i = xs_i[:batchsize]

                new_dict[self.x_plc] = xs_i.reshape(1, batchsize, self.xdim)

                for _ in range(ntrain):
                    sess.run(self.train_find_quantile_at_x_plc, feed_dict=new_dict)

                var_np = sess.run(
                    self.quantile_find_quantile_at_x_plc, feed_dict=new_dict
                )
                var_np = var_np.reshape(
                    batchsize,
                )

                max_idx = np.argmax(var_np)

                if max_var < var_np[max_idx]:
                    max_var = var_np[max_idx]
                    max_x = xs_i[max_idx]

        return max_x, max_var
