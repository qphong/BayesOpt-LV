import numpy as np
import tensorflow as tf
import sys
import time


class CVaROpt:
    """This class is to optimize CVaR of a f(x,Z) given
    the discrete distribution of Z: z_values, z_probs

    get_quantile (VaR)

    get_quantile_less_than (all VaR at alpha' <= a specified alpha)

    get_cvar (CVaR): using the result from get_quantile
    """

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
        self.quantile_plc = tf.placeholder(dtype=self.dtype, shape=())

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
            (
                self.quantile_f_vals,
                self.lt_risk_levels,
                self.lt_quantile_f_probs,
                self.lt_quantile_f_vals,
                self.cvar,
            ) = self.find_cvar(self.x)

            self.cvar_loss = tf.reduce_mean(self.cvar)
            self.cvar_train = tf.train.AdamOptimizer().minimize(
                -self.cvar_loss, var_list=[self.x]
            )

            max_idx = tf.math.argmax(self.cvar)
            self.max_cvar_val = tf.gather(self.cvar, indices=max_idx, axis=0)
            self.max_x = tf.gather(self.x, indices=max_idx, axis=0)
            # (xdim,)

            self.max_lt_risk_levels = tf.gather(
                self.lt_risk_levels, indices=max_idx, axis=0
            )
            # (nz,)
            self.max_lt_quantile_f_vals = tf.gather(
                self.lt_quantile_f_vals, indices=max_idx, axis=0
            )
            # (nz,)

            # find cvar of x_plc
            self.x_plc = tf.placeholder(dtype=self.dtype, shape=(None, self.xdim))
            _, _, _, _, self.cvar_at_x_plc = self.find_cvar(self.x_plc)
            # (None,)

    def find_cvar(self, x):
        """Find CVaR at x at risk level self.quantile_plc
        return VaR at x at risk level self.quantile_plc
               all possible risk levels <= self.quantile_plc
               (TONOTE:
                    the condition in VaR must be >= quantile
                )
               with their function values

        Returns:
            quantile_f_vals.shape = (n_init_x,)
            lt_risk_levels = (n_init_x, nz)
            lt_quantile_f_probs.shape = (n_init_x, nz)
            lt_quantile_f_vals.shape = (n_init_x, nz) (== f_vals)
            cvar = (n_init_x,)

        Example:
            Quantile values
            [0.89845524 0.75351679]
            ---
            lt risk levels
            [[0.33333334 0.5        0.        ]
            [0.33333334 0.5        0.        ]]
            ---
            lt quantile f probs
            [[0.66666669 0.33333331 0.        ]
            [0.66666669 0.33333331 0.        ]]
            ---
            lt quantile f vals
            [[0.57796961 0.89845524 0.99896768]
            [0.34608511 0.75351679 0.97646129]]
            ---
            cvar
            [0.68479815 0.48189566]
            ---
        """

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
            sorted_f_vals_cprobs >= self.quantile_plc,
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
        # (n_init_x,)

        # CVaR is a continuous function
        # shift_by_1_idx = tf.stop_gradient(tf.concat([
        #         tf.cast(inc_idxs, dtype=tf.int64),
        #         quantile_f_idxs-1], axis=-1))
        # the above is incorrect as index can be negative
        # so we can compute CVaR up to and include VaR first
        # then subtracting the extra part

        # (n_init_x, 2)
        quantile_f_cprobs = tf.gather_nd(quantile_f_cprobs, indices=idxs)
        quantile_f_cprobs = tf.reshape(quantile_f_cprobs, shape=(tf.shape(x)[0], 1))
        # (n_init_x, 1)

        # note that the probability of VaR > the probability in CVaR computation
        lt_unnormalized_quantile_f_probs = tf.where(
            sorted_f_vals_cprobs - quantile_f_cprobs <= 0.0,
            sorted_f_vals_probs,
            tf.zeros_like(sorted_f_vals_probs, dtype=self.dtype),
        )
        # (n_init_x, nz)

        # subtracting the extra probability of VaR in CVaR computation
        lt_unnormalized_quantile_f_probs = tf.where(
            tf.abs(sorted_f_vals_cprobs - quantile_f_cprobs) <= 1e-9,
            sorted_f_vals_probs - quantile_f_cprobs + self.quantile_plc,
            lt_unnormalized_quantile_f_probs,
        )
        # (n_init_x, nz)

        lt_risk_levels = tf.math.cumsum(lt_unnormalized_quantile_f_probs, axis=-1)
        # (n_init_x, nz)
        lt_risk_levels = tf.where(
            sorted_f_vals_cprobs - quantile_f_cprobs <= 0.0,
            lt_risk_levels,
            tf.zeros_like(lt_risk_levels, dtype=self.dtype),
        )
        # (n_init_x, nz)

        lt_quantile_f_probs = lt_unnormalized_quantile_f_probs / tf.reduce_sum(
            lt_unnormalized_quantile_f_probs, axis=-1, keepdims=True
        )
        # (n_init_x, nz)

        lt_quantile_f_vals = tf.sort(f_vals, axis=1)
        # (n_init_x, nz)

        cvar = tf.reduce_sum(lt_quantile_f_probs * lt_quantile_f_vals, axis=-1)
        # (n_init_x,)

        # return quantile_zs, quantile_f_vals
        return (
            quantile_f_vals,
            lt_risk_levels,
            lt_quantile_f_probs,
            lt_quantile_f_vals,
            cvar,
        )


    def maximize_in_session(
        self, sess, init_x, n_x_train, feed_dict={}, verbose=0  # (self.n_init_x, xdim)
    ):
        """Maximize CVaR of a function f(x,z) given a distribution of Z

        Returns:
            max_quantile_f_val_np: scalar
            max_x_np: (xdim,)
            max_lt_risk_levels_np: (nz,)
            max_lt_quantile_f_vals_np: (nz,)
        """

        with self.graph.as_default():
            self.x.load(init_x, sess)

            t = time.time()

            for ix in range(n_x_train):

                sess.run(self.cvar_train, feed_dict=feed_dict)

                if verbose and ix % verbose == 0:
                    # cvar_loss_np = sess.run(self.cvar_loss,
                    #     feed_dict=feed_dict)

                    (
                        max_cvar_val_np,
                        max_x_np,
                        max_lt_risk_levels_np,
                        max_lt_quantile_f_vals_np,
                    ) = sess.run(
                        [
                            self.max_cvar_val,
                            self.max_x,
                            self.max_lt_risk_levels,
                            self.max_lt_quantile_f_vals,
                        ],
                        feed_dict,
                    )

                    print(
                        "{}. CVaR: {} in {:.2f}s.".format(
                            ix, max_cvar_val_np, time.time() - t
                        )
                    )

                    t = time.time()
                    sys.stdout.flush()

            (
                max_cvar_val_np,
                max_x_np,
                max_lt_risk_levels_np,
                max_lt_quantile_f_vals_np,
            ) = sess.run(
                [
                    self.max_cvar_val,
                    self.max_x,
                    self.max_lt_risk_levels,
                    self.max_lt_quantile_f_vals,
                ],
                feed_dict,
            )

        return (
            max_cvar_val_np,
            max_x_np,
            max_lt_risk_levels_np,
            max_lt_quantile_f_vals_np,
        )

    def find_max_in_set(
        self, sess, xs, feed_dict={}, batchsize=10, verbose=0  # (nx, xdim)
    ):
        # find x with max VaR in xs
        # return x and VaR

        nx = xs.shape[0]
        new_dict = feed_dict.copy()

        max_x = None
        max_cvar = -1e9

        with self.graph.as_default():

            for i in range(0, nx, batchsize):
                new_dict[self.x_plc] = xs[i : (i + batchsize)]

                cvar_np = sess.run(self.cvar_at_x_plc, feed_dict=new_dict)
                cvar_np = np.reshape(cvar_np, (-1,))

                max_idx = np.argmax(cvar_np)

                if max_cvar < cvar_np[max_idx]:
                    max_cvar = cvar_np[max_idx]
                    max_x = xs[i : (i + batchsize)][max_idx]

        return max_x, max_cvar
