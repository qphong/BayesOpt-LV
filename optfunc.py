import tensorflow as tf
import numpy as np
import scipy as sp
import time
import scipy.stats as spst
import sys
import utils

import matplotlib.pyplot as plt


# draw random features, and their weights
def draw_random_init_weights_features(
    xdim,
    n_funcs,
    n_features,
    xx,  # (nobs, xdim)
    yy,  # (nobs, 1)
    l,
    sigma,
    sigma0,
    # (1,xdim), (), ()
    dtype=tf.float32,
    name="random_features",
):
    """
    sigma, sigma0: scalars
    l: 1 x xdim
    xx: n x xdim
    yy: n x 1
    n_features: a scalar
    different from draw_random_weights_features,
        this function set W, b, noise as Variable that is initialized randomly
        rather than sample W, b, noise from random function
    """

    n = tf.shape(xx)[0]

    xx = tf.tile(tf.expand_dims(xx, axis=0), multiples=(n_funcs, 1, 1))
    yy = tf.tile(tf.expand_dims(yy, axis=0), multiples=(n_funcs, 1, 1))
    idn = tf.tile(
        tf.expand_dims(tf.eye(n, dtype=dtype), axis=0), multiples=(n_funcs, 1, 1)
    )

    # draw weights for the random features.
    W = (
        tf.get_variable(
            name="{}_W".format(name),
            shape=(n_funcs, n_features, xdim),
            dtype=dtype,
            initializer=tf.random_normal_initializer(),
        )
        * tf.tile(
            tf.expand_dims(tf.sqrt(l), axis=0), multiples=(n_funcs, n_features, 1)
        )
    )
    # n_funcs x n_features x xdim

    b = (
        2.0
        * np.pi
        * tf.get_variable(
            name="{}_b".format(name),
            shape=(n_funcs, n_features, 1),
            dtype=dtype,
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0),
        )
    )
    # n_funcs x n_features x 1

    # compute the features for xx.
    Z = tf.cast(tf.sqrt(2.0 * sigma / n_features), dtype=dtype) * tf.cos(
        tf.matmul(W, xx, transpose_b=True) + tf.tile(b, multiples=(1, 1, n))
    )
    # n_funcs x n_features x n

    # draw the coefficient theta.
    noise = tf.get_variable(
        name="{}_noise".format(name),
        shape=(n_funcs, n_features, 1),
        dtype=dtype,
        initializer=tf.random_normal_initializer(),
    )
    # n_funcs x n_features x 1

    def true_clause():
        Sigma = tf.matmul(Z, Z, transpose_a=True) + sigma0 * idn
        # n_funcs x n x n of rank n or n_features

        mu = tf.matmul(
            tf.matmul(Z, utils.multichol2inv(Sigma, n_funcs, dtype=dtype)), yy
        )
        # n_funcs x n_features x 1

        e, v = tf.linalg.eigh(Sigma)
        e = tf.expand_dims(e, axis=-1)
        # n_funcs x n x 1

        r = tf.reciprocal(tf.sqrt(e) * (tf.sqrt(e) + tf.sqrt(sigma0)))
        # n_funcs x n x 1

        theta = (
            noise
            - tf.matmul(
                Z,
                tf.matmul(
                    v,
                    r
                    * tf.matmul(
                        v, tf.matmul(Z, noise, transpose_a=True), transpose_a=True
                    ),
                ),
            )
            + mu
        )
        # n_funcs x n_features x 1

        return theta

    def false_clause():
        Sigma = utils.multichol2inv(
            tf.matmul(Z, Z, transpose_b=True) / sigma0
            + tf.tile(
                tf.expand_dims(tf.eye(n_features, dtype=dtype), axis=0),
                multiples=(n_funcs, 1, 1),
            ),
            n_funcs,
            dtype=dtype,
        )

        mu = tf.matmul(tf.matmul(Sigma, Z), yy) / sigma0

        theta = mu + tf.matmul(tf.cholesky(Sigma), noise)
        return theta

    theta = tf.cond(
        pred=tf.less(n, n_features), true_fn=true_clause, false_fn=false_clause
    )

    return theta, W, b


def make_function_sample(x, n_features, sigma, theta, W, b, dtype=tf.float32):
    fval = tf.squeeze(
        tf.sqrt(tf.cast(2.0 * sigma / n_features, dtype=dtype))
        * tf.matmul(
            theta, tf.cos(tf.matmul(W, x, transpose_b=True) + b), transpose_a=True
        )
    )
    return fval


# find maximum of a function with multiple initializers
# a function is a tensor, so this function can be used in the above function
def find_maximum_with_multiple_init_tensor(
    xs_list, fvals, n_inits, xdim, optimizer, dtype=tf.float32
):
    """
    # xmin=-np.infty, xmax=np.infty,
    xs: list of size n_inits of (1,xdim)
    fvals: (n_inits,): function value with inputs are xs tensor
    initializers: n_inits x xdim
    """
    # initializers: n x d
    # func: a tensor function
    #     input:  tensor n x d
    #     output: tensor n x 1
    # n_inits: scalar (not a tensor)
    """
    returns:
        vals: shape = (n_inits,)
        invars: shape = (n_inits,xdim)
        maxval: scalar
        maxinvar: shape= (xdim,)
    """

    trains = [None] * n_inits

    for i in range(n_inits):
        trains[i] = optimizer.minimize(-fvals[i], var_list=[xs_list[i]])

    max_idx = tf.argmax(fvals)
    return trains, max_idx