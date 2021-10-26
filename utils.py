import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy as sp
import scipy.stats as spst
import sys

clip_min = 1e-100


def chol2inv(mat, dtype=tf.float32):

    n = tf.shape(mat)[0]
    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), tf.eye(n, dtype=dtype))
    invmat = tf.transpose(invlower) @ invlower
    return invmat


def multichol2inv(mat, n_mat, dtype=tf.float32):
    # lower = tf.linalg.cholesky(mat)
    invlower = tf.matrix_solve(
        tf.linalg.cholesky(mat),
        tf.tile(
            tf.expand_dims(tf.eye(tf.shape(mat)[1], dtype=dtype), axis=0),
            multiples=(n_mat, 1, 1),
        ),
    )
    invmat = tf.matmul(invlower, invlower, transpose_a=True)
    return invmat


def computeKnm(X, Xbar, l, sigma, kernel_type="matern52", dtype=tf.float32):
    """
    X: n x d
    l: d
    change from SE kernel_type to Matern 5/2 kernel_type
    sigma * (1 + \sqrt{5} dist + 5/3 dist^2) exp(-\sqrt{5} dist)
        where dist = ||x - x'||_2 * l
    """
    n = tf.shape(X)[0]
    m = tf.shape(Xbar)[0]

    X = X * tf.sqrt(l)
    Xbar = Xbar * tf.sqrt(l)

    Q = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), multiples=(1, m))
    Qbar = tf.tile(
        tf.transpose(tf.reduce_sum(tf.square(Xbar), axis=1, keepdims=True)),
        multiples=(n, 1),
    )

    if kernel_type == "matern52":
        dist = tf.sqrt(tf.maximum(Qbar + Q - 2 * X @ tf.transpose(Xbar), 1e-36))
        knm = (
            sigma
            * (
                tf.cast(1.0, dtype=dtype)
                + tf.cast(tf.sqrt(5.0), dtype=dtype) * dist
                + tf.cast(5.0 / 3.0, dtype=dtype) * tf.square(dist)
            )
            * tf.exp(-tf.cast(tf.sqrt(5.0), dtype=dtype) * dist)
        )

    elif kernel_type == "se":
        dist = Qbar + Q - 2 * X @ tf.transpose(Xbar)
        knm = sigma * tf.exp(-0.5 * dist)

    else:
        raise Exception("Unknown kernel:", kernel_type)

    return knm


def computeKmm(
    X, l, sigma, nd=2, kernel_type="matern52", dtype=tf.float32, debug=False
):
    """
    X: (...,n,d)
    nd = len(tf.shape(X))
    l: (1,d)
    sigma: signal variance
    return (...,n,n)
    """
    n = tf.shape(X)[-2]
    X = X * tf.sqrt(tf.reshape(l, shape=(1, -1)))
    # (...,n,d)
    Q = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    # (...,n,1)

    transpose_idxs = np.array(list(range(nd)))
    transpose_idxs[-2] = nd - 1
    transpose_idxs[-1] = nd - 2

    if kernel_type == "matern52":
        dist = tf.sqrt(
            tf.maximum(
                Q
                + tf.transpose(Q, perm=transpose_idxs)
                - 2 * X @ tf.transpose(X, perm=transpose_idxs),
                1e-36,
            )
        )

        kmm = (
            sigma
            * (
                tf.cast(1.0, dtype=dtype)
                + tf.cast(tf.sqrt(5.0), dtype=dtype) * dist
                + tf.cast(5.0 / 3.0, dtype=dtype) * tf.square(dist)
            )
            * tf.exp(-tf.cast(tf.sqrt(5.0), dtype=dtype) * dist)
        )

    elif kernel_type == "se":
        dist = (
            Q
            + tf.transpose(Q, perm=transpose_idxs)
            - 2 * X @ tf.transpose(X, perm=transpose_idxs)
        )

        kmm = sigma * tf.exp(-0.5 * dist)

    else:
        raise Exception("Unknown kernel:", kernel_type)

    if debug:
        return kmm, dist, Q, X
    else:
        return kmm


def computeNKmm(X, l, sigma, sigma0, dtype=tf.float32, kernel_type="matern52"):
    """
    X: n x d
    l: 1 x d
    sigma: signal variance
    sigma0: noise variance
    """
    return (
        computeKmm(X, l, sigma, dtype=dtype, kernel_type=kernel_type)
        + tf.eye(tf.shape(X)[0], dtype=dtype) * sigma0
    )


def compute_mean_var_f(
    x, Xsamples, Ysamples, l, sigma, sigma0, NKInv=None, fullcov=False, dtype=tf.float32
):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: m x 1
    Xsamples: m x d
    x: n x d

    return: mean: n x 1
            var : n x 1
    """
    if NKInv is None:
        NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)
        NKInv = chol2inv(NK, dtype=dtype)

    kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype)
    mean = tf.squeeze(kstar @ (NKInv @ Ysamples))

    if fullcov:
        Kx = computeKmm(x, l, sigma, dtype=dtype)
        var = Kx - kstar @ NKInv @ tf.transpose(kstar)
        diag_var = tf.linalg.diag_part(var)
        diag_var = tf.clip_by_value(
            diag_var, clip_value_min=clip_min, clip_value_max=np.infty
        )
        var = tf.linalg.set_diag(var, diag_var)
    else:
        var = sigma - tf.reduce_sum((kstar @ NKInv) * kstar, axis=1)
        var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)

    return mean, var


def compute_var_f(
    x, Xsamples, l, sigma, sigma0, NKInv=None, fullcov=False, dtype=tf.float32
):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: m x 1
    Xsamples: m x d
    x: n x d

    return: var : n x 1
    """
    if NKInv is None:
        NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)
        NKInv = chol2inv(NK, dtype=dtype)

    kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype)

    if fullcov:
        Kx = computeKmm(x, l, sigma, dtype=dtype)
        var = Kx - kstar @ NKInv @ tf.transpose(kstar)
        diag_var = tf.linalg.diag_part(var)
        diag_var = tf.clip_by_value(
            diag_var, clip_value_min=clip_min, clip_value_max=np.infty
        )
        var = tf.linalg.set_diag(var, diag_var)
    else:
        var = sigma - tf.reduce_sum((kstar @ NKInv) * kstar, axis=1)
        var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)

    return var


def computeKmm_np(X, l, sigma, kernel_type="matern52", debug=False):
    n = X.shape[0]
    xdim = X.shape[1]

    l = l.reshape(1, xdim)

    X = X * np.sqrt(l)

    Q = np.tile(np.sum(X * X, axis=1, keepdims=True), reps=(1, n))

    if kernel_type == "matern52":
        sq_dist = Q + Q.T - 2 * X.dot(X.T)
        dist = np.sqrt(np.maximum(sq_dist, 1e-36))

        kmm = (
            sigma
            * (1.0 + np.sqrt(5) * dist + 5.0 / 3.0 * np.square(dist))
            * np.exp(-np.sqrt(5.0) * dist)
        )

    elif kernel_type == "se":
        dist = Q + Q.T - 2 * X.dot(X.T)
        kmm = sigma * np.exp(-0.5 * dist)

    else:
        raise Exception("Unknown kernel:", kernel_type)

    if debug:
        if kernel_type == "matern52":
            return kmm, sq_dist, dist, Q, X
        elif kernel_type == "se":
            return kmm, dist, Q, X
    else:
        return kmm


def computeKnm_np(X, Xbar, l, sigma, kernel_type="matern52"):
    """
    X: n x d
    l: d
    """
    n = np.shape(X)[0]
    m = np.shape(Xbar)[0]
    xdim = np.shape(X)[1]

    l = l.reshape(1, xdim)

    X = X * np.sqrt(l)
    Xbar = Xbar * np.sqrt(l)

    Q = np.tile(np.sum(X * X, axis=1, keepdims=True), reps=(1, m))
    Qbar = np.tile(np.sum(Xbar * Xbar, axis=1, keepdims=True).T, reps=(n, 1))

    if kernel_type == "matern52":
        dist = np.sqrt(np.maximum(Qbar + Q - 2 * X.dot(Xbar.T), 1e-36))

        knm = (
            sigma
            * (1.0 + np.sqrt(5) * dist + 5.0 / 3.0 * np.square(dist))
            * np.exp(-np.sqrt(5.0) * dist)
        )

    elif kernel_type == "se":
        dist = Qbar + Q - 2 * X.dot(Xbar.T)
        knm = sigma * np.exp(-0.5 * dist)

    else:
        raise Exception("Unknown kernel:", kernel_type)

    return knm


def compute_mean_f_np(x, Xsamples, Ysamples, l, sigma, sigma0, kernel_type="matern52"):
    """
    x: n x xdim
    Xsample: m x xdim
    Ysamples: m x 1
    return mean: n x 1

    l: 1 x xdim
    sigma, sigma0: scalar
    """
    m = Xsamples.shape[0]
    xdim = Xsamples.shape[1]
    x = x.reshape(-1, xdim)
    n = x.shape[0]

    Ysamples = Ysamples.reshape(m, 1)

    NKmm = (
        computeKmm_np(Xsamples, l, sigma, kernel_type=kernel_type) + np.eye(m) * sigma0
    )
    invNKmm = np.linalg.inv(NKmm)

    kstar = computeKnm_np(x, Xsamples, l, sigma, kernel_type=kernel_type)
    mean = kstar.dot(invNKmm.dot(Ysamples))

    return mean.reshape(
        n,
    )


def computeNKmm_multiple_data(
    nxs, Xsamples, xs, l, sigma, sigma0, dtype=tf.float32, inverted=False
):
    """
    xs: shape = (nxs,xdim)
    compute covariance matrix of [Xsamples, x] for x in xs
        where Xsamples include noise
              x does not include noise
    return shape (nxs, n_data+1, n_data+1)
        where n_data = tf.shape(Xsamples)[0]
    """
    n_data = tf.shape(Xsamples)[0]
    noise_mat = tf.eye(n_data, dtype=dtype) * sigma0
    noise_mat = tf.pad(noise_mat, [[0, 1], [0, 1]], "CONSTANT")

    ret = []
    for i in range(nxs):
        X_concat = tf.concat([Xsamples, tf.expand_dims(xs[i, :], 0)], axis=0)
        NKmm = computeKmm(X_concat, l, sigma, dtype=dtype) + noise_mat

        if inverted:
            invNKmm = chol2inv(NKmm, dtype=dtype)
            ret.append(invNKmm)
        else:
            ret.append(NKmm)

    return tf.stack(ret)


def compute_mean_f(
    x,
    xdim,
    n_hyp,
    Xsamples,
    Ysamples,
    ls,
    sigmas,
    sigma0s,
    NKInvs,
    dtype=tf.float32,
    kernel_type="matern52",
):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: n x 1
    Xsamples: n x d
    x: 1 x d

    return: mean: n x 1
            var : n x 1
    """

    mean = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i, :], shape=(1, xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        NKInv = NKInvs[i]

        kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype, kernel_type=kernel_type)
        mean = mean + tf.squeeze(kstar @ (NKInv @ Ysamples)) / tf.constant(
            n_hyp, dtype=dtype
        )
    return mean


def precomputeInvK(xdim, l, sigma, sigma0, Xsamples, dtype):  # (1,xdim)
    # l = tf.reshape(ls[i,:], shape=(1,xdim))
    # sigma = sigmas[i]
    # sigma0 = sigma0s[i]

    NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)

    invK = chol2inv(NK, dtype=dtype)
    return invK


def get_information_gain(
    query_x,
    all_X,  # combination of data_X and interest_X
    data_X,
    lengthscale,
    sigma,
    sigma0,
    data_invK,  # for data_X
    invK,  # for [data_X, interest_X]
    dtype=tf.float64,
):
    # compute the information gain
    #    I(y_{query_x}; f_{interest_X}| data)
    # by H(y_{query_x}|data) - H(y_{query_x}| data, f_{interest_X})
    # NOTE: we only compute the reduction in log of variance

    query_var_given_data = compute_var_f(
        query_x,
        data_X,
        lengthscale,
        sigma,
        sigma0,
        data_invK,
        fullcov=False,
        dtype=dtype,
    )

    query_var_given_data_interest = compute_var_f(
        query_x, all_X, lengthscale, sigma, sigma0, invK, fullcov=False, dtype=dtype
    )

    # NOTE: we only compute the reduction in log of variance
    information = tf.log(tf.squeeze(query_var_given_data)) - tf.log(
        tf.squeeze(query_var_given_data_interest)
    )

    return information
