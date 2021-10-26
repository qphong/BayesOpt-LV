import numpy as np
import scipy.stats as scst
import scipy.special as scsp
import scipy.optimize as scopt
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
import sys

try:
    import gpflow
except:
    raise Exception("Requires gpflow!")


import utils


def fit_gp(
    X,
    Y,
    noise_var=None,
    train_noise_var=True,
    min_var=1e-4,
    max_var=4.0,
    kernel_type="matern52",
):
    # use gpflow to get the hyperparameters for the function
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph).as_default():
            xdim = X.shape[1]

            if kernel_type == "se":
                kernel = gpflow.kernels.RBF(xdim, ARD=True)
            elif kernel_type == "matern52":
                kernel = gpflow.kernels.Matern52(xdim, ARD=True)
            else:
                raise Exception("Unknown kernel:", kernel_type)

            meanf = gpflow.mean_functions.Constant()

            with gpflow.defer_build():
                m = gpflow.models.GPR(X, Y, kern=kernel, mean_function=meanf)

            if train_noise_var:
                # concentration, rate = 1.1, 1./0.5 (in BoRisk)
                # => shape, scale = 1.1, 0.5
                gamma_shape = 1.1
                gamma_scale = 0.5
                m.likelihood.variance.prior = gpflow.priors.Gamma(
                    gamma_shape, gamma_scale
                )  # shape, scale
                m.likelihood.variance.transform = gpflow.transforms.Logistic(
                    min_var, max_var
                )
                
                # "Initialize likelihood variance at the mode of the prior (from BoRisk)"
                prior_mode = (gamma_shape - 1) * gamma_scale
                m.likelihood.variance.assign(prior_mode)  # 1e-4
            elif noise_var is not None:
                m.likelihood.variance = noise_var
                m.likelihood.variance.trainable = False

            else:
                raise Exception("Require noise variance!")

            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            has_error = False

            try:
                opt.minimize(m)
            except:
                has_error = True

            if has_error:
                return has_error, None
            else:
                gpf_lscale = m.kern.lengthscales.value
                gpf_signal_var = m.kern.variance.value
                lscale = 1.0 / (gpf_lscale * gpf_lscale)
                meanf_const = m.mean_function.c.value
                noise_var = m.likelihood.variance.value

    return has_error, {
        "meanf": meanf_const,
        "signal_var": gpf_signal_var,
        "lengthscale": lscale,
        "noise_var": noise_var,
    }


def get_meshgrid(xmin, xmax, nx, xdim):
    x1d = np.linspace(xmin, xmax, nx)
    vals = [x1d] * xdim
    xds = np.meshgrid(*vals)

    xs = np.concatenate([xd.reshape(-1, 1) for xd in xds], axis=1)
    return xs


def func_gp_prior(xdim, l, sigma, seed, name=""):
    np.random.seed(seed)

    filename = "func_gp_prior_param_seed{}_{}.pkl".format(seed, name)
    n_feats = 1000

    if os.path.isfile(filename):
        with open(filename, "rb") as infile:
            data = pickle.load(infile)
            W = data["W"]
            b = data["b"]
            theta = data["theta"]

    else:
        l = np.ones([1, xdim]) * l
        W = np.random.randn(n_feats, xdim) * np.tile(np.sqrt(l), (n_feats, 1))
        b = 2.0 * np.pi * np.random.rand(n_feats, 1)
        theta = np.random.randn(n_feats, 1)

        with open(filename, "wb") as outfile:
            pickle.dump(
                {"W": W, "b": b, "theta": theta},
                outfile,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def f(x):
        x = np.array(x).reshape(-1, xdim)
        return (
            theta.T.dot(np.sqrt(2.0 * sigma / n_feats)).dot(
                np.cos(W.dot(x.T) + np.tile(b, (1, x.shape[0])))
            )
        ).squeeze()

    return f


def func_gp_prior_tf(xdim, l, sigma, seed, name="", dtype=tf.float64):
    filename = "func_gp_prior_param_seed{}_{}.pkl".format(seed, name)
    n_feats = 1000

    if os.path.isfile(filename):
        with open(filename, "rb") as infile:
            data = pickle.load(infile)
            W = tf.constant(data["W"], dtype=dtype)
            b = tf.constant(data["b"], dtype=dtype)
            theta = tf.constant(data["theta"], dtype=dtype)
    else:
        raise Exception("Require to run func_gp_prior to generate the parameters!")

    def f(x):
        x = tf.reshape(x, shape=(-1, xdim))

        return tf.squeeze(
            tf.cast(tf.sqrt(2.0 * sigma / n_feats), dtype=dtype)
            * tf.linalg.matrix_transpose(theta)
            @ (
                tf.cos(
                    W @ tf.linalg.matrix_transpose(x)
                    + tf.tile(b, multiples=(1, tf.shape(x)[0]))
                )
            )
        )

    return f


def negative_branin_uniform(dtype=tf.float64):
    xdim = 1
    zdim = 1
    input_dim = xdim + zdim

    xmin = 0.0
    xmax = 1.0

    # zmin, zmax only used for continuous z
    zmin = 0.0
    zmax = 1.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1, input_dim)
        x = 15.0 * x - np.array([5.0, 0.0])

        val = (
            -1.0
            / 51.95
            * (
                (
                    x[:, 1]
                    - 5.1 * x[:, 0] ** 2 / (4 * np.pi ** 2)
                    + 5.0 * x[:, 0] / np.pi
                    - 6.0
                )
                ** 2
                + (10.0 - 10.0 / (8.0 * np.pi)) * np.cos(x[:, 0])
                - 44.81
            )
        )
        return val

    def f_tf(x):
        x = tf.reshape(x, shape=(-1, input_dim))
        x = tf.cast(15.0, dtype) * x - tf.cast([5.0, 0.0], dtype)

        val = (
            tf.cast(-1.0, dtype)
            / tf.cast(51.95, dtype)
            * (
                tf.math.pow(
                    x[:, 1]
                    - tf.cast(5.1, dtype)
                    * x[:, 0]
                    * x[:, 0]
                    / tf.cast(4 * np.pi ** 2, dtype)
                    + tf.cast(5.0, dtype) * x[:, 0] / tf.cast(np.pi, dtype)
                    - tf.cast(6.0, dtype),
                    2,
                )
                + (
                    tf.cast(10.0, dtype)
                    - tf.cast(10.0, dtype) / tf.cast(8.0 * np.pi, dtype)
                )
                * tf.cos(x[:, 0])
                - tf.cast(44.81, dtype)
            )
        )
        return val

    mean_tnorm = (zmin + zmax) / 2.0
    std_tnorm = (zmax - zmin) / 8.0
    low_tnorm = mean_tnorm - 2.0 * std_tnorm
    high_tnorm = mean_tnorm + 2.0 * std_tnorm

    truncated_normal = tfp.distributions.TruncatedNormal(
        loc=tf.cast(mean_tnorm, dtype=tf.float64),
        scale=tf.cast(std_tnorm, dtype=tf.float64),
        low=tf.cast(low_tnorm, dtype=tf.float64),
        high=tf.cast(high_tnorm, dtype=tf.float64),
        name="branin_truncated_normal",
    )

    def z_tnorm_generator(n):
        return truncated_normal.sample(sample_shape=(n, zdim))

    def z_lpdf(z):  # (None,zdim)
        return tf.reduce_sum(truncated_normal.log_prob(z), axis=1)

    zmid = (zmin + zmax) / 2.0
    z_values = np.linspace(zmin, zmax, 30).reshape(-1, 1)
    z_probs = np.ones(30) / 30.0
    z_lprobs = np.log(z_probs)

    return {
        "function": f,
        "function_tf": f_tf,
        "name": "negative_branin_uniform",
        "xdim": xdim,
        "zdim": zdim,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "z_generator": z_tnorm_generator,
        "z_lpdf": z_lpdf,
        "zvalues": z_values,
        "zlprobs": z_lprobs,
        "zprobs": z_probs,
        "lengthscale": np.array([12.14689435, 0.3134626]),
        "signal_variance": 1.5294688560240726,
        "likelihood_variance": 1e-2,
        "rand_opt_init_x": xs,
        "max_var_discrete": 0.7324786070977395,
        "max_var_continuous": 0.64118695,
        "max_cvar_discrete": -0.2899622792949111,
    }


def negative_goldstein_uniform(dtype=tf.float64):
    xdim = 1
    zdim = 1
    input_dim = xdim + zdim

    xmin = 0.0
    xmax = 1.0

    # zmin, zmax only used for continuous z
    zmin = 0.0
    zmax = 1.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1, input_dim)
        xb = x * 4.0 - 2.0

        val = -(
            np.log(
                (
                    1
                    + (xb[:, 0] + xb[:, 1] + 1.0) ** 2
                    * (
                        19
                        - 14 * xb[:, 0]
                        + 3 * xb[:, 0] ** 2
                        - 14 * xb[:, 1]
                        + 6 * xb[:, 0] * xb[:, 1]
                        + 3 * xb[:, 1] ** 2
                    )
                )
                * (
                    30
                    + (2 * xb[:, 0] - 3 * xb[:, 1]) ** 2
                    * (
                        18
                        - 32 * xb[:, 0]
                        + 12 * xb[:, 0] ** 2
                        + 48 * xb[:, 1]
                        - 36 * xb[:, 0] * xb[:, 1]
                        + 27 * xb[:, 1] ** 2
                    )
                )
            )
            - 8.693
        )  # / 2.427
        return val

    def f_tf(x):
        x = tf.reshape(x, shape=(-1, input_dim))
        xb = x * tf.cast(4.0, dtype) - tf.cast(2.0, dtype)

        val = -(
            tf.log(
                (
                    tf.cast(1.0, dtype)
                    + tf.math.pow(xb[:, 0] + xb[:, 1] + tf.cast(1.0, dtype), 2)
                    * (
                        tf.cast(19.0, dtype)
                        - tf.cast(14.0, dtype) * xb[:, 0]
                        + tf.cast(3.0, dtype) * xb[:, 0] * xb[:, 0]
                        - tf.cast(14.0, dtype) * xb[:, 1]
                        + tf.cast(6.0, dtype) * xb[:, 0] * xb[:, 1]
                        + tf.cast(3.0, dtype) * xb[:, 1] * xb[:, 1]
                    )
                )
                * (
                    tf.cast(30.0, dtype)
                    + tf.math.pow(
                        tf.cast(2.0, dtype) * xb[:, 0] - tf.cast(3.0, dtype) * xb[:, 1],
                        2,
                    )
                    * (
                        tf.cast(18.0, dtype)
                        - tf.cast(32.0, dtype) * xb[:, 0]
                        + tf.cast(12.0, dtype) * xb[:, 0] * xb[:, 0]
                        + tf.cast(48.0, dtype) * xb[:, 1]
                        - tf.cast(36.0, dtype) * xb[:, 0] * xb[:, 1]
                        + tf.cast(27.0, dtype) * xb[:, 1] * xb[:, 1]
                    )
                )
            )
            - tf.cast(8.693, dtype)
        )  # / tf.cast(2.427, dtype)
        return val

    mean_tnorm = (zmin + zmax) / 2.0
    std_tnorm = (zmax - zmin) / 8.0
    low_tnorm = mean_tnorm - 2.0 * std_tnorm
    high_tnorm = mean_tnorm + 2.0 * std_tnorm

    truncated_normal = tfp.distributions.TruncatedNormal(
        loc=tf.cast(mean_tnorm, dtype=tf.float64),
        scale=tf.cast(std_tnorm, dtype=tf.float64),
        low=tf.cast(low_tnorm, dtype=tf.float64),
        high=tf.cast(high_tnorm, dtype=tf.float64),
        name="branin_truncated_normal",
    )

    def z_tnorm_generator(n):
        return truncated_normal.sample(sample_shape=(n, zdim))

    def z_lpdf(z):  # (None,zdim)
        return tf.reduce_sum(truncated_normal.log_prob(z), axis=1)

    zmid = (zmin + zmax) / 2.0
    z_values = np.linspace(zmin, zmax, 50).reshape(-1, 1)
    z_probs = np.ones(50) / 50.0
    z_lprobs = np.log(z_probs)

    return {
        "function": f,
        "function_tf": f_tf,
        "name": "negative_goldstein_uniform",
        "xdim": xdim,
        "zdim": zdim,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "z_generator": z_tnorm_generator,
        "z_lpdf": z_lpdf,
        "zvalues": z_values,
        "zlprobs": z_lprobs,
        "zprobs": z_probs,
        "lengthscale": np.array([81.1012626, 83.22416009]),
        "signal_variance": 0.02584212360067521,
        "likelihood_variance": 1e-2,
        "rand_opt_init_x": xs,
        "max_var_discrete": 1.7992384381492217,
        "max_var_continuous": 1.50360403,
        "max_cvar_discrete": -2.394406754560626,
    }


def portfolio_computeKmm_np(X, l, sigma):
    n = X.shape[0]
    xdim = X.shape[1]

    l = l.reshape(1, xdim)

    X = X / l

    Q = np.tile(np.sum(X * X, axis=1, keepdims=True), reps=(1, n))
    dist = Q + Q.T - 2 * X.dot(X.T)

    kmm = sigma * np.exp(-0.5 * dist)
    return kmm


def portfolio_computeKnm_np(X, Xbar, l, sigma):
    """
    X: n x d
    l: d
    """
    n = np.shape(X)[0]
    m = np.shape(Xbar)[0]
    xdim = np.shape(X)[1]

    l = l.reshape(1, xdim)

    X = X / l
    Xbar = Xbar / l

    Q = np.tile(np.sum(X * X, axis=1, keepdims=True), reps=(1, m))
    Qbar = np.tile(np.sum(Xbar * Xbar, axis=1, keepdims=True).T, reps=(n, 1))

    dist = Qbar + Q - 2 * X.dot(Xbar.T)
    knm = sigma * np.exp(-0.5 * dist)
    return knm


def portfolio_computeKnm(X, Xbar, l, sigma, dtype=tf.float32):
    """
    X: n x d
    l: d
    """
    n = tf.shape(X)[0]
    m = tf.shape(Xbar)[0]

    X = X / l
    Xbar = Xbar / l

    Q = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), multiples=(1, m))
    Qbar = tf.tile(
        tf.transpose(tf.reduce_sum(tf.square(Xbar), axis=1, keepdims=True)),
        multiples=(n, 1),
    )

    dist = Qbar + Q - 2 * X @ tf.transpose(Xbar)
    knm = sigma * tf.exp(-0.5 * dist)
    return knm


def negative_portfolio_optimization_gaussian(dtype=tf.float64):
    # noise is 1e-2
    # z follows Gaussian
    xdim = 3
    zdim = 2
    input_dim = xdim + zdim

    xmin = 0.0
    xmax = 1.0

    # zmin, zmax only used for continuous z
    zmin = 0.0
    zmax = 1.0

    xs = get_meshgrid(xmin, xmax, 5, xdim)

    with open("portfolio_data/data.pkl", "rb") as readfile:
        data = pickle.load(readfile)
        X = data["X"].astype(np.float64)  # (3000,5)
        Y = data["Y"].astype(np.float64)  # (3000,1)

    with open("portfolio_data/GP_params.pkl", "rb") as readfile:
        params = pickle.load(readfile)
        lengthscales = params["lengthscales"]
        kern_var = params["kern_variance"]
        noise_var = params["noise_variance"]
        mean_constant = params["mean_constant"]
        invKmm = params["invKmm"]

    print(Y)
    print("**PARAMS:", params)

    invKmm_tf = tf.constant(invKmm, dtype=dtype)
    mean_constant_tf = tf.constant(mean_constant, dtype=dtype)
    Y_tf = tf.constant(Y, dtype=dtype)

    def f(x):
        x = x.reshape(-1, input_dim)
        Knm = portfolio_computeKnm_np(x, X, lengthscales, kern_var)
        val = mean_constant + Knm @ invKmm @ (Y - mean_constant)  # posterior mean

        return -val.reshape(x.shape[0])

    def f_tf(x):
        x = tf.reshape(x, shape=(-1, input_dim))
        Knm = portfolio_computeKnm(x, X, lengthscales, kern_var)
        val = mean_constant_tf + Knm @ invKmm_tf @ (Y_tf - mean_constant_tf)

        return -tf.reshape(val, shape=(tf.shape(x)[0],))

    def z_tnorm_generator(n):
        return tf.random.uniform(shape=(n, zdim), minval=0.0, maxval=1.0, dtype=dtype)

    def z_lpdf(z):
        # dummy, not really pdf
        # but returning a constant
        return tf.reduce_sum(tf.ones_like(z, dtype=dtype), axis=1)

    zmid = (zmin + zmax) / 2.0
    z_values = get_meshgrid(zmid - 0.25, zmid + 0.25, 5, zdim)
    z_lprobs = -np.sum((z_values - np.ones(zdim) * zmid) ** 2, axis=1) / 0.15 ** 2
    z_lprobs = np.squeeze(z_lprobs - scsp.logsumexp(z_lprobs))
    z_probs = np.exp(z_lprobs)

    return {
        "function": f,
        "function_tf": f_tf,
        "name": "negative_portfolio_optimization_gaussian",
        "xdim": xdim,
        "zdim": zdim,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "z_generator": z_tnorm_generator,
        "z_lpdf": z_lpdf,
        "zvalues": z_values,
        "zlprobs": z_lprobs,
        "zprobs": z_probs,
        "lengthscale": lengthscales,
        "signal_variance": kern_var,
        "likelihood_variance": 1e-4,
        "rand_opt_init_x": xs,
        "max_var_discrete": 17.835917287050652,
        "max_var_continuous": 17.835917287050652,  # it takes too long to optimize, so we use the discrete case as an approximation
        "max_cvar_X": [0.0, 1.0, 0.08484073],
        "max_cvar_discrete": 21.21,
    }  # at [0., 1., 0.081978]


def negative_rescaled_hartmann6d_51(dtype=tf.float64):
    # xdim = 3
    # range: (0,1) for all dimensions
    # global maximum: -3.86278 at (0.114614, 0.555649, 0.852547)
    xdim = 5
    zdim = 1
    input_dim = xdim + zdim

    xmin = 0.0
    xmax = 1.0

    zmin = 0.0
    zmax = 1.0

    # maximum = 3.13449414
    # minimum = -1.30954062

    xs = get_meshgrid(xmin, xmax, 3, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    A = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    A_tf = tf.constant(A, dtype=dtype)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    alpha_tf = tf.constant(alpha, dtype=dtype)

    P = 1e-4 * np.array(
        [
            [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
        ]
    )
    P_tf = tf.constant(P, dtype=dtype)

    def f(x):
        x = np.tile(x.reshape(-1, 1, input_dim), reps=(1, 4, 1))
        val = (
            2.58 + np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=2)), axis=1)
        ) / 1.94
        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val * 10.0

    def f_tf(x):
        x = tf.tile(tf.reshape(x, shape=(-1, 1, input_dim)), multiples=(1, 4, 1))
        val = (
            tf.constant(2.58, dtype)
            + tf.reduce_sum(
                alpha_tf
                * tf.exp(-tf.reduce_sum(A_tf * (x - P_tf) * (x - P_tf), axis=2)),
                axis=1,
            )
        ) / tf.constant(1.94, dtype)
        return val * tf.cast(10.0, dtype)

    mean_tnorm = (zmin + zmax) / 2.0
    std_tnorm = (zmax - zmin) / 8.0
    low_tnorm = mean_tnorm - 2.0 * std_tnorm
    high_tnorm = mean_tnorm + 2.0 * std_tnorm

    truncated_normal = tfp.distributions.TruncatedNormal(
        loc=tf.cast(mean_tnorm, dtype=tf.float64),
        scale=tf.cast(std_tnorm, dtype=tf.float64),
        low=tf.cast(low_tnorm, dtype=tf.float64),
        high=tf.cast(high_tnorm, dtype=tf.float64),
        name="branin_truncated_normal",
    )

    def z_tnorm_generator(n):
        return truncated_normal.sample(sample_shape=(n, zdim))

    def z_lpdf(z):  # (None,zdim)
        return tf.reduce_sum(truncated_normal.log_prob(z), axis=1)

    zmid = (zmin + zmax) / 2.0
    z_values = get_meshgrid(zmid - 0.2, zmid + 0.2, 15, zdim)
    z_lprobs = -np.sum((z_values - np.ones(zdim) * zmid) ** 2, axis=1) / 0.2 ** 2
    z_lprobs = np.squeeze(z_lprobs - scsp.logsumexp(z_lprobs))
    z_probs = np.exp(z_lprobs)

    return {
        "function": f,
        "function_tf": f_tf,
        "name": "negative_rescaled_hartmann6d_51",
        "xdim": xdim,
        "zdim": zdim,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "z_generator": z_tnorm_generator,
        "z_lpdf": z_lpdf,
        "zvalues": z_values,
        "zlprobs": z_lprobs,
        "zprobs": z_probs,
        "lengthscale": np.array([6.9512, 1.9341, 0.506, 4.2067, 5.0986, 3.5949]),
        "signal_variance": 1.423,
        "likelihood_variance": 1e-2,
        "rand_opt_init_x": xs,
        "max_cvar_discrete": 20.5428,
    } 


def negative_rescaled_hartmann6d_15(dtype=tf.float64):
    # xdim = 3
    # range: (0,1) for all dimensions
    # global maximum: -3.86278 at (0.114614, 0.555649, 0.852547)
    xdim = 1
    zdim = 5
    input_dim = xdim + zdim

    xmin = 0.0
    xmax = 1.0

    zmin = 0.0
    zmax = 1.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    A = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    A_tf = tf.constant(A, dtype=dtype)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    alpha_tf = tf.constant(alpha, dtype=dtype)

    P = 1e-4 * np.array(
        [
            [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
        ]
    )
    P_tf = tf.constant(P, dtype=dtype)

    def f(x):
        x = np.tile(x.reshape(-1, 1, input_dim), reps=(1, 4, 1))
        val = (
            2.58 + np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=2)), axis=1)
        ) / 1.94
        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val * 10.0

    def f_tf(x):
        x = tf.tile(tf.reshape(x, shape=(-1, 1, input_dim)), multiples=(1, 4, 1))
        val = (
            tf.constant(2.58, dtype)
            + tf.reduce_sum(
                alpha_tf
                * tf.exp(-tf.reduce_sum(A_tf * (x - P_tf) * (x - P_tf), axis=2)),
                axis=1,
            )
        ) / tf.constant(1.94, dtype)
        return val * tf.cast(10.0, dtype)

    mean_tnorm = (zmin + zmax) / 2.0
    std_tnorm = (zmax - zmin) / 8.0
    low_tnorm = mean_tnorm - 2.0 * std_tnorm
    high_tnorm = mean_tnorm + 2.0 * std_tnorm

    truncated_normal = tfp.distributions.TruncatedNormal(
        loc=tf.cast(mean_tnorm, dtype=tf.float64),
        scale=tf.cast(std_tnorm, dtype=tf.float64),
        low=tf.cast(low_tnorm, dtype=tf.float64),
        high=tf.cast(high_tnorm, dtype=tf.float64),
        name="branin_truncated_normal",
    )

    def z_tnorm_generator(n):
        return truncated_normal.sample(sample_shape=(n, zdim))

    def z_lpdf(z):  # (None,zdim)
        return tf.reduce_sum(truncated_normal.log_prob(z), axis=1)

    zmid = (zmin + zmax) / 2.0
    z_values = get_meshgrid(zmid - 0.2, zmid + 0.2, 3, zdim)
    z_lprobs = -np.sum((z_values - np.ones(zdim) * zmid) ** 2, axis=1) / 0.2 ** 2
    z_lprobs = np.squeeze(z_lprobs - scsp.logsumexp(z_lprobs))
    z_probs = np.exp(z_lprobs)

    return {
        "function": f,
        "function_tf": f_tf,
        "name": "negative_rescaled_hartmann6d_15",
        "xdim": xdim,
        "zdim": zdim,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "z_generator": z_tnorm_generator,
        "z_lpdf": z_lpdf,
        "zvalues": z_values,
        "zlprobs": z_lprobs,
        "zprobs": z_probs,
        "lengthscale": np.array([6.9512, 1.9341, 0.506, 4.2067, 5.0986, 3.5949]),
        "signal_variance": 1.423,
        "likelihood_variance": 1e-4,
        "rand_opt_init_x": xs,
        "max_cvar_discrete": 14.1203,  # near [0.2544893]
    }  # haven't optimize yet


def yacht_hydrodynamics(dtype=tf.float64):
    filename = "yacht_data/gp_hyperparameters.pkl"

    with open(filename, "rb") as readfile:
        yacht = pickle.load(readfile)
        X = yacht["X"]
        Y = yacht["Y"]
        gp_hyper = yacht["gp_hyper"]

    NK = utils.computeNKmm(
        X,
        gp_hyper["lengthscale"],
        gp_hyper["signal_var"],
        gp_hyper["noise_var"],
        dtype=dtype,
        kernel_type="se",
    )
    NKInv = utils.chol2inv(NK, dtype=dtype)
    NKInvs = tf.expand_dims(NKInv, axis=0)

    input_dim = X.shape[1]
    zdim = 1
    xdim = input_dim - zdim

    xmin = 0.0
    xmax = 1.0
    zmin = 0.0
    zmax = 1.0

    xs = get_meshgrid(xmin, xmax, 4, xdim)

    def f(x):
        x = x.reshape(-1, input_dim)
        mean_f = (
            utils.compute_mean_f_np(
                x,
                X,
                Y - gp_hyper["meanf"],
                gp_hyper["lengthscale"],
                gp_hyper["signal_var"],
                gp_hyper["noise_var"],
                kernel_type="se",
            )
            + gp_hyper["meanf"]
        )
        return -mean_f.reshape(-1, 1)

    def f_tf(x):
        x = tf.reshape(x, (-1, input_dim))
        mean_f = (
            utils.compute_mean_f(
                x,
                input_dim,
                1,
                X,
                Y - gp_hyper["meanf"],
                gp_hyper["lengthscale"].reshape(1, input_dim),
                gp_hyper["signal_var"].reshape(1, 1),
                gp_hyper["noise_var"].reshape(1, 1),
                NKInvs,
                dtype=dtype,
                kernel_type="se",
            )
            + gp_hyper["meanf"]
        )
        return -mean_f

    zmid = 0.0
    z_values = np.linspace(zmin, zmax, 15).reshape(-1, 1)
    z_probs = np.ones(z_values.shape[0]) / z_values.shape[0]
    z_lprobs = np.log(z_probs)

    return {
        "function": f,
        "function_tf": f_tf,
        "name": "yacht_hydrodynamics",
        "xdim": xdim,
        "zdim": zdim,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "zvalues": z_values,
        "zlprobs": z_lprobs,
        "zprobs": z_probs,
        "lengthscale": gp_hyper["lengthscale"],
        "signal_variance": gp_hyper["signal_var"],
        "likelihood_variance": 0.0001,  # gp_hyper["noise_var"],
        "rand_opt_init_x": xs,
        "max_cvar_discrete": -1.009,  # at [0.35523405 1.         0.         0.         0.85907464], alpha=0.3
    }