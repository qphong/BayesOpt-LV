import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import functions
import varopt_discrete

func_name = "negative_branin_uniform"

quantile = 0.1

tf.reset_default_graph()
graph = tf.get_default_graph()
dtype = tf.float64

f_info = getattr(functions, func_name)()

func = f_info["function"]

func_tf = f_info["function_tf"]
xmin = f_info["xmin"]
xmax = f_info["xmax"]

xdim = f_info["xdim"]
zdim = f_info["zdim"]
input_dim = xdim + zdim

zvalues_np = f_info["zvalues"]
zprobs_np = f_info["zprobs"]


def ground_truth_func(x, z):
    # x: (1,nx,1,1,xdim)
    # z: (nz,zdim)
    # return (1,n_init_x,1,nz)
    nx = tf.shape(x)[1]
    nz = tf.shape(z)[0]
    x = tf.tile(x, multiples=(1, 1, 1, nz, 1))
    # (1,nx,1,nz,xdim)
    z = tf.reshape(z, shape=(1, 1, 1, nz, zdim))
    z = tf.tile(z, multiples=(1, nx, 1, 1, 1))
    # (1,nx,1,nz,zdim)

    xz = tf.concat([x, z], axis=-1)
    # (1,nx,1,nz,xdim+zdim)

    flatten_xz = tf.reshape(xz, shape=(-1, input_dim))
    # (nx*nz,xdim+zdim)

    vals = func_tf(flatten_xz)
    return tf.reshape(vals, shape=(1, nx, 1, nz))


n_rand_opt_init = 10

ground_truth_varopt = varopt_discrete.VaROpt(
    xdim,
    zdim,
    [xmin, xmax],
    ground_truth_func,
    zvalues_np.shape[0],
    zvalues_np,
    zprobs_np,
    n_rand_opt_init,
    graph,
    name="ground_truth_varopt",
    dtype=dtype,
)

x_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name="x_plc")
ground_truth_quantile_f_val = ground_truth_varopt.find_quantile(x_plc)

if xdim == 1:
    x_np = np.linspace(xmin, xmax, 1000).reshape(-1, xdim)
elif xdim == 2:
    x_np = functions.get_meshgrid(xmin, xmax, 100, xdim)
elif xdim >= 3:
    x_np = functions.get_meshgrid(xmin, xmax, 10, xdim)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ground_truth_quantile_f_val_np = sess.run(
        ground_truth_quantile_f_val,
        feed_dict={
            x_plc: x_np,
            ground_truth_varopt.quantile_plc: quantile,
        },
    )

print(
    "VaR range: {} - {}".format(
        np.min(ground_truth_quantile_f_val_np), np.max(ground_truth_quantile_f_val_np)
    )
)


if xdim == 1:
    plt.plot(x_np, ground_truth_quantile_f_val_np)
elif xdim == 2:
    side = int(np.sqrt(ground_truth_quantile_f_val_np.shape[0]))
    X = np.tile(np.linspace(0.0, 1.0, side).reshape(side, 1), reps=(1, side))
    plt.contourf(X, X.T, ground_truth_quantile_f_val_np.reshape(side, side))
    plt.colorbar()
elif xdim == 3:

    for seed in range(5):
        fig, ax = plt.subplots()

        np.random.seed(seed)
        proj_mat = np.random.rand(xdim, 2)
        proj_mat /= np.sqrt(np.sum(np.square(proj_mat), axis=0, keepdims=True))
        proj_x = x_np[:900].dot(proj_mat)

        cb = ax.contourf(
            proj_x[:, 0].reshape(30, 30),
            proj_x[:, 1].reshape(30, 30),
            ground_truth_quantile_f_val_np[:900].reshape(30, 30),
        )
        plt.colorbar(cb)

elif xdim == 4:

    np.random.seed(2)
    proj_mat = np.random.rand(xdim, 2)
    proj_mat /= np.sqrt(np.sum(np.square(proj_mat), axis=0, keepdims=True))
    proj_x = x_np.dot(proj_mat)

    plt.contourf(
        proj_x[:, 0].reshape(100, 100),
        proj_x[:, 1].reshape(100, 100),
        ground_truth_quantile_f_val_np.reshape(100, 100),
    )
    plt.colorbar()

plt.show()
