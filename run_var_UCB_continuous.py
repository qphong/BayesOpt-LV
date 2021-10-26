import sys
import os
import argparse
import pickle

epsilon = 1e-12

parser = argparse.ArgumentParser(description="Bayesian Optimization for Value at Risk.")
parser.add_argument(
    "-g",
    "--gpu",
    help="gpu device index for tensorflow",
    required=False,
    type=str,
    default="0",
)
parser.add_argument(
    "-q",
    "--numqueries",
    help="number/budget of queries",
    required=False,
    type=int,
    default=1,
)
parser.add_argument(
    "-r",
    "--numruns",
    help="number of random experiments",
    required=False,
    type=int,
    default=2,
)
parser.add_argument(
    "--ntrain",
    help="number of optimizing iterations",
    required=False,
    type=int,
    default=100,
)
parser.add_argument(
    "--n_init_data",
    help="number of initial observations",
    required=False,
    type=int,
    default=2,
)
parser.add_argument(
    "--n_rand_opt_init",
    help="number of random initializers for optimization",
    required=False,
    type=int,
    default=20,
)
parser.add_argument(
    "--function",
    help="function to optimize",
    required=False,
    type=str,
    default="robot_pushing_optimization",
)
parser.add_argument(
    "--quantile", help="quantile", required=False, type=float, default=0.2
)

parser.add_argument(
    "--width", help="neighbor width", required=False, type=float, default=0.1
)
parser.add_argument(
    "--nzsample", help="number of samples of z", required=False, type=int, default=50
)
parser.add_argument(
    "--nxsample", help="number of samples of x", required=False, type=int, default=50
)
parser.add_argument(
    "--ntrainsur",
    help="number of optimizing iterations to optimize the surrogate",
    required=False,
    type=int,
    default=2000,
)

parser.add_argument(
    "--minvar", help="minimum noise variance", required=False, type=float, default=1e-4
)
parser.add_argument(
    "--maxvar", help="maximum noise variance", required=False, type=float, default=4.0
)
parser.add_argument(
    "--n_iter_fitgp",
    help="fit the gp after n_iter_fitgp, if n_iter_fitgp = 0, never fit gp to the data (i.e., use pre-trained hyperparameters)",
    required=False,
    type=int,
    default=3,
)
parser.add_argument(
    "--shuffletie",
    help="if shuffletie==1, when there are many values of z with the maximum value, we break tie by randomly selecting a value of z with the maximum value",
    required=False,
    type=int,
    default=0,
)
parser.add_argument(
    "-t",
    "--dtype",
    help="type of float: float32 or float64",
    required=False,
    type=str,
    default="float64",
)


args = parser.parse_args()


# print all arguments
print("================================")
for arg in vars(args):
    print(arg, getattr(args, arg))
print("================================")

gpu_device_id = args.gpu

folder = args.function
if not os.path.exists(folder):
    os.makedirs(folder)

folder = "{}/continuous".format(folder)
if not os.path.exists(folder):
    os.makedirs(folder)


nquery = args.numqueries
nrun = args.numruns
ntrain = args.ntrain
n_init_data = args.n_init_data
n_iter_fitgp = args.n_iter_fitgp
shuffletie = args.shuffletie

ntrainsur = args.ntrainsur
nxsample = args.nxsample
nzsample = args.nzsample
width = args.width

min_var = args.minvar
max_var = args.maxvar

func_name = args.function

print("nrun: {}".format(nrun))
print("nquery: {}".format(nquery))
print("n_init_data: {}".format(n_init_data))
print("n_iter_fitgp: {}".format(n_iter_fitgp))
print("Function: {}".format(func_name))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device_id


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy as sp
import time
import scipy.stats as spst


import matplotlib.pyplot as plt


import utils
import functions
import varopt


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = False
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.95

if args.dtype == "float32":
    dtype = tf.float32
    nptype = np.float32
elif args.dtype == "float64":
    dtype = tf.float64
    nptype = np.float64
else:
    raise Exception("Unknown dtype: {}".format(args.dtype))


tf.reset_default_graph()
graph = tf.get_default_graph()


f_info = getattr(functions, func_name)()
"""
{
    'function'
    'name'
    'xdim'
    'zdim'
    'xmin'
    'xmax'
    'is_z_discrete'
    if z is continuous:
        'zmin'
        'zmax'
    elif z is discrete:
        'zvalues'
        'zprobs'

    # kernel hyperparameters
        'lengthscale'
        'signal_variance'

    'likelihood_variance'

    'rand_opt_init_x'
        if 'rand_opt_init_x' is None: initialize x randomly
}
"""

print("Information of function:")
for k in f_info:
    if k != "rand_opt_init_x":
        print("{}: {}".format(k, f_info[k]))
    else:
        print("init_x.shape: {}".format(f_info["rand_opt_init_x"].shape))

func = f_info["function"]
func_tf = f_info["function_tf"]
xmin = f_info["xmin"]
xmax = f_info["xmax"]

xdim = f_info["xdim"]
zdim = f_info["zdim"]


zmin = f_info["zmin"]
zmax = f_info["zmax"]

z_generator = f_info["z_generator"]
z_lpdf = f_info["z_lpdf"]

lengthscale_np = f_info["lengthscale"]
signal_variance_np = f_info["signal_variance"]
generate_obs_noise_var_np = f_info["likelihood_variance"]
likelihood_variance_np = f_info["likelihood_variance"]

if f_info["rand_opt_init_x"] is None:
    n_rand_opt_init = args.n_rand_opt_init
else:
    n_rand_opt_init = f_info["rand_opt_init_x"].shape[0]
    rand_opt_init_x_np = f_info["rand_opt_init_x"]

random_seed = 0
print("Random seed:", random_seed)

quantile = args.quantile

with graph.as_default():

    X_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name="X_plc")
    Z_plc = tf.placeholder(dtype=dtype, shape=(None, zdim), name="Z_plc")
    input_dim = xdim + zdim
    inputs = tf.concat([X_plc, Z_plc], axis=-1)
    # (None,xdim+zdim)
    Y_plc = tf.placeholder(dtype=dtype, shape=(None, 1), name="Y_plc")

    beta_plc = tf.placeholder(dtype=dtype, shape=(), name="beta_plc")

    lengthscale = tf.get_variable(
        dtype=dtype, shape=(1, xdim + zdim), name="lengthscale"
    )
    signal_variance = tf.get_variable(dtype=dtype, shape=(), name="signal_variance")
    likelihood_variance = tf.get_variable(
        dtype=dtype, shape=(), name="likelihood_variance"
    )

    NKmm = utils.computeNKmm(
        inputs, lengthscale, signal_variance, likelihood_variance, dtype
    )

    invK = utils.precomputeInvK(
        input_dim, lengthscale, signal_variance, likelihood_variance, inputs, dtype
    )
    # (n_observations, n_observations)

    invK_plc = tf.placeholder(dtype=dtype, shape=(None, None), name="invK_plc")

    def get_bound(x, z, beta, type="lower"):
        # x: (1,nx,nxsample,1,xdim)
        # z: (nz,zdim)
        # return (1,n_init_x,1,nz)
        nx = tf.shape(x)[1]
        nxsample = tf.shape(x)[2]
        nz = tf.shape(z)[0]
        x = tf.tile(x, multiples=(1, 1, 1, nz, 1))
        # (1,nx,nxsample,nz,xdim)
        z = tf.reshape(z, shape=(1, 1, 1, nz, zdim))
        z = tf.tile(z, multiples=(1, nx, nxsample, 1, 1))
        # (1,nx,nxsample,nz,zdim)

        xz = tf.concat([x, z], axis=-1)
        # (1,nx,nxsample,nz,xdim+zdim)

        flatten_xz = tf.reshape(xz, shape=(-1, input_dim))
        # (nx*nxsample*nz,xdim+zdim)

        mean_f, var_f = utils.compute_mean_var_f(
            flatten_xz,
            inputs,
            Y_plc,
            lengthscale,
            signal_variance,
            likelihood_variance,
            invK_plc,
            fullcov=False,
            dtype=dtype,
        )
        # mean_f: (nx*nz,1), var_f: (nx*nz,1)
        std_f = tf.sqrt(var_f)

        if type == "upper":
            bound = mean_f + beta_plc * std_f
            # (nx*nz,1)
        elif type == "lower":
            bound = mean_f - beta_plc * std_f
            # (nx*nz,1)
        elif type == "mean":
            bound = mean_f
        else:
            raise Exception("Unknown bound!")

        bound = tf.reshape(bound, shape=(1, nx, nxsample, nz))
        return bound

    # maximizer upper_varopt to find the query x
    upper_varopt = varopt.VaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        [zmin, zmax],
        n_func=1,
        f=lambda x, z: get_bound(x, z, beta_plc, type="upper"),
        z_generator=z_generator,
        n_init_x=n_rand_opt_init,
        graph=graph,
        surrogate_config={
            "layer_sizes": [30, 30, 1],
            "activations": ["sigmoid", "sigmoid", "linear"],
        },
        name="upper_varopt",
        dtype=dtype,
    )

    lower_varopt = varopt.VaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        [zmin, zmax],
        n_func=1,
        f=lambda x, z: get_bound(x, z, beta_plc, type="lower"),
        z_generator=z_generator,
        n_init_x=n_rand_opt_init,
        graph=graph,
        surrogate_config={
            "layer_sizes": [30, 30, 1],
            "activations": ["sigmoid", "sigmoid", "linear"],
        },
        name="lower_varopt",
        dtype=dtype,
    )
    # find lower bound of at the query x
    query_x_plc = tf.placeholder(dtype=dtype, shape=(1, 1, xdim), name="query_x_plc")

    (
        lower_quantile_f_val_at_queryx,
        loss_lower_quantile_f_val_at_queryx,
        train_lower_quantile_f_val_at_queryx,
    ) = lower_varopt.find_quantile(
        query_x_plc, n_func=1, n_x=1, name="find_quantile_lower_at_query_x"
    )

    (
        upper_quantile_f_val_at_queryx,
        loss_upper_quantile_f_val_at_queryx,
        train_upper_quantile_f_val_at_queryx,
    ) = upper_varopt.find_quantile(
        query_x_plc, n_func=1, n_x=1, name="find_quantile_upper_at_query_x"
    )

    # lower_quantile_z_at_obs, lower_quantile_f_val_at_obs = lower_varopt.find_quantile(X_plc)

    # estimate the maximizer
    #   by maximizing mean function
    mean_varopt = varopt.VaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        [zmin, zmax],
        n_func=1,
        f=lambda x, z: get_bound(x, z, beta_plc, type="mean"),
        z_generator=z_generator,
        n_init_x=n_rand_opt_init,
        graph=graph,
        surrogate_config={
            "layer_sizes": [30, 30, 1],
            "activations": ["sigmoid", "sigmoid", "linear"],
        },
        name="mean_varopt",
        dtype=dtype,
    )

    #   (2) by choosing observed input with max lower bound
    # _, lower_quantile_f_val_at_X = lower_varopt.find_quantile(X_plc)

    # for querying z
    #    optimize an objective function:
    # query_x_plc

    n_opt_init_z = 100

    # NOTE: whenever optimizing query_z_tf
    #    need to load query_z_tf with random values
    query_z_init = z_generator(n_opt_init_z)

    query_z_tf = tf.get_variable(
        initializer=np.zeros([n_opt_init_z, zdim], dtype=np.float64),
        dtype=dtype,
        name="query_z_tf",
    )
    upper_at_query = tf.squeeze(
        get_bound(
            tf.reshape(query_x_plc, shape=(1, tf.shape(query_x_plc)[0], 1, 1, xdim)),
            query_z_tf,
            beta_plc,
            type="upper",
        )
    )
    # (nz,)
    lower_at_query = tf.squeeze(
        get_bound(
            tf.reshape(query_x_plc, shape=(1, tf.shape(query_x_plc)[0], 1, 1, xdim)),
            query_z_tf,
            beta_plc,
            type="lower",
        )
    )
    # (nz,)

    z_logprobs = z_lpdf(query_z_tf)
    # (nz,)

    # loss_of_z_selection = tf.nn.relu(tf.squeeze(upper_quantile_f_val_at_queryx) - upper_at_query)\
    #     + tf.nn.relu(lower_at_query - tf.squeeze(lower_quantile_f_val_at_queryx))
    # # (nz,)

    upper_diff = tf.squeeze(upper_quantile_f_val_at_queryx) - upper_at_query
    lower_diff = lower_at_query - tf.squeeze(lower_quantile_f_val_at_queryx)
    bound_diff = upper_at_query - lower_at_query
    # logprobs is zero if constraints are not satisfied:
    #   upper < upper_at_x or lower > lower_at_x

    # eff_z_probs = tf.where(tf.nn.relu(-upper_diff) * tf.nn.relu(-lower_diff) >= -tf.cast(epsilon,dtype=dtype),
    #     tf.exp(z_logprobs),
    #     tf.zeros_like(z_logprobs, dtype=dtype))

    cond = tf.cast(
        tf.cast(upper_diff <= 0.0, dtype=tf.int32)
        * tf.cast(lower_diff <= 0.0, dtype=tf.int32),
        dtype=tf.bool,
    )

    eff_z_probs = tf.where(
        cond, tf.exp(z_logprobs), tf.zeros_like(z_logprobs, dtype=dtype)
    )

    loss_of_z_selection = tf.nn.relu(upper_diff) + tf.nn.relu(lower_diff) - eff_z_probs

    train_z_selection = tf.train.AdamOptimizer().minimize(
        tf.reduce_mean(loss_of_z_selection), var_list=[query_z_tf]
    )

    if shuffletie:
        print("Shuffling when tie exists in the loss of z!")
        shuffled_idxs = tf.random.shuffle(tf.range(tf.shape(loss_of_z_selection)[0]))
        shuffled_loss_of_z = tf.gather(loss_of_z_selection, indices=shuffled_idxs)
        shuffled_min_z_selection_loss_idx = tf.math.argmax(-shuffled_loss_of_z)
        min_z_selection_loss_idx = shuffled_idxs[shuffled_min_z_selection_loss_idx]
    else:
        print("No shuffling when tie exists in the loss of z")
        min_z_selection_loss_idx = tf.math.argmax(-loss_of_z_selection)

    selected_z = tf.gather(query_z_tf, indices=min_z_selection_loss_idx, axis=0)
    selected_z_loss = tf.gather(
        loss_of_z_selection, indices=min_z_selection_loss_idx, axis=0
    )
    selected_upper_diff = tf.gather(
        upper_diff, indices=min_z_selection_loss_idx, axis=0
    )
    selected_lower_diff = tf.gather(
        lower_diff, indices=min_z_selection_loss_idx, axis=0
    )
    selected_bound_diff = tf.gather(
        bound_diff, indices=min_z_selection_loss_idx, axis=0
    )
    selected_z_logprob = tf.gather(z_logprobs, indices=min_z_selection_loss_idx, axis=0)

    # find the ground truth function value for evaluating the regret
    def ground_truth_func(x, z):
        # x: (1,n_init_x,n_x_sample,1,xdim)
        # z: (nz,zdim)
        # return (1,n_init_x,1,nz)
        n_init_x = tf.shape(x)[1]
        n_x_sample = tf.shape(x)[2]
        nz = tf.shape(z)[0]
        x = tf.tile(x, multiples=(1, 1, 1, nz, 1))
        # (1,n_init_x,n_x_sample,nz,xdim)
        z = tf.reshape(z, shape=(1, 1, 1, nz, zdim))
        z = tf.tile(z, multiples=(1, n_init_x, n_x_sample, 1, 1))
        # (1,n_init_x,n_x_sample,nz,zdim)

        xz = tf.concat([x, z], axis=-1)
        # (1,n_init_x,1,nz,xdim+zdim)

        flatten_xz = tf.reshape(xz, shape=(-1, input_dim))
        # (n_init_x*nz,xdim+zdim)

        vals = func_tf(flatten_xz)
        return tf.reshape(vals, shape=(1, n_init_x, n_x_sample, nz))

    ground_truth_varopt = varopt.VaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        [zmin, zmax],
        n_func=1,
        f=ground_truth_func,
        z_generator=z_generator,
        n_init_x=n_rand_opt_init,
        graph=graph,
        surrogate_config={
            "layer_sizes": [50, 50, 1],
            "activations": ["sigmoid", "sigmoid", "linear"],
        },
        name="groundtruth_varopt",
        dtype=dtype,
    )

    # compute the ground truth quantile at estimated maximizer for computing the regret
    # find lower bound of at the query x
    est_maximizer_plc = tf.placeholder(
        dtype=dtype, shape=(1, 1, xdim), name="est_maximizer_plc"
    )
    (
        ground_truth_quantile_f_val_at_est_max,
        loss_ground_truth_quantile_f_val_at_est_max,
        train_ground_truth_quantile_f_val_at_est_max,
    ) = ground_truth_varopt.find_quantile(
        est_maximizer_plc, n_func=1, n_x=1, name="groundtruth"
    )


np.random.seed(random_seed)

# general initial observations for all random runs:
init_X_np = np.random.rand(nrun, n_init_data, xdim) * (xmax - xmin) + xmin
init_Z_np = np.random.rand(nrun, n_init_data, zdim) * (zmax - zmin) + zmin

with open(
    "{}/init_observations_seed{}.pkl".format(folder, random_seed), "wb"
) as outfile:
    pickle.dump(
        {"init_X_np": init_X_np, "init_Z_np": init_Z_np},
        outfile,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

with graph.as_default():

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print("*********************")
        print("rand_opt_init_x_np.shape = ", rand_opt_init_x_np.shape)

        # opt_x_np, opt_var_np = ground_truth_varopt.maximize_in_session(
        #             sess,
        #             n_x_train = ntrain,
        #             n_z_train = ntrainsur,
        #             feed_dict = {
        #                 ground_truth_varopt.quantile_plc: quantile,
        #                 ground_truth_varopt.neighbor_center_plc: np.expand_dims(rand_opt_init_x_np, axis=0), #i.e., n_func = 1
        #                 ground_truth_varopt.neighbor_width_plc: width,
        #                 ground_truth_varopt.n_z_sample_plc: nzsample,
        #                 ground_truth_varopt.n_x_sample_plc: nxsample
        #             },
        #             verbose = 100
        #         )

        # print("Optimal X: {}, VaR: {}".format(opt_x_np, opt_var_np))

        opt_var_np = f_info["max_var_continuous"]
        print("Optimal VaR: {}".format(opt_var_np))

        all_regrets_by_mean = np.zeros([nrun, nquery])
        all_vars_at_est_maximizer_by_mean = np.zeros([nrun, nquery])
        all_estimated_maximizers = np.zeros([nrun, nquery, xdim])
        all_estimated_maximizers_by_lower = np.zeros([nrun, nquery, xdim])
        all_estimated_max_var_by_mean = np.zeros([nrun, nquery])
        all_estimated_max_var_by_lower = np.zeros([nrun, nquery])

        for run_idx in range(nrun):
            print("{}. RANDOM RUN".format(run_idx))

            # Generate initial observations
            X_np = init_X_np[run_idx, :, :]
            Z_np = init_Z_np[run_idx, :]

            input_np = np.concatenate([X_np, Z_np], axis=-1)
            Y_np = func(input_np).reshape(-1, 1) + np.sqrt(
                generate_obs_noise_var_np
            ) * np.random.randn(input_np.shape[0]).reshape(-1, 1)

            sess.run(tf.global_variables_initializer())
            # if n_iter_fitgp == 0: meanf_const is kept at 0. always
            # it is updated accordingly
            meanf_const = 0.0

            for query_idx in range(nquery):
                print("")
                print("{}.{}. QUERYING".format(run_idx, query_idx))
                print("Controlled variable:")
                print(X_np)
                print("Environment variable:")
                print(Z_np)
                beta_np = 2.0 * np.log((query_idx + 1) ** 2 * np.pi ** 2 / 6.0 / 0.1)
                print("Observation:")
                print(Y_np)

                # NOTE: to generate Y_np we need generate_obs_noise_var_np for synthetic function, this is unknown to the BO algorithm. In BO, the noise variance is learned from the data

                # NOTE: we do not scale Y_np, because it may cause the pre-trained GP model for the function incorrect!
                #    it also causes the noise min_var incorrectly estimated
                #    i.e., scaling Y_np changes the noise var
                # scaled_Y_np = Y_np / (np.max(Y_np) - np.min(Y_np))

                if n_iter_fitgp > 0 and query_idx % n_iter_fitgp == 0:
                    print("Fit GP to observations")
                    has_error, gp_hyperparameters = functions.fit_gp(
                        input_np,
                        Y_np,
                        noise_var=likelihood_variance_np,
                        train_noise_var=True,
                        min_var=min_var,
                        max_var=max_var,
                    )

                    if has_error:
                        print("  Skip due to numerical error!")
                    else:
                        print(
                            "  Learned GP hyperparameters: {}".format(
                                gp_hyperparameters
                            )
                        )
                        meanf_const = gp_hyperparameters["meanf"]
                        signal_variance_np = gp_hyperparameters["signal_var"]
                        lengthscale_np = gp_hyperparameters["lengthscale"]
                        likelihood_variance_np = gp_hyperparameters["noise_var"]

                shifted_Y_np = Y_np - meanf_const

                lengthscale.load(lengthscale_np.reshape(1, input_dim), sess)
                signal_variance.load(np.squeeze(signal_variance_np), sess)
                likelihood_variance.load(np.squeeze(likelihood_variance_np), sess)

                feed_dict = {
                    X_plc: X_np,
                    Z_plc: Z_np,
                    Y_plc: shifted_Y_np,
                    beta_plc: beta_np,
                    mean_varopt.quantile_plc: quantile,
                    mean_varopt.neighbor_center_plc: np.expand_dims(
                        rand_opt_init_x_np, axis=0
                    ),  # i.e., n_func = 1
                    mean_varopt.neighbor_width_plc: width,
                    mean_varopt.n_z_sample_plc: nzsample,
                    mean_varopt.n_x_sample_plc: nxsample,
                    upper_varopt.quantile_plc: quantile,
                    upper_varopt.neighbor_center_plc: np.expand_dims(
                        rand_opt_init_x_np, axis=0
                    ),  # i.e., n_func = 1
                    upper_varopt.neighbor_width_plc: width,
                    upper_varopt.n_z_sample_plc: nzsample,
                    upper_varopt.n_x_sample_plc: nxsample,
                    lower_varopt.quantile_plc: quantile,
                    lower_varopt.neighbor_center_plc: np.expand_dims(
                        rand_opt_init_x_np, axis=0
                    ),  # i.e., n_func = 1
                    lower_varopt.neighbor_width_plc: width,
                    lower_varopt.n_z_sample_plc: nzsample,
                    lower_varopt.n_x_sample_plc: nxsample,
                    ground_truth_varopt.quantile_plc: quantile,
                    ground_truth_varopt.neighbor_center_plc: np.expand_dims(
                        rand_opt_init_x_np, axis=0
                    ),  # i.e., n_func = 1
                    ground_truth_varopt.neighbor_width_plc: width,
                    ground_truth_varopt.n_z_sample_plc: nzsample,
                    ground_truth_varopt.n_x_sample_plc: nxsample,
                }

                invK_np = sess.run(invK, feed_dict=feed_dict)

                feed_dict[invK_plc] = invK_np

                print("")
                print("Estimating maximizer by maximize the VaR of posterior mean.")
                # max_x_mean_np, max_quantile_f_mean_np = mean_varopt.maximize_in_session(
                #     sess,
                #     n_x_train = ntrain,
                #     n_z_train = ntrainsur,
                #     feed_dict = feed_dict,
                #     verbose = 100
                # )

                max_x_mean_np, max_quantile_f_mean_np = mean_varopt.find_max_in_set(
                    sess, X_np, feed_dict, ntrain=1000
                )
                (
                    max_x_mean_np_by_lower,
                    max_quantile_f_mean_np_by_lower,
                ) = lower_varopt.find_max_in_set(sess, X_np, feed_dict, ntrain=1000)

                all_estimated_maximizers[
                    run_idx, query_idx, :
                ] = max_x_mean_np.squeeze()

                all_estimated_maximizers_by_lower[
                    run_idx, query_idx, :
                ] = max_x_mean_np_by_lower.squeeze()

                all_estimated_max_var_by_mean[run_idx, query_idx] = np.squeeze(
                    max_quantile_f_mean_np
                )
                all_estimated_max_var_by_lower[run_idx, query_idx] = np.squeeze(
                    max_quantile_f_mean_np_by_lower
                )

                print(
                    "Estimated maximizer at {} VaR {}".format(
                        max_x_mean_np, max_quantile_f_mean_np
                    )
                )
                print(
                    "Estimated maximizer by lower at {} VaR {}".format(
                        max_x_mean_np_by_lower, max_quantile_f_mean_np_by_lower
                    )
                )
                sys.stdout.flush()

                # computing the regret
                feed_dict[est_maximizer_plc] = max_x_mean_np.reshape(1, 1, xdim)

                for _ in range(2000):
                    sess.run(train_ground_truth_quantile_f_val_at_est_max, feed_dict)

                ground_truth_quantile_f_val_at_est_max_np = sess.run(
                    ground_truth_quantile_f_val_at_est_max, feed_dict
                )
                regret = (
                    opt_var_np - ground_truth_quantile_f_val_at_est_max_np.squeeze()
                )
                print("Regret by maximizing mean: ", regret)
                print(
                    "Groundtruth VaR at the query x by max mean: ",
                    ground_truth_quantile_f_val_at_est_max_np.squeeze(),
                )
                all_regrets_by_mean[run_idx, query_idx] = regret
                all_vars_at_est_maximizer_by_mean[
                    run_idx, query_idx
                ] = ground_truth_quantile_f_val_at_est_max_np.squeeze()

                # Find query x
                print("")
                print("Finding query x by maximizing upper bound of VaR.")
                (
                    query_x_np,
                    upper_quantile_f_at_queryx_np,
                ) = upper_varopt.maximize_in_session(
                    sess,
                    n_x_train=ntrain,
                    n_z_train=ntrainsur,
                    feed_dict=feed_dict,
                    verbose=100,
                )
                upper_quantile_f_at_queryx_np = np.squeeze(
                    upper_quantile_f_at_queryx_np
                )

                print("Query x: {}".format(query_x_np))
                print(
                    "  At query, upper bound of function value: {:.6f}".format(
                        upper_quantile_f_at_queryx_np
                    )
                )
                sys.stdout.flush()

                feed_dict[query_x_plc] = query_x_np.reshape(1, 1, xdim)

                for _ in range(ntrainsur):
                    sess.run(train_lower_quantile_f_val_at_queryx, feed_dict)

                lower_quantile_f_val_at_queryx_np = sess.run(
                    lower_quantile_f_val_at_queryx, feed_dict
                )
                lower_quantile_f_val_at_queryx_np = np.squeeze(
                    lower_quantile_f_val_at_queryx_np
                )
                print(
                    "  At query, lower bound of function value: {:.6f}".format(
                        lower_quantile_f_val_at_queryx_np
                    )
                )

                query_z_tf.load(sess.run(query_z_init), sess)

                for _ in range(2000):
                    sess.run(train_z_selection, feed_dict)

                (
                    query_z_np,
                    selected_z_loss_np,
                    selected_upper_diff_np,
                    selected_lower_diff_np,
                    selected_bound_diff_np,
                    selected_z_logprob_np,
                ) = sess.run(
                    [
                        selected_z,
                        selected_z_loss,
                        selected_upper_diff,
                        selected_lower_diff,
                        selected_bound_diff,
                        selected_z_logprob,
                    ],
                    feed_dict,
                )

                print(
                    "Query z: {} with upper diff {} lower diff {} bound diff {} lprob {}, loss {}".format(
                        query_z_np,
                        selected_upper_diff_np,
                        selected_lower_diff_np,
                        selected_bound_diff_np,
                        selected_z_logprob_np,
                        selected_z_loss_np,
                    )
                )

                X_np = np.concatenate([X_np, query_x_np.reshape(1, xdim)], axis=0)
                Z_np = np.concatenate([Z_np, query_z_np.reshape(1, zdim)], axis=0)

                input_np = np.concatenate([X_np, Z_np], axis=-1)
                query_np = np.concatenate(
                    [query_x_np.reshape(1, xdim), query_z_np.reshape(1, zdim)], axis=-1
                )
                query_obs_np = func(query_np).reshape(-1, 1) + np.sqrt(
                    generate_obs_noise_var_np
                ) * np.random.randn(query_np.shape[0]).reshape(-1, 1)

                Y_np = np.concatenate([Y_np, query_obs_np], axis=0)

            with open(
                "{}/all_observations_quantile{}.pkl".format(folder, quantile), "wb"
            ) as outfile:
                pickle.dump(
                    {"X": X_np, "Z": Z_np, "Y": Y_np},
                    outfile,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            with open(
                "{}/regrets_by_mean_quantile{}.pkl".format(folder, quantile), "wb"
            ) as outfile:
                pickle.dump(
                    {
                        "regrets": all_regrets_by_mean,
                        "estimated_max_VaR_by_mean": all_estimated_max_var_by_mean,
                        "groundtruth_VaR_at_estimate": all_vars_at_est_maximizer_by_mean,
                        "estimated_max_VaR_by_lower": all_estimated_max_var_by_lower,
                        "optimal_VaR": opt_var_np,
                    },
                    outfile,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            with open(
                "{}/estimated_maximizers_quantile{}.pkl".format(folder, quantile), "wb"
            ) as outfile:
                pickle.dump(
                    {
                        "estimated_maximizers_by_mean": all_estimated_maximizers,
                        "estimated_maximizers_by_lower": all_estimated_maximizers_by_lower,
                        "optimal_VaR": opt_var_np,
                    },
                    outfile,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
