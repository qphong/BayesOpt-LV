import sys
import time
import os
import argparse
import pickle

epsilon = 1e-12

parser = argparse.ArgumentParser(
    description="Bayesian Optimization for Conditional Value at Risk."
)
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
    default=60,
)
parser.add_argument(
    "-r",
    "--numruns",
    help="number of random experiments",
    required=False,
    type=int,
    default=3,
)
parser.add_argument(
    "--ntrain",
    help="number of optimizing iterations",
    required=False,
    type=int,
    default=1000,
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
    default=2,
)
parser.add_argument(
    "--is_output_standardized",
    help="1 then the output will be standardized",
    required=False,
    type=int,
    default=0,
)
parser.add_argument(
    "--function",
    help="function to optimize",
    required=False,
    type=str,
    default="yacht_hydrodynamics",
)
parser.add_argument(
    "--quantile", help="quantile", required=False, type=float, default=0.3
)
parser.add_argument("--beta", help="beta", required=False, type=float, default=3.0)
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
    "--zmode",
    help="'random': select z randomly, 'max': select z with max confidence interval, 'info': select z by information gain",
    required=False,
    type=str,
    default="prob",
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

folder = "{}/cvar_discrete".format(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

folder = "{}/{}".format(folder, args.zmode)
if not os.path.exists(folder):
    os.makedirs(folder)


nquery = args.numqueries
nrun = args.numruns
ntrain = args.ntrain
n_init_data = args.n_init_data
n_iter_fitgp = args.n_iter_fitgp
zmode = args.zmode
shuffletie = args.shuffletie
is_output_standardized = args.is_output_standardized

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
import cvaropt_discrete


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

zvalues_np = f_info["zvalues"]
zprobs_np = f_info["zprobs"]
zlprobs_np = f_info["zlprobs"]


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

        bound = tf.reshape(bound, shape=(1, nx, 1, nz))
        return bound

    # maximizer upper_varopt to find the query x
    upper_cvaropt = cvaropt_discrete.CVaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        lambda x, z: get_bound(x, z, beta_plc, type="upper"),
        zvalues_np.shape[0],
        zvalues_np,
        zprobs_np,
        n_rand_opt_init,
        graph,
        name="upper_cvaropt",
        dtype=dtype,
    )

    lower_cvaropt = cvaropt_discrete.CVaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        lambda x, z: get_bound(x, z, beta_plc, type="lower"),
        zvalues_np.shape[0],
        zvalues_np,
        zprobs_np,
        n_rand_opt_init,
        graph,
        name="lower_cvaropt",
        dtype=dtype,
    )

    # find lower bound of at the query x
    query_x_plc = tf.placeholder(dtype=dtype, shape=(1, xdim), name="query_x_plc")

    # quantile_f_vals, lt_risk_levels, \
    #             lt_quantile_f_probs, lt_quantile_f_vals

    (
        _,
        lower_lt_risks,
        _,
        lower_lt_fvals,
        lower_cvar_at_queryx,
    ) = lower_cvaropt.find_cvar(query_x_plc)

    # estimate the maximizer
    #   by maximizing mean function
    mean_cvaropt = cvaropt_discrete.CVaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        lambda x, z: get_bound(x, z, beta_plc, type="mean"),
        zvalues_np.shape[0],
        zvalues_np,
        zprobs_np,
        n_rand_opt_init,
        graph,
        name="mean_cvaropt",
        dtype=dtype,
    )

    # for querying z
    x0_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name="x2")
    z0 = tf.constant(zvalues_np, dtype=dtype)

    upper_f0 = tf.squeeze(
        get_bound(
            tf.reshape(x0_plc, shape=(1, tf.shape(x0_plc)[0], 1, 1, xdim)),
            z0,
            beta_plc,
            type="upper",
        )
    )
    # (nz,)
    lower_f0 = tf.squeeze(
        get_bound(
            tf.reshape(x0_plc, shape=(1, tf.shape(x0_plc)[0], 1, 1, xdim)),
            z0,
            beta_plc,
            type="lower",
        )
    )
    # (nz,)

    # find the ground truth function value for evaluating the regret
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

    ground_truth_cvaropt = cvaropt_discrete.CVaROpt(
        xdim,
        zdim,
        [xmin, xmax],
        ground_truth_func,
        zvalues_np.shape[0],
        zvalues_np,
        zprobs_np,
        n_rand_opt_init,
        graph,
        name="ground_truth_cvaropt",
        dtype=dtype,
    )

    # compute the ground truth cvar at estimated maximizer for computing the regret
    # find lower bound of at the query x
    est_maximizer_plc = tf.placeholder(
        dtype=dtype, shape=(1, xdim), name="est_maximizer_plc"
    )
    _, _, _, _, ground_truth_cvar_at_est_max = ground_truth_cvaropt.find_cvar(
        est_maximizer_plc
    )


np.random.seed(random_seed)

# general initial observations for all random runs:
init_X_np = np.random.rand(nrun, n_init_data, xdim) * (xmax - xmin) + xmin
init_Z_idxs = np.random.choice(
    zvalues_np.shape[0], size=(nrun, n_init_data), replace=True
)

with open(
    "{}/init_observations_seed{}.pkl".format(folder, random_seed), "wb"
) as outfile:
    pickle.dump(
        {"init_X_np": init_X_np, "init_Z_idxs": init_Z_idxs},
        outfile,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

with graph.as_default():

    with tf.Session() as sess:
        assert min(zprobs_np) > 1e-6

        sess.run(tf.global_variables_initializer())

        # opt_cvar_np, opt_x_np, _, _ = ground_truth_cvaropt.maximize_in_session(
        #             sess,
        #             init_x = rand_opt_init_x_np,
        #             n_x_train = 400,
        #             feed_dict = {
        #                 ground_truth_cvaropt.quantile_plc: quantile
        #             },
        #             verbose = 100
        #         )

        # print("Optimal X: {}, CVaR: {}".format(opt_x_np, opt_cvar_np)); sys.stdout.flush()
        # raise Exception("test")

        opt_x_np = None
        opt_cvar_np = f_info["max_cvar_discrete"]

        all_regrets_by_mean = np.zeros([nrun, nquery])
        all_cvars_at_est_maximizer_by_mean = np.zeros([nrun, nquery])
        all_regrets_by_lower = np.zeros([nrun, nquery])
        all_estimated_maximizers_by_mean = np.zeros([nrun, nquery, xdim])
        all_estimated_maximizers_by_lower = np.zeros([nrun, nquery, xdim])
        all_estimated_max_var_by_mean = np.zeros([nrun, nquery])
        all_estimated_max_var_by_lower = np.zeros([nrun, nquery])

        for run_idx in range(nrun):
            print("{}. RANDOM RUN".format(run_idx))

            # Generate initial observations
            X_np = init_X_np[run_idx, :, :]
            Z_idxs = init_Z_idxs[run_idx, :]
            Z_np = zvalues_np[Z_idxs]

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
                # beta_np = 2. * np.log((query_idx+1)**2 * np.pi**2 / 6. / 0.1)
                # beta_np = 2. * np.log((query_idx+1) * np.pi**2 / 12.) # for portfolio_gaussian!
                # beta_np = 3.0
                beta_np = args.beta
                # beta_np = 1.5
                print("NOTE BETA = {}".format(beta_np))
                print("Observation:")
                print(Y_np)

                # NOTE: to generate Y_np we need generate_obs_noise_var_np for synthetic function, this is unknown to the BO algorithm. In BO, the noise variance is learned from the data

                # NOTE: we do not scale Y_np, because it may cause the pre-trained GP model for the function incorrect!
                #    it also causes the noise min_var incorrectly estimated
                #    i.e., scaling Y_np changes the noise var
                # scaled_Y_np = Y_np / (np.max(Y_np) - np.min(Y_np))

                if n_iter_fitgp > 0 and query_idx % n_iter_fitgp == 0:
                    print("Fit GP to observations")

                    if is_output_standardized:
                        Y_mean = np.mean(Y_np)
                        Y_std = np.std(Y_np)

                        standardized_Y = (Y_np - Y_mean) / Y_std
                    else:
                        standardized_Y = Y_np

                    has_error, gp_hyperparameters = functions.fit_gp(
                        input_np,
                        standardized_Y,
                        noise_var=likelihood_variance_np,
                        train_noise_var=True,
                        min_var=min_var,
                        max_var=max_var,
                    )

                    if has_error:
                        print(" Skip due to numerical error!")
                    else:
                        meanf_const = gp_hyperparameters["meanf"]
                        signal_variance_np = gp_hyperparameters["signal_var"]
                        lengthscale_np = gp_hyperparameters["lengthscale"]
                        likelihood_variance_np = gp_hyperparameters["noise_var"]

                if is_output_standardized:
                    standardized_Y = (Y_np - Y_mean) / Y_std
                    shifted_Y_np = standardized_Y - meanf_const
                else:
                    shifted_Y_np = Y_np - meanf_const

                lengthscale.load(lengthscale_np.reshape(1, input_dim), sess)
                signal_variance.load(np.squeeze(signal_variance_np), sess)
                likelihood_variance.load(np.squeeze(likelihood_variance_np), sess)

                feed_dict = {
                    X_plc: X_np,
                    Z_plc: Z_np,
                    Y_plc: shifted_Y_np,
                    beta_plc: beta_np,
                    mean_cvaropt.quantile_plc: quantile,
                    upper_cvaropt.quantile_plc: quantile,
                    lower_cvaropt.quantile_plc: quantile,
                    ground_truth_cvaropt.quantile_plc: quantile,
                }

                invK_np = sess.run(invK, feed_dict=feed_dict)

                feed_dict[invK_plc] = invK_np

                # max_x_mean_np, max_quantile_f_mean_np = mean_varopt.maximize_in_session(
                #     sess,
                #     init_x = rand_opt_init_x_np,
                #     n_x_train = ntrain,
                #     feed_dict = feed_dict,
                #     verbose = 1000
                # )

                max_x_mean_np, max_cvar_mean_np = mean_cvaropt.find_max_in_set(
                    sess, X_np, feed_dict, batchsize=50
                )

                all_estimated_maximizers_by_mean[
                    run_idx, query_idx, :
                ] = max_x_mean_np.squeeze()

                all_estimated_max_var_by_mean[run_idx, query_idx] = np.squeeze(
                    max_cvar_mean_np
                )

                print(
                    "Estimated maximizer at {} CVaR {}".format(
                        max_x_mean_np, max_cvar_mean_np
                    )
                )
                sys.stdout.flush()

                # computing the regret
                feed_dict[est_maximizer_plc] = max_x_mean_np.reshape(1, xdim)
                ground_truth_cvar_at_est_max_np = sess.run(
                    ground_truth_cvar_at_est_max, feed_dict
                )
                regret = opt_cvar_np - ground_truth_cvar_at_est_max_np.squeeze()

                print(
                    "Groundtruth CVaR at the query x by max mean: ",
                    ground_truth_cvar_at_est_max_np.squeeze(),
                )
                print("Regret by maximizing mean: ", regret)
                all_regrets_by_mean[run_idx, query_idx] = regret
                all_cvars_at_est_maximizer_by_mean[
                    run_idx, query_idx
                ] = ground_truth_cvar_at_est_max_np.squeeze()

                cv_ucb_start_time = time.time()

                (
                    upper_cvar_at_queryx_np,
                    query_x_np,
                    upper_lt_risks_np,
                    upper_lt_fvals_np,
                ) = upper_cvaropt.maximize_in_session(
                    sess,
                    init_x=rand_opt_init_x_np,
                    n_x_train=ntrain,
                    feed_dict=feed_dict,
                    verbose=1000,
                )

                print("Query x: {}".format(query_x_np))
                print(
                    "  At query, upper bound of function value: {:.10f}".format(
                        np.squeeze(upper_cvar_at_queryx_np)
                    )
                )
                sys.stdout.flush()

                feed_dict[query_x_plc] = query_x_np.reshape(1, xdim)
                (
                    lower_cvar_at_queryx_np,
                    lower_lt_risks_np,
                    lower_lt_fvals_np,
                ) = sess.run(
                    [lower_cvar_at_queryx, lower_lt_risks, lower_lt_fvals], feed_dict
                )

                print(
                    "  At query, lower bound of function value: {:.10f}".format(
                        np.squeeze(lower_cvar_at_queryx_np.squeeze())
                    )
                )

                print("Find risk maximizing (upper - lower)")
                # upper_lt_fvals_np - lower_lt_fvals_np
                # upper_lt_risks_np - lower_lt_risks_np
                lp = 0
                up = 0
                max_interval = -1e9
                max_interval_risk = None
                max_interval_lower_fval = None
                max_interval_upper_fval = None

                lower_lt_risks_np = np.reshape(lower_lt_risks_np, (-1,))
                upper_lt_risks_np = np.reshape(upper_lt_risks_np, (-1,))
                lower_lt_fvals_np = np.reshape(lower_lt_fvals_np, (-1,))
                upper_lt_fvals_np = np.reshape(upper_lt_fvals_np, (-1,))

                while (
                    lp < len(lower_lt_risks_np)
                    and up < len(upper_lt_risks_np)
                    and lower_lt_risks_np[lp] > 1e-6
                    and upper_lt_risks_np[up] > 1e-6
                ):

                    interval = upper_lt_fvals_np[up] - lower_lt_fvals_np[lp]

                    if interval > max_interval:
                        max_interval = interval
                        max_interval_risk = min(
                            lower_lt_risks_np[lp], upper_lt_risks_np[up]
                        )
                        max_interval_lower_fval = lower_lt_fvals_np[lp]
                        max_interval_upper_fval = upper_lt_fvals_np[up]

                    if lower_lt_risks_np[lp] < upper_lt_risks_np[up] - 1e-6:
                        lp += 1
                    elif lower_lt_risks_np[lp] > upper_lt_risks_np[up] + 1e-6:
                        up += 1
                    else:
                        lp += 1
                        up += 1

                # print(lower_lt_risks_np)
                # print(upper_lt_risks_np)
                # print(max_interval_risk)
                # print(lower_lt_fvals_np)
                # print(upper_lt_fvals_np)

                # find lacing value whose confidence interval
                # contains [max_interval_lower_fval, max_interval_upper_fval]
                print("Query z:")

                feed_dict[x0_plc] = query_x_np.reshape(1, xdim)
                upper_f0_np, lower_f0_np = sess.run([upper_f0, lower_f0], feed_dict)

                uncertain_zs_np = []
                conf_intervals = []
                uncertain_z_lprobs_np = []

                for i, zi in enumerate(zvalues_np):
                    if (
                        lower_f0_np[i] <= max_interval_lower_fval + 1e-6
                        and upper_f0_np[i] >= max_interval_upper_fval - 1e-6
                    ):
                        print(
                            "{} (p={}): function values in confidence interval [{:.10f}, {:.10f}]".format(
                                zi, zprobs_np[i], lower_f0_np[i], upper_f0_np[i]
                            )
                        )

                        uncertain_zs_np.append(zi)
                        uncertain_z_lprobs_np.append(zlprobs_np[i])

                if len(uncertain_zs_np) == 0:
                    raise Exception("Query z is None!")

                elif len(uncertain_zs_np) == 1 or zmode == "random":
                    query_z_np = uncertain_zs_np[
                        np.random.randint(len(uncertain_zs_np))
                    ]

                elif zmode == "prob":
                    if shuffletie:
                        print("Shuffling when tie exists in the loss of z!")
                        uncertain_z_lprobs_np = np.array(uncertain_z_lprobs_np)
                        shuffled_idxs = np.array(
                            list(range(len(uncertain_z_lprobs_np)))
                        )
                        np.random.shuffle(shuffled_idxs)

                        shuffled_uncertain_z_lprobs_np = uncertain_z_lprobs_np[
                            shuffled_idxs
                        ]

                        shuffled_max_idx = np.argmax(shuffled_uncertain_z_lprobs_np)

                        max_idx = shuffled_idxs[shuffled_max_idx]

                        query_z_np = uncertain_zs_np[max_idx]
                        print(
                            "TEST: {} {}".format(
                                uncertain_z_lprobs_np[max_idx],
                                np.max(uncertain_z_lprobs_np),
                            )
                        )
                    else:
                        print("No shuffling when tie exists in the loss of z")
                        query_z_np = uncertain_zs_np[np.argmax(uncertain_z_lprobs_np)]
                else:
                    raise Exception("Unknown mode!")

                cv_ucb_end_time = time.time()
                print(
                    "Selected z: {} in {} possible z by {}".format(
                        query_z_np, len(uncertain_zs_np), zmode
                    )
                )
                print(
                    "   running CV-UCB in {:.4f}s".format(
                        cv_ucb_end_time - cv_ucb_start_time
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

                print("Function evaluation: {}".format(func(query_np).reshape(-1, 1)))

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
                        "groundtruth_VaR_at_estimate": all_cvars_at_est_maximizer_by_mean,
                        "optimal_VaR": opt_cvar_np,
                    },
                    outfile,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            with open(
                "{}/regrets_by_lower_quantile{}.pkl".format(folder, quantile), "wb"
            ) as outfile:
                pickle.dump(
                    {
                        "regrets": all_regrets_by_lower,
                        "estimated_max_VaR_by_lower": all_estimated_max_var_by_lower,
                        "optimal_VaR": opt_cvar_np,
                    },
                    outfile,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            with open(
                "{}/estimated_maximizers_quantile{}.pkl".format(folder, quantile), "wb"
            ) as outfile:
                pickle.dump(
                    {
                        "estimated_maximizers_by_mean": all_estimated_maximizers_by_mean,
                        "estimated_maximizers_by_lower": all_estimated_maximizers_by_lower,
                    },
                    outfile,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
