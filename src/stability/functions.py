from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import beta
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
import sys
from joblib import Parallel, delayed


def stability_measure_model(
    Xtr,
    Xte,
    model,
    gamma,
    unif=True,  # pick True or False
    iterations=500,
    psi=0.8,
    beta_flavor=2,  # pick from: 1, 2
    subset_low=0.25,
    subset_high=0.75,
    intermediate_scores=False,
):
    """
    Parameters
    ----------
    Xtr                 : np.array of shape (n_samples,n_features) containing the training set;
    Xte                 : np.array of shape (m_samples,n_features) containing the test set;
    model               : object containing the anomaly detector;
    gamma               : float containing the contamination factor, i.e. the expected proportion of anomalies;
    unif                : bool selecting the case to exploit: If True, uniform sampling is selected, if False, biased sampling is used;
    iterations          : int regarding the number of iterations;
    psi                 : float in [0, 1], the hyperparameter controlling the shape of the beta distribution;
    beta_flavor         : int equal to either 1 or 2 selecting the way the beta distribution parameters are chosen;
    subset_low          : float containing the lower bound of subsample size as percent of training length;
    subset_high         : float containing the upper bound of subsample size as percent of training length;
    intermediate_scores : bool which selects whether also to compute all intermediate stability scores. Default = False.

    Returns
    -------
    S                   : float representing the stability measure;
    IS                  : float representing the instability measure.
    """

    np.random.seed(331)
    ntr, _ = Xtr.shape
    nte, _ = Xte.shape

    n_clust = 10
    # sample weights
    if not unif:
        cluster_labels = (
            KMeans(n_clusters=n_clust, random_state=331).fit(Xtr).labels_ + 1
        )
    else:
        weights = np.ones(ntr) * (1 / ntr)

    if psi == 0:
        psi = max(0.51, 1 - (gamma + 0.05))

    # sample weights
    norm = np.ones(ntr) * (1 / ntr)

    # compute the rankings over the test set
    point_rankings = np.zeros((nte, iterations), dtype=float)
    for i in range(iterations):
        if not unif:
            biased_weights = {}
            for w in range(1, n_clust + 1):
                biased_weights[w] = np.random.randint(1, 100)
            weights = np.asarray([biased_weights[w] for w in cluster_labels])
            weights = weights / sum(weights)
        # draw subsample
        subsample_size = np.random.randint(
            int(ntr * subset_low), int(ntr * subset_high)
        )
        sample_indices = np.random.choice(
            np.arange(ntr), size=subsample_size, p=weights, replace=False
        )
        # Check if Xtr is a DataFrame
        if isinstance(Xtr, pd.DataFrame):
            Xs = Xtr.iloc[sample_indices, :]
        else:  # Assume Xtr is a NumPy array
            Xs = Xtr[sample_indices, :]

        # fit and predict model
        model.fit(Xs)
        probs = -model.score_samples(Xte)
        anom_probs = np.nan_to_num(probs)

        # construct test set rankings
        sorted_ixs = np.argsort(anom_probs)[::-1]  # first index = lowest score
        for ii, si in enumerate(sorted_ixs):
            point_rankings[si, i] = ii + 1

    # normalize rankings
    point_rankings = point_rankings / nte  # lower rank = more normal

    if beta_flavor == 1:
        # The area of the Beta distribution is the same in the intervals [0, psi] and [psi, 1]
        beta_param = float(
            (1 / (psi + gamma - 1))
            * (2 * gamma - 1 - gamma / 3 + psi * ((3 - 4 * gamma) / 3))
        )
        alpha_param = float(
            beta_param * ((1 - gamma) / gamma) + (2 * gamma - 1) / gamma
        )

    elif beta_flavor == 2:
        # the width of the beta distribution is set such that psi percent of the mass
        # of the distribution falls in the region [1 - 2 * gamma , 1]

        # optimization function
        def f(p):
            return ((1.0 - psi) - beta.cdf(1.0 - 2 * gamma, p[0], p[1])) ** 2

        # bounds
        bounds = Bounds([1.0, 1.0], [np.inf, np.inf])
        # linear constraint
        linear_constraint = LinearConstraint(
            [[gamma, gamma - 1.0]], [2 * gamma - 1.0], [2 * gamma - 1.0]
        )
        # optimize
        p0 = np.array([1.0, 1.0])
        res = minimize(
            f,
            p0,
            method="trust-constr",
            constraints=[linear_constraint],
            options={"verbose": 0},
            bounds=bounds,
        )
        alpha_param = res.x[0]
        beta_param = res.x[1]
    else:
        print("Wrong choice! Pick a better one!")
        sys.exit()

    # compute the stability score for multiple iterations
    random_stdev = np.sqrt((nte + 1) * (nte - 1) / (12 * nte**2))
    stability_scores = []
    stability_scores_list = []
    lower = 2 if intermediate_scores else iterations
    for i in range(lower, iterations + 1):
        # point stabilities
        point_stabilities = np.zeros(nte, dtype=float)
        for ii in range(nte):
            p_min = np.min(point_rankings[ii, :i])
            p_max = np.max(point_rankings[ii, :i])
            p_std = np.std(point_rankings[ii, :i])
            p_area = beta.cdf(p_max, alpha_param, beta_param) - beta.cdf(
                p_min, alpha_param, beta_param
            )
            point_stabilities[ii] = p_area * p_std

        # aggregated stability
        stability_scores.append(
            np.mean(np.minimum(1, point_stabilities / random_stdev))
        )
        stability_scores_list.append(np.minimum(1, point_stabilities / random_stdev))

    stability_scores = 1.0 - np.array(stability_scores)
    instability_scores = 1.0 - stability_scores
    stability_scores_list = 1.0 - np.array(stability_scores_list)

    if intermediate_scores:
        return stability_scores, stability_scores_list
    else:
        return stability_scores[0], stability_scores_list


def normalize_rankings(rankings):
    """
    Normalizes the rankings using min-max normalization.

    Parameters
    ----------
    rankings : np.array
        Array of rankings.

    Returns
    -------
    np.array
        Normalized rankings.
    """
    # min_rank = rankings.min(axis=2, keepdims=True)
    # max_rank = rankings.max(axis=2, keepdims=True)
    # Min & Max over all the dataframe
    min_rank = rankings.min()
    max_rank = rankings.max()
    # Prevent division by zero
    normalized_rankings = np.where(max_rank > min_rank, (rankings - min_rank) / (max_rank - min_rank), 0)
    return normalized_rankings


def calculate_beta_parameters(psi, gamma, beta_flavor):
    """
    Calculates the parameters for the beta distribution.

    Parameters
    ----------
    psi : float
        Controls the shape of the beta distribution.
    gamma : float
        Contamination factor.
    beta_flavor : int
        Method for determining beta distribution parameters.

    Returns
    -------
    tuple
        Alpha and beta parameters for the beta distribution.
    """
    if beta_flavor == 1:
        # Method 1: Parameters based on ensuring equal mass in specified intervals
        # Calculate parameters alpha and beta for the beta distribution
        # The area of the Beta distribution is the same in the intervals [0, psi] and [psi, 1]
        beta_param = float(
            (1 / (psi + gamma - 1))
            * (2 * gamma - 1 - gamma / 3 + psi * ((3 - 4 * gamma) / 3))
        )
        alpha_param = float(
            beta_param * ((1 - gamma) / gamma) + (2 * gamma - 1) / gamma
        )
    elif beta_flavor == 2:
        # Method 2: Parameters based on a portion of the distribution's mass within a specific range
        # Use optimization to find parameters that satisfy the condition
        # the width of the beta distribution is set such that psi percent of the mass of the distribution falls in the region [1 - 2 * gamma , 1]

        # optimization function
        def f(p):
            return ((1.0 - psi) - beta.cdf(1.0 - 2 * gamma, p[0], p[1])) ** 2

        # bounds
        bounds = Bounds([1.0, 1.0], [np.inf, np.inf])
        # linear constraint
        linear_constraint = LinearConstraint(
            [[gamma, gamma - 1.0]], [2 * gamma - 1.0], [2 * gamma - 1.0]
        )
        # optimize
        p0 = np.array([1.0, 1.0])
        res = minimize(
            f,
            p0,
            method="trust-constr",
            constraints=[linear_constraint],
            options={"verbose": 0},
            bounds=bounds,
        )
        alpha_param = res.x[0]
        beta_param = res.x[1]
    else:
        raise ValueError("Invalid beta_flavor choice. Please select 1 or 2.")
    return alpha_param, beta_param


def iteration_function(Xtr, Xte, model, sample_indices, nte, ft_col_te):
    """
    Performs operations for a single iteration: model fitting and SHAP value computation.
    """
    # Select the subsample and
    # Convert the DataFrame to a more memory-efficient format if not already done
    Xs = Xtr.iloc[sample_indices, :].astype(np.float32)

    # Fit the model to the subsample
    # Ensure n_jobs is set to utilize all available cores for parallelization, if not set already
    if 'n_jobs' not in model.get_params() or model.get_params()['n_jobs'] != -1:
        model.set_params(n_jobs=-1)
    model.fit(Xs)

    # Compute SHAP values for the test set
    shap_values = shap.TreeExplainer(model).shap_values(Xte)

    # Rank features for each test instance based on their SHAP values
    iteration_rankings = np.zeros((nte, ft_col_te), dtype=float)
    for j in range(nte):
        for ii, si in enumerate(np.argsort(shap_values[j, :])[::-1]):
            iteration_rankings[j, si] = ii + 1
    return iteration_rankings


def local_stability_measure(Xtr, Xte, model, gamma=0.1, iterations=500, psi=0.8, beta_flavor=2,
                                     subset_low=0.25, subset_high=0.75):
    """
    Computes the local stability measure using parallel processing for iterations.
    """
    np.random.seed(331)
    ntr, _ = Xtr.shape
    nte, ft_col_te = Xte.shape

    # Generate subsample indices for all iterations in advance
    subsample_sizes = np.random.randint(int(ntr * subset_low), int(ntr * subset_high), size=iterations)
    subsample_indices = [np.random.choice(ntr, size=size, replace=False) for size in subsample_sizes]

    # Parallelize iterations
    results = Parallel(n_jobs=-1)(
        delayed(iteration_function)(Xtr, Xte, model, indices, nte, ft_col_te) for indices in subsample_indices)

    # Convert list of iteration rankings into a single array
    point_rankings = np.array(results).transpose((1, 2, 0))

    # Normalize rankings
    normalized_point_rankings = normalize_rankings(point_rankings)

    # Calculate beta distribution parameters
    alpha_param, beta_param = calculate_beta_parameters(psi, gamma, beta_flavor)

    # compute the stability score for multiple iterations
    random_stdev = np.sqrt((ft_col_te + 1) * (ft_col_te - 1) / (12 * ft_col_te ** 2))

    # Compute stability scores using the beta distribution and rankings
    stability_scores = np.zeros(nte)

    for j in range(nte):
        instance_stabilities = []
        for ii in range(ft_col_te):
            p_min, p_max = np.min(normalized_point_rankings[j, ii]), np.max(normalized_point_rankings[j, ii])
            p_std = np.std(normalized_point_rankings[j, ii])
            p_area = beta.cdf(p_max, alpha_param, beta_param) - beta.cdf(p_min, alpha_param, beta_param)
            instance_stabilities.append(p_area * p_std)
        # Compute aggregated stability for the current test instance across all features
        stability_scores[j] = np.mean(np.minimum(1, np.array(instance_stabilities) / random_stdev))

    # Average stability scores over all iterations to get a single measure per instance
    stability_scores_per_instance = 1.0 - stability_scores
    instability_scores = 1.0 - stability_scores_per_instance

    return stability_scores, instability_scores

# std_per_instance_and_feature = np.std(normalized_point_rankings, axis=2)

# mean_std_devs_per_instance = np.mean(std_per_instance_and_feature, axis=1)
