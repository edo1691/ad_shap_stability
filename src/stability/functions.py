from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import beta
from scipy import stats
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
import sys
from joblib import Parallel, delayed


def stability_measure_model(
        xtr,
        xte,
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
    xtr                 : np.array of shape (n_samples,n_features) containing the training set;
    xte                 : np.array of shape (m_samples,n_features) containing the test set;
    model               : object containing the anomaly detector;
    gamma               : float containing the contamination factor, i.e. the expected proportion of anomalies;
    unif                : bool selecting the case to exploit: If True, uniform sampling is selected, if False, biased
                        sampling is used;
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
    ntr, _ = xtr.shape
    nte, _ = xte.shape

    n_clust = 10
    # sample weights
    if not unif:
        cluster_labels = (
                KMeans(n_clusters=n_clust, random_state=331).fit(xtr).labels_ + 1
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
        if isinstance(xtr, pd.DataFrame):
            xs = xtr.iloc[sample_indices, :]
        else:  # Assume Xtr is a NumPy array
            xs = xtr[sample_indices, :]

        # fit and predict model
        model.fit(xs)
        probs = -model.score_samples(xte)
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
    random_stdev = np.sqrt((nte + 1) * (nte - 1) / (12 * nte ** 2))
    stability_scores = []
    stability_scores_list = []
    lower = 2 if intermediate_scores else iterations
    for i in range(lower, iterations + 1):
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
    stability_scores_list = 1.0 - np.array(stability_scores_list)

    if intermediate_scores:
        return stability_scores, stability_scores_list
    else:
        return stability_scores[0], stability_scores_list


def local_stability_measure(xtr, xte, model, gamma=0.1, iterations=500, psi=0.8, beta_flavor=2,
                            subset_low=0.25, subset_high=0.75, rank_method=True):
    np.random.seed(331)
    ntr, _ = xtr.shape
    nte, ft_col_te = xte.shape

    # Generate subsample sizes for all iterations
    subsample_sizes = np.random.randint(int(ntr * subset_low), int(ntr * subset_high), size=iterations)

    # Ensure that the subsample size does not exceed the population size
    subsample_sizes = np.minimum(subsample_sizes, ntr)

    # Generate subsample indices with correct sizes
    subsample_indices = [
        np.random.choice(ntr, size=size, replace=False)
        for size in subsample_sizes
    ]

    # Parallelize iterations with efficient memory usage
    results = Parallel(n_jobs=-1, backend='loky', prefer='threads')(
        delayed(iteration_function)(xtr, xte, model, indices, nte, ft_col_te)
        for indices in subsample_indices
    )

    # Stack and normalize rankings
    point_rankings = np.stack(results, axis=-1) / ft_col_te

    if rank_method:
        num_rankings = point_rankings.shape[-1]

        mode_values = stats.mode(point_rankings, axis=2, keepdims=True)[0]

        ranking_changes = np.abs(point_rankings - mode_values)
        ranking_changes = ranking_changes / np.maximum(point_rankings, 1e-10)

        stability_scores = 1 - np.sum(ranking_changes, axis=2) / (num_rankings - 1)
        stability_percentages = np.mean(np.clip(stability_scores, 0, 1), axis=1)

        return stability_percentages, stability_scores, point_rankings

    else:
        alpha_param, beta_param = calculate_beta_parameters(psi, gamma, beta_flavor)

        p_min = np.min(point_rankings, axis=2)
        p_max = np.max(point_rankings, axis=2)
        p_std = np.std(point_rankings, axis=2)

        p_area = beta.cdf(p_max, alpha_param, beta_param) - beta.cdf(p_min, alpha_param, beta_param)
        instance_stabilities = p_area * p_std

        random_stdev = np.sqrt((ft_col_te + 1) * (ft_col_te - 1) / (12 * ft_col_te ** 2))

        stability_scores = 1.0 - np.mean(np.clip(instance_stabilities / random_stdev, 0, 1), axis=1)
        stability_scores_list = 1.0 - np.clip(instance_stabilities / random_stdev, 0, 1)

        return stability_scores, stability_scores_list, point_rankings


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
        # Method 2: Parameters based on a portion of the distribution's mass within a specific range Use optimization
        # to find parameters that satisfy the condition the width of the beta distribution is set such that psi
        # percent of the mass of the distribution falls in the region [1 - 2 * gamma , 1]

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


def iteration_function(xtr, xte, model, sample_indices, nte, ft_col_te):
    """
    Performs operations for a single iteration: model fitting and SHAP value computation.
    """
    # Select the subsample based on the type of xtr
    if isinstance(xtr, pd.DataFrame):
        xs = xtr.iloc[sample_indices, :].values.astype(np.float32, copy=False)
    else:  # Assuming xtr is a numpy array
        xs = xtr[sample_indices, :].astype(np.float32, copy=False)

    # Fit the model to the subsample
    # Set n_jobs to -1 for parallelization, if not already set
    if model.get_params().get('n_jobs', None) != -1:
        model.set_params(n_jobs=-1)
    model.fit(xs)

    # Compute SHAP values for the test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(xte)

    # Rank features for each test instance based on their SHAP values
    # Using argsort and broadcasting to avoid loops
    sorted_indices = np.argsort(-shap_values, axis=1)
    iteration_rankings = np.argsort(sorted_indices, axis=1) + 1

    return iteration_rankings

