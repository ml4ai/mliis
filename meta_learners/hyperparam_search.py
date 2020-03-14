"""
Functions to estimate optimal hyperparameters when adapting to unseen tasks.

Contains the implementation of the Update Hyperparameter Optimization algorithm.
"""

from collections import deque
import operator
import os
from copy import copy
from typing import Callable, Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Categorical, Real, Integer

DROPOUT_RATE_NAME = "drop_rate"
AUG_RATE_NAME = "aug_rate"
BATCH_SIZE_NAME = "inner_batch_size"
LEARNING_RATE_NAME = "lr"
SUPPORTED_SEARCH_ALGS = {"GP"}

class EarlyStopper:
    """
    Computes stopping criterion given a metric and a patience.
    """

    def __init__(self, patience: int = 10, metric_should_increase: bool = True, min_steps: int = 0):
        """
        Args:
            patience: How many steps to continue training if eval does not improve.
            metric_should_increase: If True, metric is expected to increase (i.e. set to True for accuracy or IoU,
                False for a loss function such as cross entropy.
        """
        self.patience = patience
        self.metric_should_increase = metric_should_increase
        if metric_should_increase:
            self.eval_operator = operator.gt
        else:
            self.eval_operator = operator.lt
        self._best_metric = None
        self._best_num_steps = None
        self.num_evals_without_improving = 0
        self.min_steps = min_steps
        if min_steps > 0:
            self._best_num_steps = min_steps
        print("Built EarlyStopper with patience {}".format(self.patience))

    def continue_training(self, metric, total_steps_taken):
        if total_steps_taken <= self.min_steps:
            self._best_metric = metric
            return True
        elif self._best_metric is None or self.eval_operator(metric, self._best_metric):
            self.num_evals_without_improving = 0
            self._best_metric = metric
            self._best_num_steps = total_steps_taken
        else:
            self.num_evals_without_improving += 1
            if self.num_evals_without_improving > self.patience:
                return False
        return True

    def best_metric(self):
        return self._best_metric

    def best_num_steps(self):
        return self._best_num_steps


def run_m(eval_fn: Callable, params: Dict, m: int = 1):
    """
    Calls `eval_fn` with `params`, returns results. Assumes that eval_fn returns a tuple of:
        (list of task IDs, list of iterations, and list of metrics).

    Args:
        eval_fn: A callable that takes `params` and returns a tuple of:
            (list of task IDs, list of iterations, and list of metrics).
        params: kwargs fed into eval_fn call
        m: Number of times to eval_fn(**params).

    Returns:
        The metrics returned by running eval_fn with params
    """
    # print("Running evaluation with params: {}".format(params))
    all_task_ids, all_num_steps, all_metrics = [], [], []
    for _ in range(m):
        task_ids, num_steps, metrics = eval_fn(**params)
        all_task_ids.extend(task_ids)
        all_num_steps.extend(num_steps)
        all_metrics.extend(metrics)
    return all_task_ids, all_num_steps, all_metrics


def save_results(results: List[Tuple[Dict, Tuple[List, List, List]]], path: str, metric_name: str = "mIoU", append_if_exists: bool = False):
    """Takes the results and saves them to csv"""
    print("Saving results to {}".format(path))
    # Format results to make a dataframe:
    formatted = {"task_ID": [], "best_num_steps": [], metric_name: []}
    for result in results:
        config, config_results = result
        task_ids, num_steps, metrics = config_results

        for key, val in config.items():
            try:
                formatted[key].extend([val for _ in range(len(task_ids))])
            except KeyError:
                formatted[key] = [val for _ in range(len(task_ids))]

        formatted["task_ID"].extend(task_ids)
        formatted["best_num_steps"].extend(num_steps)
        formatted[metric_name].extend(metrics)
    df = pd.DataFrame(formatted)
    mode = "w"
    header = True
    if os.path.exists(path):
        if not append_if_exists:
            i = 0
            while True:
                new_path = path + "_{}".format(i)
                if not os.path.exists(new_path):
                    break
                i += 1
            path = new_path
            mode = "w"
            header = True
        else:
            mode = "a"
            header = False
    df.to_csv(path, index=False, mode=mode, header=header)
    print("Saved optimization raw results to {}".format(path))


def compute_best_configuration(results_list, metric_should_increase=True):
    if metric_should_increase:
        eval_operator = operator.gt
        best_metric = -np.inf
    else:
        eval_operator = operator.lt
        best_metric = np.inf

    for sampled_config, results in results_list:
        task_ids, num_steps, metrics = results
        miou_across_tasks = np.mean(metrics)
        if eval_operator(miou_across_tasks, best_metric):
            best_config = sampled_config
            best_metric = miou_across_tasks
            m_best_num_steps = np.median(num_steps)
            best_step_num = m_best_num_steps

    print("Best mIoU found: {}".format(best_metric))
    print("with median iteration: {}".format(best_step_num))
    print("and config: {}".format(best_config))

    return best_config, int(best_step_num), best_metric


def log_opt_progress(hyperparams, results_i, task_ids, num_steps, metrics, save_results_to):
    print("Results for hyperparams {}: task IDs: {}, best num steps: {}, mIoUs: {}".format(hyperparams, task_ids, num_steps,
                                                                                      metrics))
    print("mean mIoU: {}".format(np.nanmean(metrics)))

    if save_results_to is not None:
        save_results([results_i], save_results_to, append_if_exists=True)


def insert_sampled_into_full_set_of_hyperparams(sampled, hyperparams) -> Dict:
    for key, val in sampled.items():
        hyperparams[key] = val
    return hyperparams


def get_dim_type(value: List[Any]):
    value = value[0]
    if isinstance(value, float):
        return Real
    elif isinstance(value, int):
        return Integer
    elif isinstance(value, str):
        return Categorical
    else:
        raise ValueError("Value must be float, int, or str, but {} is {}".format(value, type(value)))


def gp_update_hyperparameter_optimization(eval_fn: Callable, hyperparams: Dict, search_key_ranges: Dict, n: int,
                              save_results_to: Optional[str] = "gp_hyper_param_search_results.csv", m: int = 1,
                              metric_should_increase: bool = True, metric_name: str = "mIoU", base: int = 2,
                                          n_initial_points: Optional[int] = None, prior: str = "log-uniform"):
    """
    Multitask hyperparameter search with Gaussian process regression of values in search_key_ranges.
    Calls `eval_fn` with `params`, replacing values in `params` with expected improvement maximizing values sampled from
     the ranges in `search_key_ranges` for the keys that are in both `params` and `search_key_ranges`.

    Args:
        eval_fn: The function to call with params that returns a metric.
        hyperparams: Dictionary of kwargs that must be specified to call eval_fn.
        search_key_ranges: Dictionary mapping a key in params to a range to sample from.
        n: number of hyperparameter configurations to sample.
        m: number of train-val splits datasets to sample
        metric_should_increase: If true
        prior: Sample points from this distribution. E.g., "log-uniform" sample from a log scaled uniform distribution.

    Returns:
        Tuple of the sampled values in a dictionary with the same keys as search_key_ranges and the resulting metric.
    """
    for key in search_key_ranges.keys():
        assert key in hyperparams, "key: {} not in hyperparams: {}".format(key, hyperparams)

    if n_initial_points is None:
        n_initial_points = int(n / 2)
    print("Sampling {} points initially at random.".format(n_initial_points))

    search_dim_types = {key: get_dim_type(val) for key, val in search_key_ranges.items()}

    dims = [search_dim_types[key](domain[0], domain[1], prior=prior, base=base, name=key) for key, domain in search_key_ranges.items() if domain[0] != domain[1]]
    dim_names = [dim.name for dim in dims]
    opt = Optimizer(
        dims,
        "GP",  # Estimate metric as a function of lr using a Gaussian Process.
        acq_func='EI',  # Use Expected Improvement as an acquisition function.
        acq_optimizer="lbfgs",  # Draw random samples from GP then optimize to find best lr to suggest.
        n_initial_points=n_initial_points,  # First points will be completely random to avoid exploiting too early.
    )

    results = []
    # Would just need to add function evaluation logic to this loop.
    for i in range(n):
        print("Running configuration sample {} of {}.".format(i + 1, n))
        print("With sampled hyperparams:")
        sampled_list = opt.ask()
        sampled = {name: x for name, x in zip(dim_names, sampled_list)}
        print(sampled)

        hyperparams = insert_sampled_into_full_set_of_hyperparams(sampled, hyperparams)

        task_ids, num_steps, metrics = run_m(eval_fn, hyperparams, m)

        # Most recent metric observed for given params
        objective = np.nanmean(metrics)
        if metric_should_increase:
            objective *= -1
        print("Objective value at sample {} of {}: {}".format(i + 1, n, objective))
        opt_result = opt.tell(sampled_list, objective)

        results_i = (sampled, (task_ids, num_steps, metrics))
        results.append(results_i)
        log_opt_progress(hyperparams, results_i, task_ids, num_steps, metrics, save_results_to)

    best_config, expected_best_step_num, best_metric = compute_best_configuration(results, metric_should_increase)

    return best_config, expected_best_step_num, best_metric, results


def lr_droprate_aug_rate_batch_size_gp_search(eval_fn: Callable, params: Dict, lr_name: str = LEARNING_RATE_NAME, lr_search_range_low: float = 0.0005, lr_search_range_high: float = 0.05,
                                              droprate_name: str = DROPOUT_RATE_NAME, drop_rate_search_range_low: float = 0.2, drop_rate_search_range_high: float = 0.2,
                                             aug_rate_name: str = AUG_RATE_NAME, aug_rate_search_range_low: float = 0.5, aug_rate_search_range_high: float = 0.5,
                                              batch_size_name: str = BATCH_SIZE_NAME, batch_size_search_range_low: int = 8, batch_size_search_range_high: int = 8,
                                              n: int = 100,
                                              save_results_to: str = "hyper_param_search_results.csv", m: int = 1,
                                              metric_should_increase: bool = True, metric_name: str = "mIoU", b: int = 3, x: float = 0.1) -> Tuple[float, int]:
    """
    Performs search over learning rates by randomly sampling within range and successively reducing the range based on
    top x percent of results. Returns the best learning rate and expected number of iterations.
    """
    lr_range = [float(lr_search_range_low), float(lr_search_range_high)]
    if lr_range[0] > lr_range[1]:
        lr_range[0], lr_range[1] = lr_range[1], lr_range[0]
    drop_range = [float(drop_rate_search_range_low), float(drop_rate_search_range_high)]
    if drop_range[0] > drop_range[1]:
        drop_range[0], drop_range[1] = drop_range[1], drop_range[0]
    aug_range = [float(aug_rate_search_range_low), float(aug_rate_search_range_high)]
    if aug_range[0] > aug_range[1]:
        aug_range[0], aug_range[1] = aug_range[1], aug_range[0]
    batch_range = [int(batch_size_search_range_low), int(batch_size_search_range_high)]
    if batch_range[0] > batch_range[1]:
        batch_range[0], batch_range[1] = batch_range[1], batch_range[0]

    search_key_ranges = {lr_name: lr_range, droprate_name: drop_range, aug_rate_name: aug_range, batch_size_name: batch_range}

    best_config, expected_best_step_num, _, _ = gp_update_hyperparameter_optimization(eval_fn=eval_fn, hyperparams=params, search_key_ranges=search_key_ranges, n=n,
                                                                save_results_to=save_results_to, m=m, metric_should_increase=metric_should_increase, metric_name=metric_name)

    return float(best_config[lr_name]), int(expected_best_step_num)
