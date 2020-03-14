"""
Helpers for evaluating models.
"""
import itertools
import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from meta_learners.hyperparam_search import LEARNING_RATE_NAME, \
    DROPOUT_RATE_NAME, lr_droprate_aug_rate_batch_size_gp_search, AUG_RATE_NAME
from utils.util import ci95
from .reptile import Gecko, DEFAULT_ITER_RANGE
from meta_learners.variables import weight_decay


def evaluate_gecko(sess,
                   model,
                   dataset,
                   num_classes=1,
                   num_shots=5,
                   eval_inner_batch_size=5,
                   eval_inner_iters=50,
                   replacement=False,
                   num_samples=100,
                   transductive=False,
                   weight_decay_rate=1,
                   meta_fn=Gecko,
                   visualize_predicted_segmentations=True,
                   save_fine_tuned_checkpoints=False,
                   save_fine_tuned_checkpoints_dir: Optional[str] = None,
                   lr_scheduler=None,
                   lr=None,
                   augment=False,
                   serially_eval_all_tasks: bool = False,
                   aug_rate: Optional[float] = None,
                   ) -> Tuple[float, Dict[str, List[float]]]:
    """
    Evaluates an image segmentation model on a dataset.
    """
    print("Evaluating with eval_inner_iters: {}".format(eval_inner_iters))
    print("Evaluating with lr: {}".format(lr))

    if save_fine_tuned_checkpoints:
        print("Saving fine-tuned checkpoints to {}".format(save_fine_tuned_checkpoints_dir))
    if weight_decay_rate != 1:
        pre_step_op = weight_decay(weight_decay_rate)
    else:
        pre_step_op = None  # no need to just multiply all vars by 1.
    gecko = meta_fn(sess,
                    transductive=transductive,
                    pre_step_op=pre_step_op,
                    lr_scheduler=lr_scheduler,
                    augment=augment,
                    aug_rate=aug_rate)

    mean_ious = []
    task_iou_map = {}
    for i in range(num_samples):
        mean_iou, task_iou_map_i = gecko.evaluate(dataset, model.input_ph, model.label_ph,
                                  model.minimize_op, model.predictions,
                                  num_classes=num_classes, num_shots=num_shots,
                                  inner_batch_size=eval_inner_batch_size,
                                  inner_iters=eval_inner_iters, replacement=replacement,
                                  eval_all_tasks=serially_eval_all_tasks,
                                  visualize_predicted_segmentations=visualize_predicted_segmentations,
                                  save_fine_tuned_checkpoints=save_fine_tuned_checkpoints,
                                  save_fine_tuned_checkpoints_dir=save_fine_tuned_checkpoints_dir,
                                  eval_sample_num=i, is_training_ph=model.is_training_ph, lr_ph=model.lr_ph, lr=lr,)
        for key, val in task_iou_map_i.items():
            try:
                task_iou_map[key].append(val)
            except KeyError:
                task_iou_map[key] = [val]
        mean_ious.append(mean_iou)

    all_ious = list(itertools.chain(*task_iou_map.values()))
    ninety_five_perc_ci = ci95(all_ious)

    mean_of_all_task_splits = np.nanmean(all_ious)
    print("Mean of all {} task-splits: {} +/- 95% CI: {}".format(len(all_ious), mean_of_all_task_splits, ninety_five_perc_ci))

    print("{} NaN values out of total number of samples: {}".format(np.count_nonzero(np.isnan(mean_ious)), num_samples))
    mean_iou = np.nanmean(mean_ious)
    print("Mean of samples:")
    print("{} mean IoU, +/- 95% CI: {}".format(mean_iou, ninety_five_perc_ci))
    print("Evaluated with eval_inner_iters: {}".format(eval_inner_iters))
    print("Evaluated with lr: {}".format(lr))

    return mean_iou, task_iou_map


def optimize_update_hyperparams(sess,
                   model,
                   dataset,
                   num_classes=1,
                   num_shots=5,
                   eval_inner_batch_size=5,
                   eval_inner_iters=5,
                   replacement=False,
                   num_samples=100,
                   transductive=False,
                   weight_decay_rate=1,
                   meta_fn=Gecko,
                   save_fine_tuned_checkpoints=False,
                   save_fine_tuned_checkpoints_dir: Optional[str] = None,
                   lr_scheduler=None,
                   lr=None,
                   lr_search_range_low: float = 0.0005,
                   lr_search_range_high: float = 0.05,
                   drop_rate=None,
                   drop_rate_search_range_low: float = 0.1,
                   drop_rate_search_range_high: float = 0.8,
                   aug_rate: float = 0.5,
                   aug_rate_search_range_low: float = 0.5,
                   aug_rate_search_range_high: float = 0.5,
                   batch_size_search_range_low: int = 8,
                   batch_size_search_range_high: int = 8,
                   augment=False,
                   serially_eval_all_tasks: bool = True,
                   min_steps: int = 0,
                   max_steps: int = 80,
                   num_configs_to_sample=100,
                   num_train_val_data_splits_to_sample_per_config=1,
                   save_dir: Optional[str] = None,  # Dir in which to save results csv.
                   results_csv_name: str = "GP_val-set_hyper_param_search_results.csv",
                   eval_tasks_with_median_early_stopping_iterations: bool = False,
                   estimator: str = "GP",
                   ):
    """
    Evaluates an image segmentation model on a dataset.
    """
    supported_estimators = {"GP"}
    assert estimator in supported_estimators

    if save_fine_tuned_checkpoints:
        print("Saving fine-tuned checkpoints to {}".format(save_fine_tuned_checkpoints_dir))
    if weight_decay_rate != 1:
        pre_step_op = weight_decay(weight_decay_rate)
    else:
        pre_step_op = None  # no need to just multiply all vars by 1.
    gecko = meta_fn(sess,
                    transductive=transductive,
                    pre_step_op=pre_step_op,
                    lr_scheduler=lr_scheduler,
                    augment=augment)

    params = {"dataset": dataset, "input_ph": model.input_ph, "label_ph": model.label_ph,
              "minimize_op": model.minimize_op, "predictions": model.predictions, "num_classes": num_classes,
              "num_shots": num_shots, "inner_batch_size": eval_inner_batch_size,
              "replacement": replacement, "eval_all_tasks": serially_eval_all_tasks, "is_training_ph": model.is_training_ph,  # serially_eval_all_tasks
              "lr_ph": model.lr_ph, LEARNING_RATE_NAME: lr, "drop_rate_ph": model.final_layer_dropout_rate_ph, DROPOUT_RATE_NAME: drop_rate, AUG_RATE_NAME: aug_rate,
              "eval_tasks_with_median_early_stopping_iterations": eval_tasks_with_median_early_stopping_iterations, "min_steps": min_steps, "max_steps": max_steps,
              }

    eval_fn = gecko.evaluate_with_early_stopping

    if eval_tasks_with_median_early_stopping_iterations:
        print("Evaluating val-set tasks with median iterations returned by early stopping.")

    before_ext, ext = os.path.splitext(results_csv_name)
    before_ext += "_{}-shot".format(num_shots)
    results_csv_name = before_ext + ext
    if save_dir is not None:
        save_results_to = os.path.join(save_dir, results_csv_name)
    else:
        save_results_to = results_csv_name

    if estimator == "GP":
        best_lr, expected_best_step_num = lr_droprate_aug_rate_batch_size_gp_search(eval_fn, params,
                                                                    lr_search_range_low=lr_search_range_low,
                                                                    lr_search_range_high=lr_search_range_high,
                                                                    drop_rate_search_range_low=drop_rate_search_range_low,
                                                                    drop_rate_search_range_high=drop_rate_search_range_high,
                                                                    aug_rate_search_range_low=aug_rate_search_range_low,
                                                                    aug_rate_search_range_high=aug_rate_search_range_high,
                                                                    batch_size_search_range_low=batch_size_search_range_low,
                                                                    batch_size_search_range_high=batch_size_search_range_high,
                                                                    n=num_configs_to_sample,
                                                                    m=num_train_val_data_splits_to_sample_per_config,
                                                                    save_results_to=save_results_to)
    else:
        raise ValueError("Unsupported hyperparameter optimizer estimator {}. `estimator` must be in {}".format(estimator, supported_estimators))

    return best_lr, expected_best_step_num


DEFAULT_K_RANGE = [1, 5, 10, 50, 100, 200, 400]

def run_k_shot_learning_curves_experiment(sess,
                   model,
                   dataset,
                   num_classes=1,
                   num_shots=5,
                   eval_inner_batch_size=8,
                   eval_inner_iters=5,
                   replacement=False,
                   num_samples=100,
                   transductive=True,
                   weight_decay_rate=1,
                   meta_fn=Gecko,
                   lr_scheduler=None,
                   lr=None,
                   augment=True,
                   aug_rate: float = 0.5,
                   csv_outpath="k-shot-results.csv",  # None,
                   iter_range=DEFAULT_ITER_RANGE,
                   ):
    print("Running k-shot learning curves experiment over k-ranges {} and dataset {}".format(DEFAULT_K_RANGE, [x.name for x in dataset]))
    if iter_range is None:
        iter_range = DEFAULT_ITER_RANGE
    print("Using iter range {}".format(iter_range))

    gecko = meta_fn(sess,
                    transductive=transductive,
                    pre_step_op=weight_decay(weight_decay_rate),
                    lr_scheduler=lr_scheduler,
                    augment=augment,
                    aug_rate=aug_rate)

    ks, results = gecko.evaluate_m_k_shot_ranges_all_tasks(tasks=dataset, k_range=DEFAULT_K_RANGE, m=num_samples, input_ph=model.input_ph,
                                                           label_ph=model.label_ph, minimize_op=model.minimize_op, predictions=model.predictions,
                                                           inner_batch_size=eval_inner_batch_size,
                                                           inner_iters=eval_inner_iters, replacement=replacement, is_training_ph=model.is_training_ph,
                                                           lr_ph=model.lr_ph, lr=lr, test_samples=20, iter_range=iter_range, aug_rate=aug_rate)

    print("k-shot learning curve results:")
    print("ks:")
    print(ks)
    print("IoUs")
    print(results)

    if csv_outpath is not None:
        df = pd.DataFrame({"k": ks, "mIoU": results})
        if not os.path.isfile(csv_outpath):
            df.to_csv(csv_outpath, index=False)
        else:
            df.to_csv(csv_outpath, mode="a", header=False)

        df.to_csv(csv_outpath, index=False)
    return ks, results
