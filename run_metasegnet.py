"""
Meta-trains and evaluates image segmentation models.
"""
import copy
import datetime
import json
import logging
import os
import sys

import numpy as np
import random
import tensorflow as tf

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data.fss_1000_utils import FP_K_TEST_TASK_IDS
from data.pascal_5_constants import folds

from models.constants import SUPPORTED_MODELS
from models.efficientlab import EfficientLab
from models.reslinknet_model import ResLinkNet
from models.unet_model import UnetModel
from models.lr_schedulers import supported_learning_rate_schedulers
from meta_learners.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs, hyper_search_kwargs
from meta_learners.metaseg import read_dataset, read_fss_1000_dataset, read_fp_k_shot_dataset
from meta_learners.supervised_reptile.supervised_reptile.eval import evaluate_gecko, optimize_update_hyperparams, \
    run_k_shot_learning_curves_experiment
from meta_learners.supervised_reptile.supervised_reptile.train import train_gecko
from utils.util import latest_checkpoint, validate_datasets

logger = logging.getLogger(__name__)


def main():
    """
    Load data and train an image segmentation model on it.
    """
    verbose = True
    eval_train_tasks = True
    logger.info("Running image segmentation meta-learning...")
    start_time = datetime.datetime.now()
    print("Experiment started at: {}".format(start_time))

    args = argument_parser().parse_args()

    if args.optimize_update_hyperparms_on_val_set:
        assert args.num_val_tasks > 0, "Must specify number of validation tasks greater than 0 to optimize update hyperparams."

    random.seed(args.seed)
    global DATA_DIR
    DATA_DIR = args.data_dir

    print('Defining model architecture:')
    loss_name = model_kwargs(args)['loss_name']
    print('Using loss {}'.format(loss_name))
    args.model_name = args.model_name.lower()
    lr_scheduler = None
    if args.model_name == "efficientlab":
        restore_ckpt_dir = model_kwargs(args)["restore_ckpt_dir"]
        model = EfficientLab(**model_kwargs(args))
        initial_lr = args.learning_rate
        total_inner_steps = train_kwargs(args)["eval_inner_iters"]
        lr_scheduler_name = args.learning_rate_scheduler
        if supported_learning_rate_schedulers[lr_scheduler_name] is not None:
            if "step" in lr_scheduler_name:
                lr_sched_kwargs = {"decay_rate": args.step_decay_rate, "decay_after_n_steps": args.decay_after_n_steps}
            else:
                lr_sched_kwargs = {}
            lr_scheduler = supported_learning_rate_schedulers[lr_scheduler_name](initial_lr, total_inner_steps, **lr_sched_kwargs)
        else:
            lr_scheduler = None
    else:
        raise ValueError("model_name must be in {}".format(SUPPORTED_MODELS))
    print('{} instantiated.'.format(args.model_name))
    print("Model contains {} trainable parameters.".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    # Define the meta-learner:
    print("Meta-learning with algorithm:")
    if args.foml:
        print("FOMAML")
    else:
        print("Reptile")
    train_fn, evaluate_fn = train_gecko, evaluate_gecko

    # Get the meta-learning dataset. Each item in train_set and test_set is a task:
    print("Setting up meta-learning dataset")
    serially_eval_all_test_tasks = args.serially_eval_all_test_tasks
    if args.run_k_shot_learning_curves_experiment:
        test_set, test_task_names = read_fp_k_shot_dataset(DATA_DIR, image_size=args.image_size)
        val_set = None
        train_set = None
    elif args.fp_k_test_set:
        print("Holding out FP-k classes: {}".format(FP_K_TEST_TASK_IDS))
        dataset = read_fss_1000_dataset(DATA_DIR, num_val_tasks=args.num_val_tasks, test_task_ids=FP_K_TEST_TASK_IDS)
        train_set, val_set, test_set, train_task_names, val_task_names, test_task_names = dataset
        if len(val_set) == 0:
            val_set = None
        pascal_fold_num = None
    elif args.fss_1000:
        dataset = read_fss_1000_dataset(DATA_DIR, num_val_tasks=args.num_val_tasks)
        train_set, val_set, test_set, train_task_names, val_task_names, test_task_names = dataset
        if len(val_set) == 0:
            val_set = None
        pascal_fold_num = None
    else:
        print("Training and validating with PASCAL-5^i")
        pascal_fold_num = args.pascal_5_fold
        print("Holding out PASCAL-5^i Fold number: {}".format(pascal_fold_num))
        test_task_names = folds[pascal_fold_num]
        # Substrings of basenames of tfrecord files on disk inside DATA_DIR:
        print("Holding out test tasks: {}".format(test_task_names))

        train_set, val_set, test_set = read_dataset(DATA_DIR,
                                           test_task_names=test_task_names,
                                           num_train_shots=args.train_shots,
                                           image_size=args.image_size,
                                           num_test_shots=20)
        # serially_eval_all_test_tasks = False

    validate_datasets(args, train_set, val_set, test_set)

    if verbose:
        print('Found {} testing tasks:'.format(len(test_set)))
        for test_task in test_set:
            print("{}".format(test_task.name))
        if train_set is not None:
            print('Found {} training tasks:'.format(len(train_set)))
            for train_task in train_set:
                print("{}".format(train_task.name))

    with tf.Session() as sess:
        if args.model_name == "efficientlab":
            if restore_ckpt_dir is not None and not args.pretrained:
                print("Restoring from checkpoint {}".format(restore_ckpt_dir))
                model.restore_model(sess, restore_ckpt_dir, filter_to_scopes=[model.feature_extractor_name])
        if not args.pretrained:
            print("Meta-training...")

            if args.continue_training_from_checkpoint is not None:
                continue_training_from_checkpoint = latest_checkpoint(args.continue_training_from_checkpoint)
                print('Continuing meta-training from checkpoint: {}'.format(continue_training_from_checkpoint))
                tf.train.Saver().restore(sess, continue_training_from_checkpoint)
                model.variables_initialized = True

            _ = train_fn(sess, model, train_set, val_set or test_set, args.checkpoint, lr_scheduler=lr_scheduler,
                         augment=args.augment, **train_kwargs(args))
        else:
            if args.do_not_restore_final_layer_weights:
                print('Restoring from checkpoint: {}'.format(args.checkpoint))
                # model.restore_model(sess, args.checkpoint, filter_to_scopes=[model.feature_extractor_name, model.feature_decoder_name], filter_out_scope=model.final_layer_scope, convert_ckpt_to_rel_path=True)
                model.restore_model(sess, args.checkpoint, filter_out_scope=model.final_layer_scope, convert_ckpt_to_rel_path=True)
            else:
                checkpoint = latest_checkpoint(args.checkpoint)
                print('Restoring from checkpoint: {}'.format(checkpoint))
                tf.train.Saver().restore(sess, checkpoint)

        eval_kwargs = evaluate_kwargs(args)

        if args.optimize_update_hyperparms_on_val_set:
            print("Optimizing the update routine hyperparams on the val set")
            assert len(val_set) > 0, "Dev set has no tasks"
            save_fine_tuned_checkpoints_test = eval_kwargs["save_fine_tuned_checkpoints"]
            eval_kwargs["save_fine_tuned_checkpoints"] = False
            # TODO: thread through dropout search range low and high from CLI args.
            num_train_val_data_splits_to_sample_per_config = 1 if args.fss_1000 else 4
            estimated_lr, estimated_steps = optimize_update_hyperparams(sess, model, val_set,
                                    lr_scheduler=lr_scheduler,
                                    pascal_fold=pascal_fold_num, pascal_data_dir=DATA_DIR,
                                    serially_eval_all_tasks=serially_eval_all_test_tasks,
                                    num_configs_to_sample=args.num_configs_to_sample, save_dir=args.checkpoint,
                                   results_csv_name=args.uho_results_csv_name,
                                    num_train_val_data_splits_to_sample_per_config=num_train_val_data_splits_to_sample_per_config,
                                    max_steps=args.max_steps, min_steps=args.min_steps,
                                    b=args.uho_outer_iters, **eval_kwargs, **hyper_search_kwargs(args))
            eval_kwargs["save_fine_tuned_checkpoints"] = save_fine_tuned_checkpoints_test
            eval_kwargs["eval_inner_iters"] = estimated_steps
            eval_kwargs["lr"] = estimated_lr

            # Optionally meta-fine-tune on train + val sets here with optimal params, for small number of meta-iters
            # (e.g. 200, which is ~1.33 epochs on FSS-1000), and meta-step-final
            if args.meta_fine_tune_steps_on_train_val > 0:
                print("Fine-tuning meta-learned init for {} meta-steps with optimized hyperparameters.".format(args.meta_fine_tune_steps_on_train_val))
                training_params = train_kwargs(args)
                training_params["inner_iters"] = estimated_steps
                training_params["lr"] = estimated_lr
                training_params["meta_step_size"] = training_params["meta_step_size_final"]
                _ = train_fn(sess, model, train_set + val_set, test_set,
                             os.path.join(args.checkpoint, "fine-tuned_on_train_val_with_optimized_update_hyperparams"),
                             lr_scheduler=lr_scheduler, augment=args.augment, **training_params)

        del eval_kwargs["eval_tasks_with_median_early_stopping_iterations"]

        if args.run_k_shot_learning_curves_experiment:
            k_shot_eval_kwargs = copy.deepcopy(eval_kwargs)
            del k_shot_eval_kwargs["save_fine_tuned_checkpoints"]
            del k_shot_eval_kwargs["save_fine_tuned_checkpoints_dir"]
            run_k_shot_learning_curves_experiment(sess, model, test_set, lr_scheduler=lr_scheduler, iter_range=args.k_shot_iter_range, **k_shot_eval_kwargs)
        else:
            print('Evaluating {}-shot learning on training tasks.'.format(args.shots))
            if eval_train_tasks:
                save_fine_tuned_checkpoints_test = eval_kwargs["save_fine_tuned_checkpoints"]
                eval_kwargs["save_fine_tuned_checkpoints"] = args.save_fine_tuned_checkpoints_train
                mean_train_iou, _ = evaluate_fn(sess, model, train_set, visualize_predicted_segmentations=False,
                                                 lr_scheduler=lr_scheduler, serially_eval_all_tasks=False,
                                                 **eval_kwargs)
                eval_kwargs["save_fine_tuned_checkpoints"] = save_fine_tuned_checkpoints_test

            if args.eval_val_tasks:
                test_set = val_set
                test_task_names = val_task_names
                test_set_string = "val"
            else:
                test_set_string = "test"
            print('Evaluating {}-shot learning on meta-{} tasks.'.format(args.shots, test_set_string))
            mean_test_iou, task_name_iou_map = evaluate_fn(sess, model, test_set, visualize_predicted_segmentations=False,
                                        lr_scheduler=lr_scheduler, pascal_fold=pascal_fold_num, pascal_data_dir=DATA_DIR,
                                        serially_eval_all_tasks=serially_eval_all_test_tasks, **eval_kwargs)

            print("Evaluated meta-{} tasks:".format(test_set_string))
            print(task_name_iou_map)
            if eval_train_tasks:
                print("Mean meta-train IoU: {}".format(mean_train_iou))
            # Do NOT change this print (it's used to grep logs):
            print("Mean IoU over all meta-test tasks: {}".format(mean_test_iou))

            # Write results out:
            results_path = os.path.join(args.checkpoint, "meta-test_results.json")
            with open(results_path, "w") as f:
                json.dump(task_name_iou_map, f)
            print("Wrote results to {}".format(results_path))


    end_time = datetime.datetime.now()
    print("Experiment finished at: {}, taking {}".format(end_time, end_time - start_time))


if __name__ == '__main__':
    main()

