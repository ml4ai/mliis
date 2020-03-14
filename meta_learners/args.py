"""
Command-line argument parsing.
"""

import argparse
from functools import partial

import tensorflow as tf

from meta_learners.hyperparam_search import SUPPORTED_SEARCH_ALGS
from models.constants import SUPPORTED_MODELS
from meta_learners.supervised_reptile.supervised_reptile.reptile import Reptile, FOML, Gecko, FOMLIS
from models.lr_schedulers import supported_learning_rate_schedulers


def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fine-tune-task', help='Fine-tune meta-learned init on specified task.',  type=str,
                        required=False, default=None)
    parser.add_argument('--fine-tuned-checkpoint', help='Directory to save fine-tuned checkpoint to', type=str,
                        required=False, default=None)
    parser.add_argument('--pretrained', help='Continue training or evaluate a pre-trained model.',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--classes', help='number of classes per inner task', default=1, type=int)
    parser.add_argument('--shots', help='number of examples per class at meta-test time', default=5, type=int)
    parser.add_argument('--train-shots', help='shots in a training batch', default=5, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=8, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=8, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true')
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-step-final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=5, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=8, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=4, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=200, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
    parser.add_argument('--weight-decay', help='weight decay rate', default=1, type=float)
    parser.add_argument('--transductive', help='evaluate all samples at once', action='store_true')
    parser.add_argument('--foml', help='use FOML instead of Reptile', action='store_true')
    parser.add_argument('--foml-tail', help='number of shots for the final mini-batch in FOML',
                        default=None, type=int)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    parser.add_argument('--n_unet_encoding_stacks', help = 'Number of U-net encoding stacks.', required=False, type=int,
                        default=4)
    parser.add_argument('--data-dir', help='Path to directory housing meta-learning data.')
    parser.add_argument('--loss_name', help='Name of the loss function to use. Should be cross_entropy, soft_iou, or bce_dice', default='cross_entropy')
    parser.add_argument('--save_fine_tuned_checkpoints', help="If speced, save fine-tuned weights for test-set tasks.", action='store_true')
    parser.add_argument('--save_fine_tuned_checkpoints_train', help="If speced, save fine-tuned weights for train-set tasks.",
                        action='store_true')
    parser.add_argument('--save_fine_tuned_checkpoints_dir', help="Directory in which to save fine-tuned weights during evaluation.", required=False, default='/tmp/checkpoints/fine-tuned')
    parser.add_argument('--model_name',
                        help="Name of the model architecture to meta-train. Must be in the set: {}.".format(SUPPORTED_MODELS), required=False,
                        default='efficientlab')
    parser.add_argument("--start_num_feature_maps_power", help="2 ** start_num_feature_maps_power will be the number of channels in the first layer", type=int, default=5)
    parser.add_argument("--restore_efficient_net_weights_from", help="path to dir to restore efficientnet weights from", type=str, default=None)
    parser.add_argument("--spatial_pyramid_pooling", help="Use AutoDeepLab style spatial pyramid pooling layers. On default, applies mobilenetV2 spp, which is just 1x1 concatenated with image-level feature.", action="store_true")
    parser.add_argument("--skip_decoding",
                        help="Use DeepLab v3+ style long skip connection and seperable convs in the decoder layer.",
                        action="store_true")
    parser.add_argument("--rsd", help="List of integers specifying the 1-indexed reduction endpionts from EfficientNet to input into the lightweight skip decoding layers of EfficientLab.", type=int, nargs="+")
    parser.add_argument("--feature_extractor_name", help="efficientnet-b0 or efficientnet-b3", type=str, default="efficientnet-b0")
    parser.add_argument("--learning_rate_scheduler", help="Inner loop learning rate scheduler. Should be one of {}".format(supported_learning_rate_schedulers.keys()), type=str, action="store", required=False, default="fixed")
    parser.add_argument("--step_decay_rate", type=float, required=False, default=0.5)
    parser.add_argument("--decay_after_n_steps", type=int, required=False, default=5)
    parser.add_argument("--l2", help="Applies l2 weight decay to all weights in network", action="store_true")
    parser.add_argument("--l1", help="Applies l1 weight decay to all weights in network", action="store_true")
    parser.add_argument("--darc1", help="Applies darc1 regularizer to final activations of network",
                        action="store_true")
    parser.add_argument("--augment", help="Apply augmentations to training data",
                        action="store_true")
    parser.add_argument("--final_layer_dropout_rate", help="Probability to dropout inputs at final layer.", type=float, default=0.0)
    parser.add_argument("--image_size", help="size of image in pixels. images assumed to square", type=int, default=320)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--continue_training_from_checkpoint", help="Continue training from this checkpoint", default=None)
    parser.add_argument("--fss_1000", help="Train and val with the FSS-1000 dataset.", action="store_true")
    parser.add_argument("--num_val_tasks", help="Number of validation tasks to held out in addition to the 240 test tasks.", type=int, default=0)
    parser.add_argument("--eval_val_tasks",
                        help="If speced, will run final validation procedures on val-set as opposed to test set.", action="store_true")
    parser.add_argument("--serially_eval_all_test_tasks", help="Evaluate all tasks in test set in serial. "
          "Set serially_eval_all_tasks to True when running large experiments and set --eval-samples to a small int. "
          "Code will evaluate `eval_samples * num_test_tasks` models at the end of meta-training", action="store_true")
    parser.add_argument("--optimize_update_hyperparms_on_val_set", help="Search over update procedure hyperparams on the val set."
                        "tasks, returning the best learning rate and expected best number of adaptation iterations.",
                        action="store_true")
    parser.add_argument("--num_configs_to_sample", help="Number of configurations to randomly sample and evaluate if optimizing update hyperparams", default=100, type=int)
    parser.add_argument("--meta_fine_tune_steps_on_train_val", help="Run meta-fine tuning on train-val after optimizing hyperparams on val set.", type=int, default=0, required=False)
    parser.add_argument("--uho_outer_iters", type=int, default=2)
    parser.add_argument("--lr_search_range_low", default=0.0005, type=float)
    parser.add_argument("--lr_search_range_high", default=0.05, type=float)
    parser.add_argument("--drop_rate_search_range_low", default=0.2, type=float)
    parser.add_argument("--drop_rate_search_range_high", default=0.2, type=float)
    parser.add_argument("--aug_rate_search_range_low", default=0.5, type=float)
    parser.add_argument("--aug_rate_search_range_high", default=0.5, type=float)
    parser.add_argument("--batch_size_search_range_low", default=8, type=int)
    parser.add_argument("--batch_size_search_range_high", default=8, type=int)

    parser.add_argument("--run_k_shot_learning_curves_experiment", action="store_true", help="If speced, will run the k-shot learning experiments, "
                                                                                             "evaluating a model across a range of k-shot examples.")
    parser.add_argument("--fp_k_test_set", help="Hold out the test task for the fp-k classes.", action="store_true")
    parser.add_argument("--disable_rsd_residual_connections", help="Do not use residual connections in rsd modules.", action="store_true")
    parser.add_argument("--do_not_restore_final_layer_weights", help="When restoring model from checkpoint, do not restore the final layer weights.", action="store_true")
    parser.add_argument("--eval_tasks_with_median_early_stopping_iterations", help="If this and hyperparam search provided, will eval all tasks with the median number of early stopping iters.", action="store_true")
    parser.add_argument("--min_steps", help="min inner iters to train for UHO.", type=int, default=0)
    parser.add_argument("--max_steps", help="max inner iters to train for UHO.", type=int, default=80)
    parser.add_argument("--k_shot_iter_range", help="List of iterations to evaluate each k-shot at if running k-shot learning curves experiment", nargs='+', type=int, required=False, default=None)
    parser.add_argument("--sample_foml_train_val_with_replacement", help="If true, will sample train set and val set of tail shots with replacement", action="store_true")
    parser.add_argument("--aug_rate", help="Probability to augment image mask pair", type=float, default=0.5)
    parser.add_argument("--uho_results_csv_name", help="Path to write hyperparam search results to.", type=str, default="val-set_hyper_param_search_results.csv")
    parser.add_argument("--uho_estimator", default="GP", type=str)
    return parser


def model_kwargs(parsed_args):
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    parsed_args.model_name = parsed_args.model_name.lower()
    if parsed_args.model_name not in SUPPORTED_MODELS:
        raise ValueError("Model name must be in the set: {} but is {}".format(SUPPORTED_MODELS, parsed_args.model_name))
    res = {'learning_rate': parsed_args.learning_rate}
    if parsed_args.model_name == "efficientlab":
        restore_ckpt_dir = parsed_args.restore_efficient_net_weights_from
        res["restore_ckpt_dir"] = restore_ckpt_dir
        if parsed_args.spatial_pyramid_pooling:
            res["spatial_pyramid_pooling"] = True
        if parsed_args.skip_decoding:
            res["skip_decoding"] = True
        if parsed_args.rsd:
            res["rsd"] = parsed_args.rsd
        else:
            res["rsd"] = None
        res["feature_extractor_name"] = parsed_args.feature_extractor_name
        res["l2"] = parsed_args.l2
        res["l1"] = parsed_args.l1
        res["darc1"] = parsed_args.darc1
        res["final_layer_dropout_rate"] = parsed_args.final_layer_dropout_rate
        res["label_smoothing"] = parsed_args.label_smoothing
        if "dice" not in parsed_args.loss_name:
            res["dice"] = False
        if parsed_args.disable_rsd_residual_connections:
            res["disable_rsd_residual_connections"] = True
    if parsed_args.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    else:
        res['optimizer'] = partial(tf.train.AdamOptimizer, beta1=0)
    res['loss_name'] = parsed_args.loss_name
    res['n_unet_encoding_stacks'] = parsed_args.n_unet_encoding_stacks
    res['start_num_feature_maps_power'] = parsed_args.start_num_feature_maps_power
    res["n_rows"] = parsed_args.image_size
    res["n_cols"] = parsed_args.image_size
    return res


def hyper_search_kwargs(pa):
    assert pa.uho_estimator in SUPPORTED_SEARCH_ALGS, "{} not in supported hyperparam search algs {}".format(pa.uho_estimator, SUPPORTED_SEARCH_ALGS)
    res = {
        "lr_search_range_low": pa.lr_search_range_low,
        "lr_search_range_high": pa.lr_search_range_high,
        "drop_rate_search_range_low": pa.drop_rate_search_range_low,
        "drop_rate_search_range_high": pa.drop_rate_search_range_high,
        "aug_rate_search_range_low": pa.aug_rate_search_range_low,
        "aug_rate_search_range_high": pa.aug_rate_search_range_high,
        "batch_size_search_range_low": pa.batch_size_search_range_low,
        "batch_size_search_range_high": pa.batch_size_search_range_high,
        "estimator": pa.uho_estimator
    }
    return res

def optim_kwargs(parsed_args):
    res = {
        "learning_rate": parsed_args.learning_rate,
        "label_smoothing": parsed_args.label_smoothing
    }
    return res


# TODO: delete unused args
def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    if parsed_args.learning_rate_scheduler not in supported_learning_rate_schedulers:
        raise ValueError("Learning rate scheduler, {}, not in supported set: {}".format(parsed_args.learning_rate_scheduler, supported_learning_rate_schedulers.keys()))
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'meta_fn': _args_meta_fn(parsed_args),
        "aug_rate": parsed_args.aug_rate
    }


def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
        'save_fine_tuned_checkpoints': parsed_args.save_fine_tuned_checkpoints,
        'save_fine_tuned_checkpoints_dir': parsed_args.save_fine_tuned_checkpoints_dir,
        'meta_fn': _args_meta_fn(parsed_args),
        "augment": parsed_args.augment,
        'lr': None,  # Defined in model kwargs or estimated after meta-training.
        "eval_tasks_with_median_early_stopping_iterations": parsed_args.eval_tasks_with_median_early_stopping_iterations,
        "aug_rate": parsed_args.aug_rate
    }



def train_gecko_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_gecko(parsed_args),
    }


def evaluate_gecko_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
        'save_fine_tuned_checkpoints': parsed_args.save_fine_tuned_checkpoints,
        'save_fine_tuned_checkpoints_dir': parsed_args.save_fine_tuned_checkpoints_dir,
        'reptile_fn': _args_gecko(parsed_args)
    }


def _args_meta_fn(parsed_args):
    if parsed_args.foml:
        return partial(FOMLIS, train_shots=parsed_args.train_shots, tail_shots=parsed_args.foml_tail, sample_train_val_with_replacement=parsed_args.sample_foml_train_val_with_replacement)
    return Gecko


def _args_gecko(parsed_args):
    if parsed_args.foml:
        return partial(FOMLIS, train_shots=parsed_args.train_shots, tail_shots=parsed_args.foml_tail, sample_train_val_with_replacement=parsed_args.sample_foml_train_val_with_replacement)
    return Gecko


def _args_reptile(parsed_args):
    if parsed_args.foml:
        return partial(FOML, tail_shots=parsed_args.foml_tail)
    return Reptile

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
