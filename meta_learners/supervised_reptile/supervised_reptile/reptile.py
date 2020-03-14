"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""
import copy
import os
import random
from typing import Optional, Tuple, List, Dict, Union
import warnings

import numpy as np
import tensorflow as tf

from augmenters.np_augmenters import Augmenter
from meta_learners.hyperparam_search import EarlyStopper
from meta_learners.metaseg import _sample_mini_image_segmentation_dataset, _mini_batches, \
    _split_train_test_segmentation, DEFAULT_NUM_TEST_EXAMPLES, _sample_train_test_segmentation_with_replacement
from utils.viz import plot_mask_on_image, savefig_mask_on_image
from meta_learners.variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                                     VariableState)

from utils.util import save_fine_tuned_checkpoint

DEFAULT_ITER_RANGE = [1, 5, 10, 25, 50, 100, 200]

class Gecko:
    """
    A meta-learning session for image segmentation that extends Reptile.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None, lr_scheduler=None, augment: bool = False, aug_rate: Optional[float] = None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

        self.eval_sample_number = 0

        # a learning rate scheduler class with a .cur_lr method
        self.lr_scheduler = lr_scheduler
        if augment:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None
        self.aug_rate = aug_rate
        print("Augmentation rate {}".format(self.aug_rate))

        if self._transductive:
            print("Using transduction in meta-learning.")
        else:
            print("Not using transduction in meta-learning.")

        # if self._pre_step_op:
        #     print("Using pre meta-step op:")
        #     print(self._pre_step_op)

        self.meta_fn = "Reptile"

        print("Reptile meta-learning session instantiated.")

    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size,
                   lr_ph=None,
                   lr=None,
                   verbose=False,):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of segmentation classes. Currently ignored.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
          lr_ph: learning rate placeholder for schedule learning rates
        """
        # Hardcode binary Gecko:
        num_classes = 1

        old_vars = self._model_state.export_variables()
        new_vars = []
        for _ in range(meta_batch_size):
            if verbose:
                print('Sampling new task.')
            mini_dataset = _sample_mini_image_segmentation_dataset(self.session, dataset, num_classes, num_shots)
            for i, batch in enumerate(_mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement, augmenter=self.augmenter)):
                if verbose:
                    print('Sampling new mini_batch.')
                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                if (lr_ph is not None) and (lr is not None):
                    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                             lr_ph: lr})
                if (lr_ph is not None) and (self.lr_scheduler is not None):
                    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                             lr_ph: self.lr_scheduler.cur_lr(cur_step=i)})
                else:
                    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement,
                 eval_all_tasks=False,
                 num_tasks_to_sample=1,
                 test_shots=DEFAULT_NUM_TEST_EXAMPLES,
                 visualize_predicted_segmentations=False,
                 verbose=False,
                 save_fine_tuned_checkpoints=False,
                 save_fine_tuned_checkpoints_dir: Optional[str] = None,
                 eval_sample_num: Optional[int] = None,
                 is_training_ph: Optional[tf.Tensor]=None,
                 lr_ph: Optional[tf.Tensor] = None,
                 lr: Optional[float] = None,
                 drop_rate_ph: Optional[tf.Tensor] = None,
                 drop_rate: Optional[float] = None,
                 aug_rate: Optional[float] = None,
                 ) -> Tuple[float, Dict[str, float]]:
        """
        Run a single evaluation of training the image segmentation model on new tasks.

        Samples a few-shot learning task and measures
        performance with mean Intersection over Union (IoU)

        Args:
          dataset: a list of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of floating point image mask/label scores.
          num_classes: number of data classes to sample. Parameter is currently ignored.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          eval_all_tasks: evaluate a few shot problem for all tasks in `dataset`
          test_shots: number of images to evaluate segmentation performance on.
          lr_ph: learning rate placeholder for scheduling the learning rate in the inner loop

        Returns:
          Mean Intersection over Union
        """
        print("Evaluating {} meta-learning.".format(self.meta_fn))

        if aug_rate is None:
            aug_rate = self.aug_rate

        if eval_all_tasks:
            sampled_tasks = dataset
        else:
            random.shuffle(dataset)
            sampled_tasks = dataset[:num_tasks_to_sample]

        print("Evaluating {} {}-shot tasks.".format(len(sampled_tasks), num_shots))

        task_names = []
        ious = []
        task_iou_map = {}
        for i, sampled_task in enumerate(sampled_tasks):
            sampled_task, task_name = _sample_mini_image_segmentation_dataset(self.session, [sampled_task], num_classes,
                                                                               num_shots + test_shots,
                                                                               return_task_name=True)
            task_names.append(task_name)
            print("Evaluating {}".format(task_name))

            train_set, test_set = _split_train_test_segmentation(sampled_task, test_shots)  # , test_train_test_split=True)

            task_iou = self._evaluate(
                 train_set=train_set,
                 test_set=test_set,
                 input_ph=input_ph,
                 label_ph=label_ph,
                 minimize_op=minimize_op,
                 predictions=predictions,
                 inner_batch_size=inner_batch_size,
                 inner_iters=inner_iters,
                 replacement=replacement,
                 visualize_predicted_segmentations=visualize_predicted_segmentations,
                 verbose=verbose,
                 save_fine_tuned_checkpoints=save_fine_tuned_checkpoints,
                 save_fine_tuned_checkpoints_dir=save_fine_tuned_checkpoints_dir,
                 eval_sample_num=eval_sample_num,
                 is_training_ph=is_training_ph,
                 lr_ph=lr_ph,
                 lr=lr,
                 task_name=task_name,
                 drop_rate_ph=drop_rate_ph,
                 drop_rate=drop_rate,
                 aug_rate=aug_rate,
                 )
            ious.append(task_iou)
            task_iou_map[task_name] = task_iou

        mean_iou_score = np.nanmean(ious)
        print("Evaluated {} task/s".format(len(sampled_tasks)))
        print('Mean IoU from train on {} images and evaluate on {} test images: {}'.format(num_shots, test_shots,
                                                                                           mean_iou_score))
        return mean_iou_score, task_iou_map

    def _evaluate(self,
                 train_set,
                 test_set,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 inner_batch_size,
                 inner_iters,
                 replacement,
                 visualize_predicted_segmentations=False,
                 verbose=False,
                 save_fine_tuned_checkpoints=False,
                 save_fine_tuned_checkpoints_dir: Optional[str] = None,
                 eval_sample_num: Optional[int] = None,
                 is_training_ph: Optional[tf.Tensor]=None,
                 lr_ph: Optional[tf.Tensor] = None,
                 lr: Optional[float] = None,
                 drop_rate_ph: Optional[tf.Tensor] = None,
                 drop_rate: Optional[float] = None,
                 aug_rate: Optional[float] = None,
                 task_name: Optional[str] = None,
                 ):
        """Evaluates a single task's train-test split."""
        old_vars = self._full_state.export_variables()  # keep vars in memory to reimport later

        # Fine-tune to task:
        for inner_iter, batch in enumerate(_mini_batches(train_set, inner_batch_size, num_batches=inner_iters,
                                                         replacement=replacement, augmenter=self.augmenter, aug_rate=aug_rate)):
            if verbose:
                print('Training on batch in eval task.')
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)

            if (lr_ph is not None) and (lr is not None) and (drop_rate_ph is not None) and (drop_rate is not None):
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels, drop_rate_ph: drop_rate,
                                                 lr_ph: lr})
            elif (lr_ph is not None) and (lr is not None):
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                         lr_ph: lr})
            elif (lr_ph is not None) and (self.lr_scheduler is not None):
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                         lr_ph: self.lr_scheduler.cur_lr(cur_step=inner_iter)})
            else:
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})

        if save_fine_tuned_checkpoints:
            checkpoint_dir_to_save_to = os.path.join(save_fine_tuned_checkpoints_dir, task_name)
            save_fine_tuned_checkpoint(save_fine_tuned_checkpoint_dir=checkpoint_dir_to_save_to,
                                       sess=self.session, step=inner_iter,
                                       eval_sample_num=eval_sample_num)

        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions, is_training_ph, task_name=task_name)

        # TODO: delete viz
        if visualize_predicted_segmentations:
            # Visualize data right before it goes into final IOU computation
            for j in range(len(test_preds)):
                image_i = test_set[j][0]
                label_i = test_set[j][1]
                predicted_mask_i = test_preds[j]
                print("Background label")
                plot_mask_on_image(image_i, label_i[:, :, 0])
                print("Class of interest label")
                plot_mask_on_image(image_i, label_i[:, :, 1])
                print("Prediction of class of interest")
                plot_mask_on_image(image_i, predicted_mask_i[:, :, 1])
                if j > 1:
                   break

        # Pass prediction and label arrays to _iou:
        class_iou = [self._iou(test_preds[j], test_set[j][1]) for j in range(len(test_preds))]
        class_iou = np.nanmean(class_iou)
        print("Mean task IoU: {}".format(class_iou))
        self._full_state.import_variables(old_vars)
        return class_iou

    def evaluate_with_early_stopping(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 min_steps,
                 max_steps,
                 replacement,
                 eval_all_tasks=False,
                 num_tasks_to_sample=20,
                 test_shots=DEFAULT_NUM_TEST_EXAMPLES,
                 is_training_ph: Optional[tf.Tensor]=None,
                 lr_ph: Optional[tf.Tensor] = None,
                 lr: Optional[float] = None,
                 drop_rate_ph: Optional[tf.Tensor] = None,
                 drop_rate: Optional[float] = None,
                 aug_rate: Optional[float] = None,
                 eval_tasks_with_median_early_stopping_iterations: bool = False,  # speed up by setting to False
                 ) -> Tuple[List[str], List[int], List[float]]:
        """
        Samples few-shot learning tasks, splitting into test and val sets and measures
        performance with mean Intersection over Union (IoU) on the val set.

        Returns:
          List of task names, list of best num steps, and IoU scores.
        """
        print("Evaluating {} meta-learning.".format(self.meta_fn))
        if eval_all_tasks:
            sampled_tasks = dataset
        else:
            random.shuffle(dataset)
            sampled_tasks = dataset[:num_tasks_to_sample]

        print("Evaluating {} {}-shot tasks.".format(len(sampled_tasks), num_shots))

        task_names = []
        ious = []
        if min_steps != max_steps:
            num_steps = []
            for i, sampled_task in enumerate(sampled_tasks):
                sampled_task, task_name = _sample_mini_image_segmentation_dataset(self.session, [sampled_task], num_classes,
                                                                                   num_shots + test_shots,
                                                                                   return_task_name=True)
                task_names.append(task_name)

                train_set, test_set = _split_train_test_segmentation(sampled_task, test_shots)

                # Fine-tune to task:
                best_n_steps, best_miou = self._early_stopping_learn(train_set, test_set, input_ph, label_ph, minimize_op,
                                                                     predictions, inner_batch_size, min_steps=min_steps,
                                                                     max_steps=max_steps,
                                                                     replacement=replacement, is_training_ph=is_training_ph, lr_ph=lr_ph, lr_scheduler=self.lr_scheduler, lr=lr,
                                                                     drop_rate_ph=drop_rate_ph, drop_rate=drop_rate, aug_rate=aug_rate)
                ious.append(best_miou)
                num_steps.append(best_n_steps)
            estimated_best_num_steps = int(np.median(num_steps))
        else:
            estimated_best_num_steps = min_steps
            num_steps = [estimated_best_num_steps] * len(sampled_tasks)

        # Conditionally evaluate all samples with np.median(num_steps):
        if eval_tasks_with_median_early_stopping_iterations or min_steps == max_steps:
            print("Estimated best number of steps {}".format(estimated_best_num_steps))
            mean_iou_score, task_iou_map = self.evaluate(dataset=sampled_tasks,
                         input_ph=input_ph,
                         label_ph=label_ph,
                         minimize_op=minimize_op,
                         predictions=predictions,
                         num_classes=num_classes,
                         num_shots=num_shots,
                         inner_batch_size=inner_batch_size,
                         inner_iters=estimated_best_num_steps,
                         replacement=replacement,
                         eval_all_tasks=eval_all_tasks,
                         num_tasks_to_sample=num_tasks_to_sample,
                         test_shots=test_shots,
                         is_training_ph=is_training_ph,
                         lr_ph=lr_ph,
                         lr=lr,
                         drop_rate_ph=drop_rate_ph,
                         drop_rate=drop_rate,
                         aug_rate=aug_rate,
                         )
            task_names = task_iou_map.keys()
            ious = task_iou_map.values()
        else:
            mean_iou_score = np.nanmean(ious)

        print("Evaluated {} task/s".format(len(sampled_tasks)))
        print('Mean IoU from train on {} images and evaluate on {} test images: {}'.format(num_shots, test_shots,
                                                                                           mean_iou_score))
        return task_names, num_steps, ious

    def evaluate_m_k_shot_ranges_all_tasks(self, tasks, k_range, m, input_ph, label_ph, minimize_op, predictions, inner_batch_size,
                                           inner_iters, replacement, is_training_ph=None, lr_ph=None, lr=None, test_samples=20, iter_range=DEFAULT_ITER_RANGE, aug_rate: float = 0.5):
        assert len(iter_range) == len(k_range)
        params = {"input_ph": input_ph, "label_ph": label_ph, "minimize_op": minimize_op, "predictions": predictions,
                  "inner_batch_size":inner_batch_size, "inner_iters": inner_iters, "replacement": replacement,
                  "is_training_ph": is_training_ph, "lr_ph": lr_ph, "lr": lr, "aug_rate": aug_rate}
        ks = []
        results = []
        for task in tasks:
            for _ in range(m):
                res = self.evaluate_k_shot_range(task, k_range=k_range, iter_range=iter_range, test_samples=test_samples, **params)
                print("k-shot results {}".format({k: r for k, r in zip(k_range, res)}))
                results.extend(res)
                ks.extend(k_range)
        return ks, results

    def evaluate_k_shot_range(self, task, k_range, iter_range=DEFAULT_ITER_RANGE, test_samples=20, early_stopping_min_val_samples=5, esimate_inner_iters_with_early_stoppping: bool = True, **params):
        """Evaluates k-shot learning results for a single task over a range of ks."""

        mious = []

        sampled_task, task_name = _sample_mini_image_segmentation_dataset(self.session, [task], num_classes=1,
                                                                          num_shots=max(k_range) + test_samples,
                                                                          return_task_name=True)
        training_examples, test_set = _split_train_test_segmentation(sampled_task, test_shots=test_samples)

        for i, k in enumerate(k_range):
            print("Evaluating {}-shot learning".format(k))
            train_set = training_examples[:k]

            if esimate_inner_iters_with_early_stoppping:
                # if k >= test_samples * 2:
                #     early_stopping_min_val_samples = test_samples

                if k >= early_stopping_min_val_samples * 2:
                    val_shots = int(0.2 * k)  # Use 20% of training dataset for validating against during ES
                    print("Split training dataset into {} train shots and {} val shots for early stopping to estimate number of steps.".format(k - val_shots, val_shots))
                    d_tr, d_val = _split_train_test_segmentation(train_set, test_shots=val_shots)
                    inner_iters, _ = self._early_stopping_learn(d_tr, d_val, min_steps=1, max_steps=500, **params)
                    params["inner_iters"] = inner_iters
            else:
                params["inner_iters"] = iter_range[i]
            task_iou = self._evaluate(train_set, test_set, **params)

            mious.append(task_iou)

        print("Evaluated task {} over k-range {}".format(task_name, k_range))

        return mious

    def _early_stopping_learn(self, train_set, val_set, input_ph, label_ph, minimize_op, predictions, inner_batch_size,
                              min_steps, max_steps, replacement, is_training_ph=None, lr_ph=None, lr_scheduler=None, lr=None,
                              drop_rate_ph=None, drop_rate=None, patience=50, inner_iters=None, aug_rate: Optional[float] = None):
        """Estimates number of steps to take when learning a new task."""
        del inner_iters
        old_vars = self._full_state.export_variables()  # keep vars in memory to reimport later
        if lr_scheduler is not None and lr is not None:
            raise ValueError("Only lr_scheduler or lr should be speced. Not both.")

        early_stopper = EarlyStopper(patience, min_steps=min_steps)
        for inner_iter, batch in enumerate(_mini_batches(train_set, inner_batch_size, num_batches=max_steps,
                                                         replacement=replacement, augmenter=self.augmenter, aug_rate=aug_rate)):
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            if (lr_ph is not None) and (lr is not None) and (drop_rate_ph is not None) and (drop_rate is not None):
                # print('Setting lr to {}'.format(lr))
                # print("Setting dropout rate to {}".format(drop_rate))
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels, drop_rate_ph: drop_rate,
                                                 lr_ph: lr})
            elif (lr_ph is not None) and (lr is not None):
                # print('Setting lr to {}'.format(lr))
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                 lr_ph: lr})
            elif (lr_ph is not None) and (lr_scheduler is not None):  # TODO: delete lr_scheduler for release
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                 lr_ph: lr_scheduler.cur_lr(cur_step=inner_iter)})
            else:  # Use hyperparam defaults:
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})

            # Eval:
            test_preds = self._test_predictions(train_set, val_set, input_ph, predictions, is_training_ph)

            # Pass prediction and label arrays to iou computation:
            ious = [self._iou(test_preds[j], val_set[j][1]) for j in range(len(test_preds))]
            miou = np.nanmean(ious)
            if not early_stopper.continue_training(miou, inner_iter + 1):
                break
        best_num_steps = early_stopper.best_num_steps()
        best_iou = early_stopper.best_metric()
        print("Best iteration found: {}, with mean-IoU {}".format(best_num_steps, best_iou))
        self._full_state.import_variables(old_vars)
        return best_num_steps, best_iou

    def _test_predictions(self, train_set, test_set, input_ph, predictions, is_training_ph=None, task_name: Optional[str] = None):
        """
        Run the operations to evaluate `predictions` tensor from the graph on the `test_set`.

        Args:
            train_set: training examples (for non-transductive eval)
            test_set: test examples to predict the labels of
            input_ph: input placeholder to the graph
            predictions: predictions tensor to evaluate
            is_training_ph: placeholder which will switch operations from training to testing modes.
        Returns:
            Prediction arrays
        """
        # To save predictions run in shell: export SAVE_PREDICTIONS=1
        try:
            DEBUG = bool(os.environ["SAVE_PREDICTIONS"])
        except KeyError:
            DEBUG = False
        if self._transductive:
            inputs, _ = zip(*test_set)
            if is_training_ph is None:
                res = self.session.run(predictions, feed_dict={input_ph: inputs})
            else:
                # Use estimated population mean and variances for batch norm by setting is_training_ph to False (which also turns off dropout):
                res = self.session.run(predictions, feed_dict={input_ph: inputs, is_training_ph: False})

            if DEBUG:
                for i, query_prediction in enumerate(zip(inputs, res)):
                    query, prediction = query_prediction
                    task_name = "" if task_name is None else task_name
                    save_path = "predictions/prediction_{}_{}.jpeg".format(task_name, i)
                    savefig_mask_on_image(query, prediction, save_path=save_path)
            return res
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            if is_training_ph is not None:
                predicted_mask = self.session.run(predictions, feed_dict={input_ph: inputs, is_training_ph: False})[-1]
            else:
                predicted_mask = self.session.run(predictions, feed_dict={input_ph: inputs})[-1]
            res.append(predicted_mask)
        return res

    @staticmethod
    def _iou(prediction: np.array, label: np.array, epsilon: float = 1e-7, class_of_interest_channel: Optional[Union[int, slice]] = 1, round_labels: bool = True):
        """
        Return the intersection over union score of two binary arrays. Intended for single images, not batches.
        Args:
            prediction: floating point array of predictions of shape [rows, columns, 2]
            label: boolean array of labels of shape [rows, columns, 2]
            epsilon: small floating point value to avoid divided by zero error in the event that there is no union (no predictions, no labels)
        """
        if len(prediction.shape) > 3:
            raise ValueError("Function is intended for single image masks, not batches.")
        if prediction.shape != label.shape:
            raise ValueError("prediction shape and label shape must be equal but are: {} and {} respectively.".format(prediction.shape, label.shape))
        if class_of_interest_channel is not None:
            prediction = prediction[:, :, class_of_interest_channel]
            label = label[:, :, class_of_interest_channel]

        prediction = np.round(prediction)
        if round_labels:
            label = np.round(label)

        intersection = np.logical_and(prediction, label)
        union = np.logical_or(label, prediction)
        return (np.sum(intersection) + epsilon) / (np.sum(union) + epsilon)


# Shaban et al. IoU metric code:
# https://github.com/lzzcd001/OSLSM/blob/master/OSLSM/os_semantic_segmentation/test.py
# the z(h)en function for measuring IOU
def measure(y_in, pred_in, thresh: float = .5):
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn


def iou_img(tp, fp, fn):
    return tp/float(max(tp+fp+fn, 1))


class FOMLIS(Gecko):
    """
    An implementation of "first-order MAML" (FOMAML) for image segmentation.

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """
    def __init__(self, *args, train_shots: Optional[int] = None, tail_shots: Optional[int] = None, sample_train_val_with_replacement: bool = False, **kwargs):
        """
        Create a first-order MAML session.

        Args:
          args: args for Reptile.
          tail_shots: if specified, this is the number of
            examples per class to reserve for the final
            mini-batch.
          kwargs: kwargs for Reptile.
        """
        super(FOMLIS, self).__init__(*args, **kwargs)
        self.train_shots = train_shots - tail_shots if tail_shots is not None else train_shots
        self.tail_shots = tail_shots
        self.sample_train_val_with_replacement = sample_train_val_with_replacement
        if sample_train_val_with_replacement:
            print("Sampling train val with replacement.")
        print("Specializing meta-learner to FOMAML.")

    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size,
                   verbose=False,
                   lr_ph=None,
                   lr=None,
                   ):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            if verbose:
                print('Sampling new task.')
            mini_dataset = _sample_mini_image_segmentation_dataset(self.session, dataset, num_classes,
                                                                   num_shots)
            mini_batches = self._mini_batches(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)

            for j, batch in enumerate(mini_batches):
                if verbose:
                    print('Sampling new mini_batch.')
                inputs, labels = zip(*batch)
                if j == inner_iters - 1:
                    last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                if (lr_ph is not None) and (lr is not None):
                    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels,
                                                             lr_ph: lr})
                else:
                    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement, augmenter=self.augmenter, aug_rate=self.aug_rate):
                yield value
            return
        if self.sample_train_val_with_replacement:
            train, tail = _sample_train_test_segmentation_with_replacement(mini_dataset, train_shots=self.train_shots, test_shots=self.tail_shots)
        else:
            train, tail = _split_train_test_segmentation(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement, augmenter=self.augmenter, aug_rate=self.aug_rate):
            yield batch
        yield tail


class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = self._model_state.export_variables()
        new_vars = []
        for _ in range(meta_batch_size):
            # Sample a task:
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            # loop through batches of the task:
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            # Append each task's learned parameters to new_vars:
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        # Average the updated model variables across new tasks in meta-step: (See eq. 5 (batch version) in Nichol et al. 2018 On First-Order Meta-Learning Algorithms)
        new_vars = average_vars(new_vars)
        # Subtract new_vars from old_vars, scale with meta_step_size (epsilon in paper), and then add to old_vars to take a gradient step:
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        train_set, test_set = _split_train_test(
            _sample_mini_dataset(dataset, num_classes, num_shots+1))
        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res


class FOML(Reptile):
    """
    A basic implementation of "first-order MAML" (FOML).

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """
    def __init__(self, *args, tail_shots=None, **kwargs):
        """
        Create a first-order MAML session.

        Args:
          args: args for Reptile.
          tail_shots: if specified, this is the number of
            examples per class to reserve for the final
            mini-batch.
          kwargs: kwargs for Reptile.
        """
        super(FOML, self).__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            mini_batches = self._mini_batches(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)
            for batch in mini_batches:
                inputs, labels = zip(*batch)
                last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch
        yield tail


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)


def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set


