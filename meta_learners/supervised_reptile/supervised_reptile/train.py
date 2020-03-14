"""
Training helpers for supervised meta-learning.
"""

import os
import time

import numpy as np
import tensorflow as tf
from typing import Optional

from utils.util import log_estimated_time_remaining
from .reptile import Gecko
from meta_learners.variables import weight_decay


# pylint: disable=R0913,R0914
def train_gecko(sess,
                model,
                train_set,
                test_set,
                save_dir,
                num_classes=5,
                num_shots=5,
                inner_batch_size=5,
                inner_iters=20,
                replacement=False,
                meta_step_size=0.1,
                meta_step_size_final=0.1,
                meta_batch_size=1,
                meta_iters=10000,
                eval_inner_batch_size=5,
                eval_inner_iters=50,
                eval_interval=10,
                weight_decay_rate=1,
                time_deadline=None,
                train_shots=None,
                transductive=False,
                meta_fn=Gecko,
                log_fn=print,
                save_checkpoint_every_n_meta_iters=100,
                max_checkpoints_to_keep=2,
                augment=False,
                lr_scheduler=None,
                lr=None,
                save_best_seen=False,
                num_tasks_to_eval=100,
                aug_rate: Optional[float]=None):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

    if save_best_seen:
        best_save_dir = os.path.join(save_dir, "best_eval")
        if not os.path.exists(best_save_dir):
            os.mkdir(best_save_dir)
        best_saver = tf.train.Saver(max_to_keep=1)
    best_eval_iou = -np.inf

    if weight_decay_rate != 1:
        pre_step_op = weight_decay(weight_decay_rate)
    else:
        pre_step_op = None  # no need to just multiply all vars by 1.
    reptile = meta_fn(sess,
                      transductive=transductive,
                      pre_step_op=pre_step_op, lr_scheduler=lr_scheduler, augment=augment, aug_rate=aug_rate)
    iou_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('IoU', iou_ph)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
    try:
        if not model.variables_initialized:
            print("Initializing variables.")
            tf.global_variables_initializer().run()
            sess.run(tf.global_variables_initializer())
    except AttributeError:
        print("Model does not explicitly track whether variable initialization has already been run on the graph.")
        print("Initializing variables.")
        tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())


    for i in range(meta_iters):
        begin_time = time.time()
        print('Reptile training step {} of {}'.format(i + 1, meta_iters))
        frac_done = i / meta_iters
        print('{} done'.format(frac_done))
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        print("Current meta-step size: {}".format(cur_meta_step_size))
        reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                           num_classes=num_classes, num_shots=(train_shots or num_shots),
                           inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                           replacement=replacement,
                           meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size, lr_ph=model.lr_ph, lr=lr,)
        # call Gecko.evaluate to track progress:
        if i % eval_interval == 0:
            print('Evaluating training performance.')
            # track accuracy with mean intersection over union:
            mean_ious = []
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                mean_iou, _ = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                           model.minimize_op, model.predictions,
                                           num_classes=num_classes, num_shots=num_shots,
                                           inner_batch_size=eval_inner_batch_size,
                                           inner_iters=eval_inner_iters, replacement=replacement,
                                            eval_all_tasks=False,
                                            num_tasks_to_sample=num_tasks_to_eval,
                                            save_fine_tuned_checkpoints=False, is_training_ph=model.is_training_ph,
                                            lr_ph=model.lr_ph)
                summary = sess.run(merged, feed_dict={iou_ph: mean_iou})
                writer.add_summary(summary, i)
                # Log the learning rate:
                summary = tf.Summary(value=[tf.Summary.Value(tag="meta_step_size", simple_value=cur_meta_step_size)])
                writer.add_summary(summary, i)
                writer.flush()
                mean_ious.append(mean_iou)
            log_fn('Train step %d: train=%f test=%f' % (i, mean_ious[0], mean_ious[1]))

            if save_best_seen and mean_ious[1] > best_eval_iou:
                best_eval_iou = mean_ious[1]
                print("Highest test-set evaluation IoU seen at step {}: {}".format(i, best_eval_iou))
                print("Saving checkpoint to {}.".format(best_save_dir))
                best_saver.save(sess, os.path.join(best_save_dir, 'model.ckpt'), global_step=i)

        if i % save_checkpoint_every_n_meta_iters == 0 or i == meta_iters - 1:  # save checkpoint every n (should be 100) meta-iters and final meta-iter
            print("Saving checkpoint to {}.".format(save_dir))
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break
        log_estimated_time_remaining(begin_time, i, meta_iters)
    return reptile
