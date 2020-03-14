"""
Trains an image segmentation model with SGD.

python joint_train.py --seperate_background_channel --data_dir joint_fewshot_shards_uint8_background_channel --augment --epochs 10 --steps_per_epoch 2 --batch_size 3 --val_batches 2 --sgd --l2 --final_layer_dropout_rate 0.2 --rsd 2 --restore_efficient_net_weights_from models/efficientnet/efficientnet-b0
python joint_train.py --fp_k_test_set --seperate_background_channel  --augment --epochs 10 --steps_per_epoch 2 --batch_size 3 --val_batches 2 --sgd --l2 --final_layer_dropout_rate 0.2 --rsd 2 --data_dir joint_fewshot_shards_uint8_background_channel_fp-k-test-set --restore_efficient_net_weights_from models/efficientnet/efficientnet-b0
python joint_train.py --test_on_val_set --seperate_background_channel --data_dir joint_fewshot_shards_uint8_background_channel_val-set/ --augment --epochs 10 --steps_per_epoch 2 --batch_size 3 --val_batches 2 --sgd --l2 --final_layer_dropout_rate 0.2 --rsd 2 --restore_efficient_net_weights_from models/efficientnet/efficientnet-b0

"""
import argparse
import os
import time
from functools import partial
from typing import List, Tuple, Optional, Callable

import numpy as np
import tensorflow as tf

from augmenters.np_augmenters import Augmenter, translate, fliplr, additive_gaussian_noise, exposure
from data.fss_1000_utils import TEST_TASK_IDS, TRAIN_TASK_IDS, FP_K_TEST_TASK_IDS
from joint_train.data.input_fn import TFRecordSegmentationDataset
from models.constants import SUPPORTED_MODELS
from models.efficientlab import EfficientLab
from meta_learners.supervised_reptile.supervised_reptile.reptile import Gecko
from utils.util import log_estimated_time_remaining

TRAIN_ID = "train"
VAL_ID = "val"
TEST_ID = "test"


def parse_args():
    """
    Returns an argument parser object for image segmentation training script.
    """

    parser = argparse.ArgumentParser(description="Train segmentation model via SGD.")

    # Data
    parser.add_argument("--data_dir", help="Path to folder containing tfrecords", required=True)
    parser.add_argument("--fp_k_test_set", help="Hold out the test task for the fp-k classes.", action="store_true")
    parser.add_argument("--test_on_val_set", help="If speced, will train on train shards and test on val shards, else will train on both train and val and test on test.", action="store_true")

    # Model
    parser.add_argument('--model_name',
                        help="Name of the model architecture to meta-train. Must be in the set: {}.".format(SUPPORTED_MODELS), required=False,
                        default='EfficientLab')
    parser.add_argument("--rsd", help="List of integers specifying the 1-indexed reduction endpionts from EfficientNet to input into the lightweight skip decoding layers of EfficientLab.", type=int, nargs="+")
    parser.add_argument("--feature_extractor_name", help="efficientnet-b0 or efficientnet-b3", type=str, default="efficientnet-b0")
    parser.add_argument("--image_size", help="size of image in pixels. images assumed to square", type=int, default=224)
    parser.add_argument("--seperate_background_channel", help="Whether or not to make a mutually exclusive background channel.", action='store_true', default=False)

    # Training
    parser.add_argument("--restore_efficient_net_weights_from", help="path to dir to restore efficientnet weights from", type=str, default=None)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    parser.add_argument('--loss_name', help='Name of the loss function to use. Should be cross_entropy, cross_entropy_dice, or ce_dice', default='ce_dice')
    parser.add_argument("--l2", help="Applies l2 weight decay to all weights in network", action="store_true")
    parser.add_argument("--augment", help="Apply augmentations to training data",
                        action="store_true")
    parser.add_argument("--final_layer_dropout_rate", help="Probability to dropout inputs at final layer.", type=float, default=0.0)
    parser.add_argument('--batch_size', help='Training batch size', default=64, type=int)
    parser.add_argument('--epochs', help='Number of training epochs', default=200, type=int)
    parser.add_argument("--steps_per_epoch", help="Number of gradient steps to take per epoch. If unspecified will be determined from batch size and number of examples.", type=int, default=None)
    parser.add_argument("--learning_rate", default=0.005, type=float)
    parser.add_argument("--final_learning_rate", default=5e-7, type=float)
    parser.add_argument("--label_smoothing", default=0.0, type=float)


    # Evaluation
    parser.add_argument("--val_batches", default=20, type=int)
    parser.add_argument('--pretrained', help='Evaluate a pre-trained model.',
                        action='store_true', default=False)
    parser.add_argument('--eval_interval', help='Training steps per evaluation', default=2, type=int)

    # Misc. config
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint', help='Checkpoint directory to write to (or restore from).', default='/tmp/model_checkpoint', type=str)
    return parser.parse_args()



def get_model_kwargs(parsed_args):
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    parsed_args.model_name = parsed_args.model_name.lower()
    if parsed_args.model_name not in SUPPORTED_MODELS:
        raise ValueError("Model name must be in the set: {}".format(SUPPORTED_MODELS))
    res = {'learning_rate': parsed_args.learning_rate}
    restore_ckpt_dir = parsed_args.restore_efficient_net_weights_from
    res["restore_ckpt_dir"] = restore_ckpt_dir
    if parsed_args.lsd:
        res["rsd"] = parsed_args.lsd
    res["feature_extractor_name"] = parsed_args.feature_extractor_name
    res["l2"] = parsed_args.l2
    res["final_layer_dropout_rate"] = parsed_args.final_layer_dropout_rate
    res["label_smoothing"] = parsed_args.label_smoothing
    if "dice" not in parsed_args.loss_name:
        res["dice"] = False
    if parsed_args.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    else:
        res['optimizer'] = partial(tf.train.AdamOptimizer, beta1=0)
    res['loss_name'] = parsed_args.loss_name
    res["n_rows"] = parsed_args.image_size
    res["n_cols"] = parsed_args.image_size
    return res


def after_step():
    """Function to be called after a step of gradient descent"""
    raise NotImplementedError


def after_epoch():
    """Function to be called after an epoch"""
    raise NotImplementedError


def get_train_test_shards_from_dir(data_dir, ext: str = ".tfrecord.gzip", test_on_val_set: bool = False):
    all_shards = os.listdir(data_dir)
    all_shards = [x for x in all_shards if ext in x]
    train_shards = [x for x in all_shards if TEST_ID not in x]
    test_shards = [x for x in all_shards if TRAIN_ID not in x]

    if test_on_val_set:
        train_shards = [x for x in train_shards if VAL_ID not in x]
        test_shards = [x for x in all_shards if VAL_ID in x]
        assert len(set(train_shards + test_shards)) == len(all_shards) - len([x for x in all_shards if TEST_ID in x])
    else:
        assert len(set(train_shards + test_shards)) == len(all_shards)

    assert len(set(test_shards).intersection(set(train_shards))) == 0
    return [os.path.join(data_dir, x) for x in train_shards], [os.path.join(data_dir, x) for x in test_shards]


def get_training_data(data_dir: str, num_classes: int, batch_size: int, image_size: int, ext: str = ".tfrecord.gzip", augment:bool = False, seperate_background_channel: bool = True, test_on_val_set: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
    train_shards, test_shards = get_train_test_shards_from_dir(data_dir, ext, test_on_val_set=test_on_val_set)

    if augment:
        if seperate_background_channel:
            mask_filled_translate = partial(translate, mask_fill=[1] + [0] * num_classes)
        else:
            mask_filled_translate = partial(translate, mask_fill=[0] * num_classes)

        augmenter = Augmenter(aug_funcs=[mask_filled_translate, fliplr, additive_gaussian_noise, exposure])
    else:
        augmenter = None
    dataset = TFRecordSegmentationDataset(tfrecord_paths=train_shards, image_width=image_size, mask_channels=num_classes, augmenter=augmenter, seperate_background_channel=seperate_background_channel)
    dataset, ds_init_op = dataset.make_dataset(batch_size)
    return dataset, ds_init_op


def train(sess: tf.Session, model: EfficientLab, dataset_init_op: tf.Operation, epochs: int, steps_per_epoch: int, images, masks, save_dir: str, lr_fn: Callable, restore_ckpt_dir: Optional[str] = None, val_batches: int = 20, save_checkpoint_every_n_epochs: int = 2, time_deadline=None, max_checkpoints_to_keep: int = 2, eval_interval: int = 2, report_allocated_tensors_on_oom: bool = True):
    """

    Args:
        sess:
        model:
        dataset_init_op:
        epochs:
        steps_per_epoch:
        images:
        masks:
        save_dir:
        lr_fn: A function that takes in the epoch number and returns the learning rate. For constant, learning rate, define a lambda: lr_fn = lambda i: lr
        val_batches: Number of batches to evaluate at the end of each epoch
        save_checkpoint_every_n_epochs:
        time_deadline:
        max_checkpoints_to_keep:

    Returns:

    """
    assert isinstance(epochs, int)
    assert isinstance(steps_per_epoch, int)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("Logging to {}".format(save_dir))

    saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

    if restore_ckpt_dir is not None:
        print("Restoring from checkpoint {}".format(restore_ckpt_dir))
        model.restore_model(sess, restore_ckpt_dir, filter_to_scopes=[model.feature_extractor_name])

    try:
        if not model.variables_initialized:
            print("Initializing variables.")
            tf.global_variables_initializer().run()
            sess.run(tf.global_variables_initializer())
    except AttributeError:
        print("Model does not explicitly track whether variable initialization has already been run on the graph at attribute .variables_initialized.")
        print("Initializing variables.")
        tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())

    print("Training...")
    sess.run(dataset_init_op)

    print("Saving graph definition to {}.".format(save_dir))
    saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=0)
    tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)

    if report_allocated_tensors_on_oom:
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    else:
        run_opts = None

    ious = []
    for i in range(epochs):
        start_time = time.time()
        print('Epoch: ', i)
        lr = lr_fn(i)
        print("lr: ", lr)
        for _ in range(steps_per_epoch):
            try:
                _ = sess.run(model.minimize_op, feed_dict={model.lr_ph: lr}, options=run_opts)
            except tf.errors.OutOfRangeError:
                sess.run(dataset_init_op, options=run_opts)
        print("Finished epoch {} with {} steps.".format(i, steps_per_epoch))
        epoch_minutes = log_estimated_time_remaining(start_time, i, epochs, unit_name="epoch")
        iters_per_sec = steps_per_epoch / (epoch_minutes * 60)
        print("Iterations per second: {}".format(iters_per_sec))

        if i % eval_interval == 0:
            # TODO implement val set accuracy callback
            print("Validating")
            iou, loss = iou_callback(sess, model, val_batches, run_opts)
            print("Loss: {}".format(loss))
            print("IoU on epoch {} estimated on {} batches:".format(i, val_batches))
            print(iou)
            ious.append(iou)

        if i % save_checkpoint_every_n_epochs == 0 or i == epochs - 1:
            print("Saving checkpoint to {}.".format(save_dir))
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)

        if time_deadline is not None and time.time() > time_deadline:
            break

    print("Training complete. History:")
    print("Train set Intersection over Union (IoU):")
    print(ious)


def iou_callback(sess, model: EfficientLab, val_batches, run_opts):
    ious = []
    losses = []
    for _ in range(val_batches):
        images, preds, labels, loss = sess.run([model.input_ph, model.predictions, model.label_ph, model.loss], options=run_opts, feed_dict={model.is_training_ph: False})
        # viz(images, preds, labels)
        ious.append(compute_iou_metric(preds, labels))
        losses.append(loss)
    iou = np.nanmean(ious)
    loss = np.nanmean(losses)
    return iou, loss


def compute_iou_metric(predictions: np.ndarray, labels: np.ndarray):
    assert len(predictions) == len(labels)
    assert len(predictions.shape) == 4
    # Pass prediction and label arrays to _iou:
    iou = [Gecko._iou(predictions[i], labels[i], class_of_interest_channel=None) for i in range(predictions.shape[0])]
    iou = np.nanmean(iou)
    return iou


def viz(images, preds, labels):
    from utils.debug_tf_dataset import plot_mask
    import matplotlib.pyplot as plt

    images = images / 255.

    if len(images.shape) == 4:
        for j in range(images.shape[0]):
            print("image")
            plt.figure(j)
            plt.imshow(images[j])
            plt.show()
            print("label mask")
            mask_j = labels[j]
            k = plot_mask(mask_j, j + 1)
            print("predicted mask")
            pred = preds[j]
            plot_mask(pred, j + 2, channel_index=k)
    else:
        plt.figure(0)
        plt.imshow(images)
        plt.show()
        plot_mask(labels, 1)


def main():
    # Reference: https://github.com/SMHendryx/tf-segmentation-trainer/blob/master/train.py
    start = time.time()
    # Args:
    args = parse_args()
    data_dir = args.data_dir
    learning_rate = args.learning_rate
    final_learning_rate = args.final_learning_rate
    epochs = args.epochs

    #all_classes, train_classes = get_classes_from_dir(data_dir, ext=".tfrecord.gzip")

    train_classes, test_classes = TRAIN_TASK_IDS, TEST_TASK_IDS

    all_classes = sorted(list(train_classes + test_classes))

    if args.fp_k_test_set:
        test_classes = FP_K_TEST_TASK_IDS
        train_classes = [x for x in all_classes if x not in test_classes]

    assert len(set(test_classes).intersection(set(train_classes))) == 0, "train-test class names overlap"
    assert len(train_classes + test_classes) == len(set(all_classes))

    num_classes = len(all_classes)
    next_element, dataset_init_op = get_training_data(data_dir, num_classes=num_classes, batch_size=args.batch_size, image_size=args.image_size, augment=args.augment, seperate_background_channel=args.seperate_background_channel, test_on_val_set=args.test_on_val_set)
    images = next_element[0]
    masks = next_element[1]

    model_kwargs = get_model_kwargs(args)
    restore_ckpt_dir = model_kwargs["restore_ckpt_dir"]
    model = EfficientLab(images=images, labels=masks, n_classes=num_classes, seperate_background_channel=args.seperate_background_channel, binary_iou_loss=False, **model_kwargs)

    if args.steps_per_epoch is None:
        steps_per_epoch = int(760 * 10 // args.batch_size)
    else:
        steps_per_epoch = args.steps_per_epoch

    def lr_fn(i, epochs=epochs, initial_lr=learning_rate, final_lr=final_learning_rate):
        frac_done = i / epochs
        cur_lr = frac_done * final_lr + (1 - frac_done) * initial_lr
        return cur_lr

    with tf.Session() as sess:
        train(sess, model, dataset_init_op, args.epochs, steps_per_epoch=steps_per_epoch, save_dir=args.checkpoint,
              lr_fn=lr_fn, val_batches=args.val_batches, images=images, masks=masks, eval_interval=args.eval_interval, restore_ckpt_dir=restore_ckpt_dir)

    print("Finished training")
    end = time.time()
    print("Experiment took {} hours".format((end - start) / 3600.))


if __name__ == '__main__':
    main()
