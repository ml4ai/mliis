import glob
import hashlib
import os
import re
import time
import warnings
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import tensorflow as tf
from tensorflow import Session
from tensorflow.python import pywrap_tensorflow

# from data.input_fn import parse_example, _IMAGE_WIDTH


def hash_np_array(a: np.array) -> bytes:
    """Returns the sha-256 hash bytes of a stringified numpy array."""
    m = hashlib.sha256()
    m.update(a.tostring())
    return m.digest()


def count_examples_in_tfrecords(paths: List[str], count_unique_examples: bool = False, hash_only_on_image: bool = False, image_size: Optional[int] = None) -> int:
    # if image_size is None:
    #     image_size = _IMAGE_WIDTH
    if not isinstance(paths, list):
        paths = list(paths)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    # if count_unique_examples:
    #     examples = set()
    c = 0
    with tf.Session() as sess:
        for fn in paths:
            for record in tf.python_io.tf_record_iterator(fn, options=options):
                # if count_unique_examples:
                #     image, mask = parse_example(record, image_size)
                #     if hash_only_on_image:
                #         examples.add(hash_np_array(image.eval()))
                #     else:
                #         examples.add((hash_np_array(image.eval()), hash_np_array(mask.eval())))
                c += 1
    # if count_unique_examples:
    #     c = len(examples)
    return c


def count_unique_task_examples(dir: str, task_name: str) -> int:
    shards = glob.glob(os.path.join(dir, "*.tfrecord*"))
    shards = [x for x in shards if task_name in x]
    return count_examples_in_tfrecords(shards, count_unique_examples=True, hash_only_on_image=True)


def latest_checkpoint(checkpoint_dir: str, ckpt_prefix: str = "model.ckpt", return_relative: bool = True) -> str:
    if return_relative:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            text = f.readline()
        pattern = re.compile(re.escape(ckpt_prefix + "-") + r"[0-9]+")
        basename = pattern.findall(text)[0]
        return os.path.join(checkpoint_dir, basename)
    else:
        return tf.train.latest_checkpoint(checkpoint_dir)


def get_list_of_tensor_names(sess: tf.Session) -> List[str]:
    sess.as_default()
    graph = tf.get_default_graph()
    return [t.name for op in graph.get_operations() for t in op.values()]


def get_list_of_node_shapes(sess: tf.Session) -> List:
    with sess:
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        return [n._output_shapes for n in graph_def.node]


def mkdir(path):
    """
    Recursive create dir at `path` if `path` does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_fine_tuned_checkpoint(save_fine_tuned_checkpoint_dir: str, sess: Session, step: Optional[int] = None,
                               eval_sample_num: Optional[int] = None):
    if save_fine_tuned_checkpoint_dir is None:
        raise ValueError("Must specify directory in which to save fine-tuned checkpoints if saving them.")
    if eval_sample_num is not None:
        save_fine_tuned_checkpoint_dir = os.path.join(save_fine_tuned_checkpoint_dir, str(eval_sample_num))
    mkdir(save_fine_tuned_checkpoint_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_fine_tuned_checkpoint_dir, 'model.ckpt'), global_step=step)
    print("Saved fine-tuned checkpoint to {}.".format(save_fine_tuned_checkpoint_dir))


def get_training_set_hash_map(training_set: List[Tuple]) -> Dict[bytes, bytes]:
    """Returns a dict mapping sha-256 of image to sha-256 of corresponding mask."""
    hash_map = {}
    for pair in training_set:
        image_hash = hash_np_array(pair[0])
        mask_hash = hash_np_array(pair[1])
        hash_map[image_hash] = mask_hash
    return hash_map


def log_estimated_time_remaining(start_time, cur_step, total_steps, unit_name="meta-step"):
    elapsed = (time.time() - start_time) / 60.
    print("This {} took:".format(unit_name), elapsed, "minutes.")
    print('Estimated training hours remaining:%.4f' % ((total_steps - cur_step) * elapsed / 60.))
    return elapsed


def get_image_paths(list_of_paths):
    return [x for x in list_of_paths if is_image_file(x)]


def is_image_file(path):
    _, ext = os.path.splitext(path)
    if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', ".mat"]:
        return True
    else:
        return False


def cosine_similarity(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Computes the cosine similarity of the channels of a and b at every location of a and b.
    Assumes channels last.

    Args:
        a and b should be 3D tensors of the same shape. This func should be called with map_fn to operate on each example in the batch independently
    Returns:
        2D tensor of shape tf.shape(a)[0: 2]   # i.e. a scalar for all rows and cols
    """
    # return tf.tensordot(a, b) / (tf.nn.l2_normalize(a, -1) * tf.nn.l2_normalize(b, -1))
    normalize_a = tf.nn.l2_normalize(a, [0, 1])
    normalize_b = tf.nn.l2_normalize(b, [0, 1])
    print("normalize_a.shape")
    print(normalize_a.shape)
    out = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=-1)
    print("out.shape")
    print(out.shape)
    return out


def initialize_uninitialized_vars(session, list_of_variables=None):
    if list_of_variables is None:
        list_of_variables = tf.global_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    warnings.warn("Initializing the following variables: {}".format(uninitialized_variables))
    print("Initializing the following variables: {}".format(uninitialized_variables))
    session.run(tf.variables_initializer(uninitialized_variables))
    return uninitialized_variables


def validate_datasets(args, train_set, val_set, test_set):
    if not args.pretrained and not args.run_k_shot_learning_curves_experiment:
        assert len(train_set) > 0, "Training set must have examples."
    assert len(test_set) > 0, "Test set must have examples."
    if args.eval_val_tasks and val_set is not None:
        if len(val_set) == 0:
            raise ValueError("Val set has no tasks to evaluate")


def ci95(a: Union[List[float], np.ndarray]):
    """Computes the 95% confidence interval of the array `a`."""
    sigma = np.std(a)
    return 1.96 * sigma / np.sqrt(len(a))


def runtime_metrics(runtimes: List[Union[float, int]]):
    """runtimes is a list of time it takes to process one image"""
    ci = ci95(runtimes)
    return np.mean(runtimes), ci
