"""
Loading the MetaSeg dataset.

To use these APIs, you will need a directory that
contains all of the labeled binary segmentation images inside of task-level tfrecords, e.g.:
meta_seg_data_dir/
    test/
        aeroplane_val.tfrecord.gzip
        ...
    train
"""

import os
import glob
import random
import warnings
from typing import List, Tuple, Union, Optional

import numpy as np
import tensorflow as tf

from augmenters.np_augmenters import Augmenter
from data import input_fn
from data.fss_1000_utils import split_train_test_tasks, get_fss_tasks, TEST_TASK_IDS
from utils.viz import savefig_mask_on_image
from utils.util import count_examples_in_tfrecords, hash_np_array

DEFAULT_NUM_TEST_EXAMPLES = 5
DEFAULT_K_SHOT_SET = [{"airliner", "aeroplane"}, {"bus"}, {"motorbike"}, {"potted_plant", "potted plant"}, {"television", "tvmonitor"}]


def read_dataset(data_dir:str,
                 test_task_names:List[str]=["table", "dog", "horse", "motorbike", "person"],  # pascal-5i fold 2
                 num_train_shots: int = 5,
                 num_test_shots: int = DEFAULT_NUM_TEST_EXAMPLES,
                 test_tfrecords_must_include:str = "_val.",
                 count_unique_examples: bool = False,
                 image_size: Optional[int] = None
    ) -> Tuple[List["BinarySegmentationTask"], Optional[List["BinarySegmentationTask"]], List["BinarySegmentationTask"]]:
    """
    Reads in a meta-learning image segmentation dataset.
    Args:
      data_dir: a directory containing tfrecords files for each semantic class.

    Returns:
      Tuple of (training_tasks, testing_tasks). Both items in the tuple are lists of BinarySegmentationTasks.
    """
    from data.pascal_5_constants import superset
    # Remove spaces:
    superset = [x.replace(" ", "") for x in superset]
    test_task_names = [test_task.replace(" ", "")
                       for test_task in test_task_names]
    training_tasks, test_tasks = [], []
    iterator = None

    shards = glob.glob(os.path.join(data_dir, "*.tfrecord*"))

    for task_name in superset:
        if task_name in test_task_names:
            suffix = test_tfrecords_must_include
            patterns = [task_name + suffix]
        else:
            suffix = "_train."
            # Get all the shards for training tasks:
            patterns = [task_name + suffix, task_name + test_tfrecords_must_include]

        task_shards = []
        for shard in shards:
            for pattern in patterns:
                if pattern in shard:
                    task_shards.append(shard)

        n_egs = count_examples_in_tfrecords(task_shards, count_unique_examples=count_unique_examples)
        print("{} examples in task {}".format(n_egs, task_name))
        if task_name in test_task_names:
            batch_size = n_egs
            # batch_size = num_train_shots + num_test_shots
        elif n_egs < (num_train_shots + num_test_shots):
            batch_size = n_egs
        else:
            # This is a meta-training task and there are more examples than requested training and validation shots
            batch_size = num_train_shots + num_test_shots
        task = BinarySegmentationTask(
            iterator=iterator,
            tfrecord_paths=task_shards,
            batch_size=batch_size,
            name=task_name,
            image_size=image_size)
        # Reinitialize all tasks using the same iterator.
        if not iterator:
            print("making new iterator in read_dataset for task: {}".format(task_name))
            iterator = task.iterator

        if task_name in test_task_names:
            test_tasks.append(task)
        else:
            training_tasks.append(task)

    return training_tasks, None, test_tasks


def read_fss_1000_dataset(data_dir: str,
                          num_val_tasks: int = 0,
                          num_test_tasks: int = 240,
                          test_task_ids: Optional[List[str]] = TEST_TASK_IDS,
                          count_unique_examples: bool = False,
                          image_size: Optional[int] = 224
                          ) -> Tuple[List["BinarySegmentationTask"], List["BinarySegmentationTask"], List["BinarySegmentationTask"],
                List[str], List[str], List[str]]:
    """
    Reads in the FSS-1000 meta-learning image segmentation dataset. Assumes each task is in a shard.
    Args:
        data_dir: a directory containing tfrecords files for each semantic class.

    Returns:
        Tuple of (train_tasks, val_tasks, test_tasks, train_task_names, val_task_names, test_task_names).
        First three objects are instances of BinarySegmentationTask.
    """
    verbose = False

    all_tasks = get_fss_tasks(data_dir)

    if test_task_ids is None:
        train_shards, test_shards = split_train_test_tasks(all_tasks, num_test_tasks)
    else:
        train_shards, test_shards = [], []
        for task in all_tasks:
            comparer = os.path.basename(task).replace(".tfrecord.gzip", "")
            if comparer in test_task_ids:
                test_shards.append(task)
            else:
                train_shards.append(task)
        assert all([os.path.basename(x).replace(".tfrecord.gzip", "") in test_task_ids for x in test_shards]), "Test shard not in test_task_ids"
        assert all([not os.path.basename(x).replace(".tfrecord.gzip", "") in test_task_ids for x in train_shards]), "Test set task found in train shards"

    train_shards, val_shards = split_train_test_tasks(train_shards, num_val_tasks, reproducbile_splits=True)

    print("{} training tasks, {} val tasks, {} test tasks.".format(len(train_shards), len(val_shards), len(test_shards)))

    train_tasks, val_tasks, test_tasks = [], [], []
    iterator = None

    print("Building FSS-1000 training task samplers...")
    train_task_names = []
    for task in train_shards:
        task_name = os.path.basename(task)
        train_task_names.append(task_name)
        batch_size = count_examples_in_tfrecords([task], count_unique_examples=count_unique_examples)
        if verbose:
            print("{} examples in task {}".format(batch_size, task_name))
        few_shot_seg_task = BinarySegmentationTask(
            iterator=iterator,
            tfrecord_paths=task,
            batch_size=batch_size,
            name=task_name,
            image_size=image_size,
            verbose=False)
        # Initialize all tasks using the same iterator.
        if not iterator:
            print("making new iterator in read_dataset for task: {}".format(task_name))
            iterator = few_shot_seg_task.iterator
        train_tasks.append(few_shot_seg_task)

    print("Building FSS-1000 val task samplers...")
    val_task_names = []
    for task in val_shards:
        task_name = os.path.basename(task)
        val_task_names.append(task_name)
        batch_size = count_examples_in_tfrecords([task], count_unique_examples=count_unique_examples)
        if verbose:
            print("Meta-val task: {}".format(task_name))
            print("{} examples in task {}".format(batch_size, task_name))
        few_shot_seg_task = BinarySegmentationTask(
            iterator=iterator,
            tfrecord_paths=task,
            batch_size=batch_size,
            name=task_name,
            image_size=image_size,
            verbose=False)
        val_tasks.append(few_shot_seg_task)

    print("Building FSS-1000 test task samplers...")
    test_task_names = []
    for task in test_shards:
        task_name = os.path.basename(task)
        test_task_names.append(task_name)
        batch_size = count_examples_in_tfrecords([task], count_unique_examples=count_unique_examples)
        if verbose:
            print("Meta-test task: {}".format(task_name))
            print("{} examples in task {}".format(batch_size, task_name))
        few_shot_seg_task = BinarySegmentationTask(
            iterator=iterator,
            tfrecord_paths=task,
            batch_size=batch_size,
            name=task_name,
            image_size=image_size,
            verbose=False)
        test_tasks.append(few_shot_seg_task)

    return train_tasks, val_tasks, test_tasks, train_task_names, val_task_names, test_task_names


def read_fp_k_shot_dataset(data_dir: str,
                           all_task_names = DEFAULT_K_SHOT_SET,
                           image_size: Optional[int] = 224
                           ) -> Tuple[List["BinarySegmentationTask"], List[str]]:
    """
    Reads in the FP-k-shot meta-learning image segmentation dataset, which contains FSS-1000 and PASCAL-5^i classes.
    Args:
        data_dir: a directory containing tfrecords files for each semantic class.
        all_task_names: list of set of synonyms defining the tasks.
    Returns:
        BinarySegmentationTask objects for the tasks in `tasks`.
    """
    verbose = True

    all_tasks = get_fss_tasks(data_dir)

    print("{} tasks found.".format(len(all_tasks)))

    test_tasks = []
    iterator = None

    print("Building k-shot-FSS-1000 test task samplers...")
    test_task_names = []
    for synonyms in all_task_names:
        task_shards = []
        task_globs = []
        for i, synonym in enumerate(synonyms):
            synonym = synonym.replace(" ", "")
            if i == 0:
                task_name = synonym
                print("Processing task: {}".format(task_name))
            syn_shards = [x for x in all_tasks if synonym in os.path.basename(x)]
            task_shards.extend(syn_shards)
            task_globs.append(os.path.join(data_dir, "{}*.tfrecord*".format(synonym)))

        print("task shards: {}".format(task_shards))

        test_task_names.append(task_name)
        batch_size = count_examples_in_tfrecords(task_shards)
        if verbose:
            print("{} examples in task {}".format(batch_size, task_name))
        few_shot_seg_task = BinarySegmentationTask(
            iterator=iterator,
            tfrecord_paths=task_globs,
            batch_size=batch_size,
            name=task_name,
            image_size=image_size,
            verbose=False)
        # Initialize all tasks using the same iterator.
        if not iterator:
            print("making new iterator in read_dataset for task: {}".format(task_name))
            iterator = few_shot_seg_task.iterator
        test_tasks.append(few_shot_seg_task)

    return test_tasks, test_task_names


class BinarySegmentationTask:
    """
    Segmentation maps for binary segmentations.
    Label dimensions are [n_row, n_col, 2] (one-hot encoding).
    """
    def __init__(self,
                 tfrecord_paths,
                 iterator=None,
                 batch_size=32,
                 seed=None,
                 name: str = None,
                 image_size: Optional[int] = None,
                 verbose: bool = False):
        self.tfrecord_paths = tfrecord_paths
        self.batch_size = batch_size


        if image_size is not None:
            dataset = input_fn.make_dataset(self.tfrecord_paths, self.batch_size, image_width=image_size)
        else:
            dataset = input_fn.make_dataset(self.tfrecord_paths, self.batch_size)
        if not iterator:
            print("making new iterator in BinarySegmentationTask.__init__ for task: {}".format(name))
            iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                       dataset.output_shapes)
        self.iterator = iterator
        self._initialization_op = self.iterator.make_initializer(dataset)
        self._next_element = self.iterator.get_next()

        self.name = name
        if verbose:
            print("BinarySegmentationTask for data {} will return batches of size {}".format(self.name, self.batch_size))

    def sample(self, sess, num_images, verbose=False) -> List[List[np.array]]:
        """
        Sample tuple of (image, label) from tfrecords
        Args:
            sess: tf session
        Returns:
            A sequence of (image, label) pairs
        """
        if num_images > self.batch_size:
            raise ValueError("Tried to sample {} examples.Cannot sample more than {} examples that generator was initialized with.".format(num_images, self.batch_size))

        # Reinitialize iterator with this task's dataset, then fetch one batch.
        sess.run(self._initialization_op)
        # Fetch a batch of size self.batch_size of images and corresponding masks:
        images, masks = sess.run(self._next_element)

        return [[image, mask] for image, mask in zip(images[:num_images], masks[:num_images])]


def _sample_mini_image_segmentation_dataset(sess, dataset, num_classes, num_shots, return_task_name: bool = False) -> Union[list, Tuple[list, str]]:
    """
    Samples a binary, image segmentation task from a dataset with num_shots examples.
    num_classes currently ignored

    Returns:
      An iterable of (input, label) tuples of length num_shots.
    """
    l = list(dataset)

    # Sample random task:
    class_obj = random.sample(l, 1)[0]

    # print("Sampled task: {}".format(class_obj.name))

    if num_shots > class_obj.batch_size:  # Account for fewer examples available than num_shots
        warnings.warn("Requested {} examples but dataset can return max of {} examples.".format(num_shots, class_obj.batch_size))
        num_shots = class_obj.batch_size

    if not return_task_name:
        return class_obj.sample(sess, num_shots)
    else:
        return class_obj.sample(sess, num_shots), class_obj.name


def _mini_batches(samples, batch_size, num_batches, replacement: bool = False, augmenter: Optional[Augmenter] = None, aug_rate: Optional[float] = None,):
    """
    Generate mini-batches from some data.
    Args:
      replacement: bool. If False, loop through all examples before sampling an example again.
    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    if aug_rate is not None:
        prob_to_return_original = 1.0 - aug_rate
    else:
        prob_to_return_original = None
    samples = list(samples)
    if len(samples) == 0:
        raise ValueError('No samples to sample. `samples` has no length: {}'.format(samples))
    if replacement:
        for _ in range(num_batches):
            cur_batch = random.sample(samples, batch_size)
            if augmenter is not None:
                _cur_batch = []
                for sample in cur_batch:
                    sample = augmenter.apply_augmentations(sample[0], sample[1], prob_to_return_original)
                    _cur_batch.append(sample)
                cur_batch = _cur_batch
            yield cur_batch
        return
    cur_batch = []
    batch_count = 0
    # i = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            if augmenter is not None:
                sample = augmenter.apply_augmentations(sample[0], sample[1], prob_to_return_original)
            # savefig_mask_on_image(sample[0], sample[1], save_path=os.path.join("augs",  str(i) + ".png"))
            # i += 1
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


def assert_train_test_split(train_set, test_set):
    train_hashes = set()
    for image, _ in train_set:
        train_hashes.add(hash_np_array(image))
    for image, _ in test_set:
        assert hash_np_array(image) not in train_hashes


def _sample_train_test_segmentation_with_replacement(samples: List, train_shots: int = 5, test_shots: int = 5):
    indices = np.random.randint(len(samples), size=train_shots)
    train_set = [samples[x] for x in indices]
    indices = np.random.randint(len(samples), size=test_shots)
    test_set = [samples[x] for x in indices]
    return train_set, test_set


def _split_train_test_segmentation(samples, test_shots=1, test_train_test_split: bool = False, shuffle_before_split: bool = True):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs. Should already be shuffled.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    samples = list(samples)[:]

    if shuffle_before_split:
        random.shuffle(samples)

    train_set = samples[:-test_shots]
    test_set = samples[-test_shots:]
    if test_train_test_split:
        assert_train_test_split(train_set, test_set)
    return train_set, test_set
