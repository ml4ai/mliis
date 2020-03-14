"""
This script converts the image-mask pairs of the FSS-1000 dataset to tfrecords, with one task per tfrecord.
"""
import argparse
import glob
import math
import os
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path
import sys
import time
import random
import warnings
from typing import List, Tuple, Optional

import imageio
import numpy as np
import tensorflow as tf

from data.fss_1000_utils import TEST_TASK_IDS, FP_K_TEST_TASK_IDS, split_train_test_tasks, TOTAL_NUM_FSS_CLASSES, \
    IMAGE_DIMS
from joint_train.data.constants import SERIALIZED_DTYPE

MAX_NUM_PROCESSES = 8


def parse_arguments(argv):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Writes FSS-1000 images to TFRecords.')
    parser.add_argument(
      '--input_dir',
      type=str,
      default=None,
      help='Absolute path to base directory of the FSS-1000 dataset.')
    parser.add_argument(
      '--tfrecord_dir',
      required=True,
      type=str,
      help='Directory to write tfrecords to.')
    parser.add_argument(
      '--overwrite',
      required=False,
      default=False,
      action="store_true",
      help='Overwrite existing tfrecords?')
    parser.add_argument("--compress", action="store_true", default=False)
    parser.add_argument("--fp_k_test_set", help="Hold out the test task for the fp-k classes.", action="store_true")
    parser.add_argument("--num_val_tasks", help="Number of validation tasks to hold out in addition to the 240 test tasks.", type=int, default=0)
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def get_fss_dir_paths(data_dir):
    return glob.glob(os.path.join(data_dir, "*/"))


def get_image_mask_pairs(task: str, image_ext: str = ".jpg", mask_ext: str = ".png") -> List[Tuple[str, str]]:
    masks = glob.glob(os.path.join(task, "*" + mask_ext))
    image_mask_pairs = []
    for mask in masks:
        image = mask.replace(mask_ext, image_ext)
        if os.path.exists(image):
            image_mask_pairs.append((image, mask))
        else:
            warnings.warn("No corresponding image found for mask: {}".format(mask))
    return image_mask_pairs


def main():
    """Write images to TFRecords."""
    print("Converting FSS-1000 image-mask pairs to tfrecord shards.")
    dry_run = False
    start = time.time()
    print(start)
    args = parse_arguments(sys.argv)
    if args.compress:
        ext = ".tfrecord.gzip"
    else:
        ext = ".tfrecord"

    if args.fp_k_test_set:
        test_task_ids = FP_K_TEST_TASK_IDS
    else:
        test_task_ids = TEST_TASK_IDS

    train_dirs, test_dirs, all_classes = get_fss_train_test_absolute_dir_paths(args.input_dir, test_task_ids=test_task_ids)

    train_dirs, val_dirs = split_train_test_tasks(train_dirs, args.num_val_tasks, reproducbile_splits=True)

    assert len(train_dirs) + len(val_dirs) + len(test_dirs) == TOTAL_NUM_FSS_CLASSES

    if not dry_run:
        mkdir(args.tfrecord_dir)

    for set_name, paths in zip(["train", "val", "test"], [train_dirs, val_dirs, test_dirs]):
        image_mask_pairs = []
        for folder in paths:
            print("folder: {}".format(folder))
            image_mask_pairs.extend(get_image_mask_pairs(folder))

        tfrecord_filename = os.path.join(args.tfrecord_dir, set_name + ext)

        if not dry_run:
            if not os.path.exists(tfrecord_filename) or args.overwrite:
                write_tfrecords(tfrecord_filename, image_mask_pairs, all_classes, compress=args.compress)
                print("Wrote tfrecord file to {}".format(tfrecord_filename))

    print("Finished.")
    print("Took {} minutes.".format((time.time() - start) / 60.0))


def get_fss_train_test_absolute_dir_paths(data_dir, test_task_ids: List[str] = TEST_TASK_IDS):
    expected_classes = TOTAL_NUM_FSS_CLASSES
    all_classes = get_classes_from_subdirs(data_dir, expected_classes)
    train_classes = list(set(all_classes) - set(test_task_ids))
    test_classes = list(set(test_task_ids))
    assert len(train_classes) + len(test_classes) == expected_classes

    return [os.path.join(data_dir, x) for x in train_classes], [os.path.join(data_dir, x) for x in test_classes], sorted(all_classes)


def get_classes_from_subdirs(data_dir, expected_num_classes: int):
    """Get the classes from the names of the subdirectories"""
    all_classes = os.listdir(data_dir)
    all_classes = [x for x in all_classes if os.path.isdir(os.path.join(data_dir, x))]
    if len(all_classes) != expected_num_classes:
        print("length of found classes does not equal number of expected classes")
        import pdb; pdb.set_trace()
    all_classes = sorted(all_classes)
    return all_classes


def mkdir(path):
    """
    Recursive create dir at `path` if `path` does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def one_hot_encode(mask, class_name: str, class_names: List[str], image_width: int = IMAGE_DIMS, truth_value: int = 255, seperate_background_channel: bool = True):
    if seperate_background_channel:
        background = truth_value - mask

    n_classes = len(class_names)
    i = class_names.index(class_name)

    if seperate_background_channel:
        n_classes += 1
        i += 1

    all_classes = np.zeros([image_width, image_width, n_classes])
    all_classes[:, :, i] = mask

    if seperate_background_channel:
        all_classes[:, :, 0] = background

    return all_classes


def image_to_feature(image_filename, take_first_channel=False, one_hot_encode_mask=False, all_classes: Optional[List[str]] = None, serialize_as=SERIALIZED_DTYPE):
    """
    Converts target image to a bytes feature.

    Args:
        image_filename: Full path of image with image type extension.
        take_first_channel: Set to True for masks.

    Returns:
        TF bytes Feature for the image.
    """
    im = imageio.imread(image_filename)
    class_name = os.path.basename(Path(image_filename).parent)
    img_shape = im.shape
    height, width = im.shape[0], im.shape[1]
    if height != IMAGE_DIMS or width != IMAGE_DIMS:
        print("{} is not of expected image dimensions. Skipping this sample".format(image_filename))
        return None
    if take_first_channel:
        if len(img_shape) > 2:
            im = im[:, :, 0]
    if one_hot_encode_mask:
        im = one_hot_encode(im, class_name=class_name, class_names=all_classes)
    im = im.astype(serialize_as)
    bytes_list = tf.train.BytesList(value=[im.tobytes()])
    return tf.train.Feature(bytes_list=bytes_list)


def make_example(image_filename, mask_filename, all_classes):
    """Collect TF Features into a TF Example."""
    image = image_to_feature(image_filename)
    mask = image_to_feature(mask_filename, take_first_channel=True, one_hot_encode_mask=True, all_classes=all_classes),
    if (image is None) or (mask is None):
        return None
    feature = {
        'image': image,
        'mask': mask,
    }
    features = tf.train.Features(feature=feature)
    return tf.train.Example(features=features)


def image_mask_are_valid(image: str, mask: str) -> bool:
    def _valid(image_filename):
        im = imageio.imread(image_filename)
        height, width = im.shape[0], im.shape[1]
        if height != IMAGE_DIMS or width != IMAGE_DIMS:
            print("{} is not of expected image dimensions. Skipping this sample".format(image_filename))
            return False
        return True
    res = [_valid(x) for x in [image, mask]]
    if all(res):
        return True
    return False


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def write_tfrecords(tfrecord_basename, filename_pairs, all_classes: List[str], max_examples:int = 200, compress: bool = False):
    """Write tfrecord shards in parallel"""
    if isinstance(filename_pairs, zip):
        filename_pairs = list(filename_pairs)
    elif not isinstance(filename_pairs, list):
        raise ValueError("filename_pairs must be list or zip object but is {}".format(type(filename_pairs)))
    random.shuffle(filename_pairs)
    num_examples = len(filename_pairs)

    num_shards = int(math.ceil(float(num_examples) / float(max_examples)))
    record_names = ['%s-%05i-of-%05i' % (tfrecord_basename, n + 1, num_shards) for n in range(num_shards)]

    shard_filename_pairs = [[] for _ in range(num_shards)]
    for n, filename_pair in enumerate(filename_pairs):
        sublist_index = n % num_shards
        shard_filename_pairs[sublist_index].append(filename_pair)

    iterable = [(fn, fn_pairs, ac, compress_bool) for fn, fn_pairs, ac, compress_bool in zip(record_names, shard_filename_pairs, repeat(all_classes), repeat(compress))]

    num_processes = min(num_shards, MAX_NUM_PROCESSES)
    with Pool(num_processes) as pool:
        pool.starmap(write_tfrecord, iterable)


def write_tfrecord(tfrecord_filename, filename_pairs, all_classes: List[str], compress: bool = False):
    """Write TFExamples containing images and masks to TFRecord(s).

    Args:
      tfrecord_filename: Filename to write record to, full path and extension.
      filename_pairs: List of (image_filename, mask_filename) tuples.
      all_classes: Dict mapping string to integer index
    """
    print("Writing examples to tfrecord at {}".format(tfrecord_filename))
    if compress:
        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    else:
        options = None
    writer = tf.python_io.TFRecordWriter(
        tfrecord_filename,
        options)
    for n, filename_pair in enumerate(filename_pairs):
        image_filename, mask_filename = filename_pair
        if not image_mask_are_valid(image_filename, mask_filename):
            continue

        example = make_example(image_filename, mask_filename, all_classes)

        if example is None:
            continue
        serialized_example = example.SerializeToString()
        writer.write(serialized_example)
    writer.close()
    print("Examples written to {}".format(tfrecord_filename))


def old_write_tfrecord(tfrecord_filename, filename_pairs, all_classes: List[str], max_examples:int = 200):
    """Write TFExamples containing images and masks to TFRecord(s).

    Args:
      tfrecord_filename: Filename to write record to, full path and extension.
      filename_pairs: List of (image_filename, mask_filename) tuples.
      all_classes: Dict mapping string to integer index
      max_examples: Maximum number of examples to write to each TFRecord shard.
    """
    options = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    if isinstance(filename_pairs, zip):
        filename_pairs = list(filename_pairs)
    elif not isinstance(filename_pairs, list):
        raise ValueError("filename_pairs must be list or zip object but is {}".format(type(filename_pairs)))
    random.shuffle(filename_pairs)
    num_examples = len(filename_pairs)
    # Casting for consistent behavior in python 2 and 3.
    num_shards = int(math.ceil(float(num_examples) / float(max_examples)))
    writers = [
        tf.python_io.TFRecordWriter(
            '%s-%05i-of-%05i' % (tfrecord_filename, n + 1, num_shards),
            options,
        ) for n in range(num_shards)
    ]

    for n, filename_pair in enumerate(filename_pairs):
        writer = writers[n % num_shards]
        image_filename, mask_filename = filename_pair
        if not image_mask_are_valid(image_filename, mask_filename):
            continue

        example = make_example(image_filename, mask_filename, all_classes)

        if example is None:
            continue
        serialized_example = example.SerializeToString()
        writer.write(serialized_example)
    for writer in writers:
        writer.close()
    print("Examples written to {}".format(tfrecord_filename))


if __name__ == '__main__':
    main()
