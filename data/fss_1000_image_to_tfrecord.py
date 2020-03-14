"""
This script converts the image-mask pairs of the FSS-1000 dataset to tfrecords, with one task per tfrecord.

Example usage:
python fss_1000_image_to_tfrecord.py --input_dir fewshot_data --tfrecord_dir fewshot_shards/
"""
import argparse
import glob
import os
import sys
import time
import random
import warnings
from typing import List, Tuple

import imageio
import tensorflow as tf


IMAGE_DIMS = 224  # Length of one side of input images. Images assumed to be square.


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
      type=bool,
      help='Overwrite existing tfrecords?')
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
    print("Converting FSS-1000 image-mask pairs to tfrecords, one per task.")
    dry_run = False
    start = time.time()
    print(start)
    args = parse_arguments(sys.argv)

    task_dirs = get_fss_dir_paths(args.input_dir)
    print("{} tasks found".format(len(task_dirs)))

    if not dry_run:
        mkdir(args.tfrecord_dir)

    for task in task_dirs:
        image_mask_pairs = get_image_mask_pairs(task)
        task_name = os.path.basename(task.rstrip("/"))
        print("Processing task: {}".format(task_name))
        tfrecord_filename = os.path.join(args.tfrecord_dir, task_name + ".tfrecord.gzip")

        if not dry_run:
            if not os.path.exists(tfrecord_filename) or args.overwrite:
                write_tfrecord(tfrecord_filename, image_mask_pairs)
                print("Wrote tfrecord file to {}".format(tfrecord_filename))

    print("Finished.")
    print("Took {} minutes.".format((time.time() - start) / 60.0))


def mkdir(path):
    """
    Recursive create dir at `path` if `path` does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def image_to_feature(image_filename, take_first_channel=False):
    """
    Converts target image to a bytes feature.

    Args:
        image_filename: Full path of image with image type extension.
        take_first_channel: Set to True for masks.

    Returns:
        TF bytes Feature for the image.
    """
    im = imageio.imread(image_filename)
    img_shape = im.shape
    height, width = im.shape[0], im.shape[1]
    if height != IMAGE_DIMS or width != IMAGE_DIMS:
        print("{} is not of expected image dimensions. Skipping this sample".format(image_filename))
        return None
    if take_first_channel:
        if len(img_shape) > 2:
            im = im[:, :, 0]
    bytes_list = tf.train.BytesList(value=[im.tobytes()])
    return tf.train.Feature(bytes_list=bytes_list)


def make_example(image_filename, mask_filename):
    """Collect TF Features into a TF Example."""
    image = image_to_feature(image_filename)
    mask = image_to_feature(mask_filename, take_first_channel=True),
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


def write_tfrecord(tfrecord_filename, filename_pairs, max_examples=None):
    """Write TFExamples containing images and masks to TFRecord(s).

    Args:
        tfrecord_filename: Filename to write record to, full path and extension.
        filename_pairs: List of (image_filename, mask_filename) tuples.
        max_examples: Maximum number of examples to write to each TFRecord shard.
    """
    options = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    if isinstance(filename_pairs, zip):
        filename_pairs = list(filename_pairs)
    elif not isinstance(filename_pairs, list):
        raise ValueError("filename_pairs must be list or zip object but is {}".format(type(filename_pairs)))
    random.shuffle(filename_pairs)  # Shuffle examples within a task.

    writer = tf.python_io.TFRecordWriter(tfrecord_filename, options=options)
    i = 0
    for filename_pair in filename_pairs:
        image_filename, mask_filename = filename_pair
        if not image_mask_are_valid(image_filename, mask_filename):
            continue
        example = make_example(image_filename, mask_filename)
        serialized_example = example.SerializeToString()
        writer.write(serialized_example)
        i += 1
    writer.close()
    print("{} examples written to {}".format(i, tfrecord_filename))


if __name__ == '__main__':
    main()
