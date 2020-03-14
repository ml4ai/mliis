"""Input function for sampling few-shot image segmentation data from tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from typing import List, Optional
import multiprocessing

import tensorflow as tf

# File shuffle buffer size should be larger than the number of shards.
# Shuffle buffer size should be larger than the number of examples per shard.
# Cycle length equal to num shards will reduce randomness as only the first
#   examples will be read. 1 reads each TFRecord to the end.
_COMPRESSION_TYPE = "GZIP"
_IMAGE_WIDTH = 512
_PREFETCH_BUFFER_SIZE = 1  # Batches for training in inner loop
_FILE_SHUFFLE_BUFFER_SIZE = 256  # Max number of TFRecord files to shuffle, should be larger than the number of shards
_SHUFFLE_BUFFER_SIZE = 512  # Max size when all tasks are loaded into memory, should be much larger than number of examples, capture handle of the shards at a time
_CYCLE_LENGTH = 1  # Number of TFRecords read simultaneously
_BLOCK_LENGTH = 1  # Number of consecutive elements from each iterator


def parse_example(example, image_width, scale_to_0_1: bool = False):
    """
    Parse a TF Example into corresponding image and mask.

    Positive class is assumed to be encoded as the int value 255 in tfrecords.

    Args:
      example: Batch of TF Example protos.
      image_width: Width to resize to of image and mask, assumed to be same as height.
      scale_to_0_1: if True, divide by 255.

    Returns:
      A pair of 3 channel float images and 2 channel float segmentation masks.
      The first channel in the masks will be the background, the second channel
      will be the class of interest
    """

    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
    }

    parsed_example = tf.parse_single_example(example, features)

    image = tf.decode_raw(parsed_example['image'], tf.uint8)
    # if image_width is not None:
    image = tf.reshape(image, (image_width, image_width, 3))
    image = tf.cast(image, tf.float32)
    if scale_to_0_1:
        image /= 255.

    mask = tf.decode_raw(parsed_example['mask'], tf.uint8)
    # if image_width is not None:
    mask = tf.reshape(mask, (image_width, image_width))
    mask = tf.stack([255 - mask, mask], axis=2)  # Converts pos. class to 2-class.
    mask = tf.cast(mask, tf.float32) / 255.

    return image, mask


def make_dataset(
        file_pattern,
        batch_size,
        compression_type=_COMPRESSION_TYPE,
        image_width=_IMAGE_WIDTH,
        prefetch_buffer_size=_PREFETCH_BUFFER_SIZE,
        file_shuffle_buffer_size= _FILE_SHUFFLE_BUFFER_SIZE,
        shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE,
        cycle_length=_CYCLE_LENGTH,
        ):
    """Makes a TF Dataset from tfrecords.

    Args:
      file_pattern: TF Record filenames ending in a wildcard.
      batch_size: Number of training examples per batch.
      compression_type: Compression type for reading TFRecords.
      image_width: Width of image and mask, assumed to be same as height.
      prefetch_buffer_size: Number of data elements to preload to device.
      file_shuffle_buffer_size: Size of the shuffle buffer for shard filenames.
      shuffle_buffer_size: Number of training examples to sample from,
          trade-off in favor of speed over randomness for smaller values.
      cycle_length: Number of TFRecords to read from simultaneously.
      compression_type: Set to 'GZIP' if compressed, None if no compression.

    Returns:
      TF Record dataset for (image, mask) pairs.
    """
    verbose = True
    # TODO: parameterize num_cpus so that when running on hpc can set to <= 28
    num_cpus = 20
    if verbose:
        print("tf dataset listing files matching pattern: {}".format(file_pattern))
    dataset = tf.data.Dataset.list_files(file_pattern)  # Shuffles by default.
    dataset = dataset.repeat()
    dataset = dataset.shuffle(file_shuffle_buffer_size)
    # Behavior of interleave with cycle_length 1 is the same as flat_map. Each
    # TFRecord will be read until exhaustion before moving on to the next.
    dataset = dataset.interleave(lambda filenames: tf.data.TFRecordDataset(
                                        filenames=filenames,
                                        compression_type=compression_type),
                                 cycle_length=cycle_length,
                                 block_length=_BLOCK_LENGTH)
    dataset = dataset.map(lambda example: parse_example(example, image_width),
                          num_parallel_calls=num_cpus)
    # META-LEARNING CODE NEEDS THE RESULTING BATCH TO BE A LIST OF UNIQUE EXAMPLES.
    # dataset = dataset.shuffle(shuffle_buffer_size)
    # dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset


def load_from_tfrecords(filenames, image_width=_IMAGE_WIDTH) -> List:
    """
    Loads all examples in the tfrecord into memory.
    Example usage:
    ```
    import tensorflow as tf
    import random
    data = load_from_tfrecords(["path.tfrecords"])
    with tf.Session() as sess:
    for tuple in random.sample(data, 100):
        for tensor in tuple:  # Loop through (image, mask)
            print(tensor.eval())
    ```
    """
    compression = tf.python_io.TFRecordCompressionType.GZIP
    options = tf.python_io.TFRecordOptions(compression)
    examples = []
    for file in filenames:
        for message in tf.python_io.tf_record_iterator(file, options=options):
            examples.append(parse_example(message, image_width=image_width))
    return examples

# def debug(dataset):
#     """Debugging utility for tf.data.Dataset."""
#     iterator = tf.data.Iterator.from_structure(
#         dataset.output_types, dataset.output_shapes)
#     next_element = iterator.get_next()
#
#     ds_init_op = iterator.make_initializer(dataset)
#
#     with tf.Session() as sess:
#         sess.run(ds_init_op)
#         viz(sess, next_element)
#         import pdb; pdb.set_trace()
#         res = sess.run(next_element)
#         # for i in range(len(res)):
#         #     print("IoU of label with itself:")
#         #     print(Gecko._iou(res[i][1], res[i][1], class_of_interest_channel=None))
#         print(res)
#
#
# def plot_mask(mask_j: np.ndarray, figure_index=0, channel_index: Optional[int] = None):
#     import matplotlib.pyplot as plt
#     plt.figure(figure_index)
#     if channel_index is None:
#         for k in range(mask_j.shape[2]):
#             if np.sum(mask_j[:, :, k]) == 0:
#                 continue
#             break
#         print("class at channel {}".format(k))
#     else:
#         k = channel_index
#     plt.imshow(mask_j[:, :, k])
#     plt.show()
#     print("IoU of label with itself:")
#     print(Gecko._iou(mask_j.copy(), mask_j.copy(), class_of_interest_channel=None, round_labels=True))
#     import pdb; pdb.set_trace()
#     return k
#
#
# def viz(sess, next_element, num_to_viz=2):
#     try:
#         import matplotlib.pyplot as plt
#
#         for i in range(num_to_viz):
#             res = sess.run(next_element)
#             image = res[0].astype(int)
#             mask = res[1]
#             if len(image.shape) == 4:
#                 for j in range(image.shape[0]):
#                     plt.figure(i + j)
#                     plt.imshow(image[j])
#                     plt.show()
#                     mask_j = mask[j]
#                     plot_mask(mask_j, i + j)
#             else:
#                 plt.figure(i)
#                 plt.imshow(image)
#                 plt.show()
#                 plot_mask(mask, i )
#     except Exception as e:
#         print(e)
#         import pdb; pdb.set_trace()