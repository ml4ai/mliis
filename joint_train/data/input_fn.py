"""Input function for sampling few-shot image segmentation data from tfrecords."""

from typing import List, Optional

import numpy as np
import tensorflow as tf

from joint_train.data.constants import SERIALIZED_DTYPE
from augmenters.np_augmenters import Augmenter
from utils.debug_tf_dataset import debug

_COMPRESSION_TYPE = "GZIP"
_PREFETCH_BUFFER_SIZE = 10  # Prefetch batches for training. Reduce if testing locally.
# Shuffle buffer size should be larger than the number of examples per shard.
# Cycle length equal to num shards will reduce randomness as only the first
#   examples will be read. 1 reads each TFRecord to the end.
_SHUFFLE_BUFFER_SIZE = 400  # Reduce if testing locally. # Maintain a buffer of _SHUFFLE_BUFFER_SIZE elements, and randomly select the next element from that buffer
_NUM_SUBPROCESSES = 28
_NP_TO_TF_DTYPES = {np.uint8: tf.uint8}
_DEFAULT_READER_BUFFER_SIZE_BYTES = int(2.56e8)


class TFRecordSegmentationDataset:

    def __init__(self, tfrecord_paths: List[str], image_width:int,
                    image_channels=3, mask_channels=1, seed=0, augmenter: Optional[Augmenter]=None, seperate_background_channel: bool = True):
        """
        Image segmentation tf.data.Dataset constructor for batching from tfrecords.
        Args:
            tfrecord_paths: list of paths to tfrecords
            image_width: side of images and masks (all images and masks assumed to be square and of the same size)
            image_channels: Int number of channels in the images.
            mask_channels: Int number of channels in the masks.
            seed: Integer seed for the RNG in the data pipeline.
            augmenter: An object with an apply_augmentations method that will be wrapped with tf.py_func and applied to input examples
        """
        self.tfrecord_paths = tfrecord_paths
        self.num_shards = len(self.tfrecord_paths)
        self.tfrecord_paths_tensor = tf.constant(self.tfrecord_paths)
        self.image_width = image_width
        self.image_channels = image_channels
        if seperate_background_channel:
            mask_channels += 1
        self.mask_channels = mask_channels
        print("building dataset with labels with {} mask channels".format(self.mask_channels))
        self.serialized_image_raw_dtype = _NP_TO_TF_DTYPES[SERIALIZED_DTYPE]
        self.serialized_mask_raw_dtype = _NP_TO_TF_DTYPES[SERIALIZED_DTYPE]
        self.seed = seed
        self.augmenter = augmenter

    def make_dataset(self, batch_size=8, num_parallel_calls=_NUM_SUBPROCESSES, num_concurrent_reads: Optional[int] = None):
        """
        Parse tfrecords from paths, shuffle and batch.
        the next element in dataset op and the dataset initializer op.
        Args:
            batch_size: Number of images/masks in each batch returned.
            num_parallel_calls: Number of parallel subprocesses for Dataset.map calls.
            num_concurrent_reads: Interleave reading from this number of tfrecords. Sets `cycle_length` param to
                interleave. If unspecified, will be set to the number of shards.
        Returns:
            next_element: A tensor with shape [2], where next_element[0]
                          is image batch, next_element[1] is the corresponding
                          mask batch.
            init_op: Data initializer op, needs to be executed in a session
                     for the data queue to be filled up and the next_element op
                     to yield batches.
        """
        print("Making dataset from shards {}".format(self.tfrecord_paths))
        dataset = tf.data.Dataset.from_tensor_slices(self.tfrecord_paths)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self.num_shards)
        if num_concurrent_reads is None:
            num_concurrent_reads = self.num_shards
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=_COMPRESSION_TYPE, buffer_size=_DEFAULT_READER_BUFFER_SIZE_BYTES),
            cycle_length=num_concurrent_reads, block_length=1)
        dataset = dataset.map(lambda record: self._parse_example(record), num_parallel_calls=num_parallel_calls)

        if self.augmenter is not None:
            apply_augs = lambda i, m: self.augmenter.apply_augmentations(i, m, return_image_mask_in_list=False)
            dataset = dataset.map(
                lambda image, mask: tf.py_func(apply_augs, [image, mask], [tf.float32, tf.float32]),
                                       num_parallel_calls=num_parallel_calls)
            dataset = dataset.map(lambda image, mask: (
                tf.reshape(image, (self.image_width, self.image_width, self.image_channels)),
                tf.reshape(mask, (self.image_width, self.image_width, self.mask_channels))),
                                  num_parallel_calls=num_parallel_calls)

        dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(_PREFETCH_BUFFER_SIZE)
        # debug(dataset)  # Uncomment to visualize examples
        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()
        ds_init_op = iterator.make_initializer(dataset)

        return next_element, ds_init_op

    def _parse_example(self, example, scale_to_0_1: bool = False):
        """
        Parse a TF Example into corresponding image and mask.

        Positive class is assumed to be encoded as the int value 255 in tfrecords.

        Args:
            record_name: name of the tfrecord indicating the class
            example: Batch of TF Example protos.
            image_width: Width to resize to of image and mask, assumed to be same as height.
            scale_to_0_1: if True, scale images by dividing by 255.

        Returns:
            A pair of 3 channel float images and n-channel float segmentation masks.
            The first channel in the masks will be the background, the rest
            will be the classes of interest.
        """

        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
        }
        parsed_example = tf.parse_single_example(example, features)

        image = tf.decode_raw(parsed_example['image'], self.serialized_image_raw_dtype)
        image = tf.reshape(image, (self.image_width, self.image_width, self.image_channels))
        image = tf.cast(image, tf.float32)
        if scale_to_0_1:
            image /= 255.

        mask = tf.decode_raw(parsed_example['mask'], self.serialized_mask_raw_dtype)
        mask = tf.reshape(mask, (self.image_width, self.image_width, self.mask_channels))
        mask = tf.cast(mask, tf.float32) / 255.
        return image, mask


def parse_example(example, image_width:int = 224, image_channels: int = 3, mask_channels: int = 1000, scale_to_0_1: bool = False, serialized_mask_raw_dtype = tf.float64):
    """
    Parse a TF Example into corresponding image and mask.

    Positive class is assumed to be encoded as the int value 255 in tfrecords.

    Args:
        record_name: name of the tfrecord indicating the class
        example: Batch of TF Example protos.
        image_width: Width to resize to of image and mask, assumed to be same as height.
        scale_to_0_1: if True, divide by 255.

    Returns:
        A pair of 3 channel float images and n-channel float segmentation masks.
        The first channel in the masks will be the background, the rest
        will be the classes of interest.
    """

    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
    }

    parsed_example = tf.parse_single_example(example, features)

    image = tf.decode_raw(parsed_example['image'], tf.uint8)
    image = tf.reshape(image, (image_width, image_width, image_channels))
    image = tf.cast(image, tf.float32)
    if scale_to_0_1:
        image /= 255.

    mask = tf.decode_raw(parsed_example['mask'], serialized_mask_raw_dtype)  # tf.uint8)
    mask = tf.reshape(mask, (image_width, image_width, mask_channels))
    mask = tf.cast(mask, tf.float32) / 255.
    return image, mask


def load_from_tfrecords(filenames) -> List:
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
            examples.append(parse_example(message))
            break
    return examples
