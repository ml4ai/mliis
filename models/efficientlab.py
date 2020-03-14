"""
EfficientLab neural network model definition adapted for few-shot meta-learning.
"""
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

from models.efficientnet import efficientnet_builder
from models.efficientnet.constants import MEAN_RGB, STDDEV_RGB
from models.efficientnet.efficientnet_model import conv_kernel_initializer
from models.regularizers import darc1_term, l2_term, l1_term
from utils.util import latest_checkpoint

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)
FINAL_LAYER_WEIGHTS_NAME = "final_layer_weights"
FEATURE_DECODER_SCOPE_NAME = "decode"


class EfficientLab:
    """Image segmentation network with EfficientNet feature extractor and residual skip decoder."""
    def __init__(self, images: Optional[tf.Tensor] = None, labels: Optional[tf.Tensor] = None, is_training: bool = True, n_classes=1, n_rows=224, n_cols=224,
                 spatial_pyramid_pooling: bool = False, skip_decoding: bool = False,
                 feature_extractor_name: str = "efficientnet-b0", l2: bool = True, l1: bool = False,
                 darc1: bool = False, final_layer_dropout_rate: Optional[float] = 0.2, dice: bool = True,
                 optimizer: Optional[tf.train.Optimizer] = None, rsd: Optional[List[int]] = [2], disable_lsd_residual_connections: bool = False, seperate_background_channel: bool = True, binary_iou_loss: bool = True, **optim_kwargs):
        if optimizer is None:
            optimizer = DEFAULT_OPTIMIZER
        print("Using optimizer {}".format(optimizer))
        self.optimizer_class = optimizer
        self.n_input_channels = 3
        self.n_input_rows = n_rows
        self.n_input_cols = n_cols
        self.binary_iou_loss = binary_iou_loss  # IoU loss term will be for 2 channel masks.
        self.seperate_background_channel = seperate_background_channel
        if self.seperate_background_channel:
            self.n_output_channels = n_classes + 1  # Add 1 for background class
        else:
            self.n_output_channels = n_classes
        if images is None:
            self.input_ph = tf.placeholder(tf.float32, name="X", shape=(None, self.n_input_rows, self.n_input_cols,
                                                                        self.n_input_channels))
            print("Input placeholder: {}".format(self.input_ph))
        else:
            assert isinstance(images, tf.Tensor), "images must be of type tf.Tensor but is {}".format(type(images))
            self.input_ph = images
            print("Input images: {}".format(self.input_ph))
        self.input_shape = tf.shape(self.input_ph)

        if labels is None:
            self.label_ph = tf.placeholder(tf.float32, name="Y", shape=(None, self.n_input_rows, self.n_input_cols,
                                                                    self.n_output_channels))
            print("Label placeholder: {}".format(self.label_ph))
        else:
            assert isinstance(labels, tf.Tensor), "labels must be of type tf.Tensor but is {}".format(type(labels))
            self.label_ph = labels
            print("Labels: {}".format(self.label_ph))

        self.flat_label_ph = tf.reshape(self.label_ph, [-1, self.n_output_channels])
        self.is_training_ph = tf.placeholder_with_default(is_training, shape=())

        self.l2 = l2
        self.l1 = l1
        self.darc1 = darc1
        self.dice = dice  # Will cause model to be trained with soft dice loss function

        supported_feature_extractors = ["efficientnet-b0", "efficientnet-b3"]
        if feature_extractor_name not in supported_feature_extractors:
            raise ValueError("feature_extractor_name must be in {} but is: {}".format(supported_feature_extractors,
                                                                                      feature_extractor_name))
        self.feature_extractor_name = feature_extractor_name
        if self.feature_extractor_name == "efficientnet-b0":
            self.aspp_dimension = 112
            self.max_block_num = 10
        elif self.feature_extractor_name == "efficientnet-b3":
            self.aspp_dimension = 136
            self.max_block_num = 17
        else:
            raise ValueError("Feature extractor must be in {} to have aspp dimension defined.".format(supported_feature_extractors))

        self.feature_decoder_name = FEATURE_DECODER_SCOPE_NAME

        self.spatial_pyramid_pooling = spatial_pyramid_pooling  # DeepLab style atrous spatial pyramid pooling
        self.skip_decoding = skip_decoding  # DeepLab v3+ style decoder

        self.rsd = rsd  # Our improved decoder
        self.disable_lsd_residual_connections = disable_lsd_residual_connections

        self.final_layer_scope = self.feature_decoder_name + "/" + FINAL_LAYER_WEIGHTS_NAME

        self.feature_extractor = self.build_efficientnet

        self.final_layer_dropout_rate_ph = final_layer_dropout_rate
        if self.final_layer_dropout_rate_ph is not None and self.final_layer_dropout_rate_ph > 0:
            print("Using dropout at final layer with drop rate {}".format(self.final_layer_dropout_rate_ph))
            self.final_layer_dropout_rate_ph = tf.placeholder_with_default(self.final_layer_dropout_rate_ph, shape=())
            self._final_layer_dropout = tf.keras.layers.Dropout(self.final_layer_dropout_rate_ph)
        else:
            self._final_layer_dropout = None

        self.weights_initializer = conv_kernel_initializer
        self.final_layer_weights_initializer = self.weights_initializer

        # Explicitly track state of whether or not variables have been initialized:
        self.variables_initialized = False

        self.lr_ph = None  # set in build optimizer
        self.build_model(self.input_ph, **optim_kwargs)

    def build_model(self, features, **optim_kwargs):
        """Build model with input features (images) in range [0, 255]."""
        features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
        features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
        # TODO: could implement graph code augmentation here. Wrap in tf.cond(self.is_training_ph, lambda _features: augment_batch(_features), lambda: features)
        embedded_image, skip_connections = self.encode(features)
        self.probabilities = self.decode(embedded_image, skip_connections)
        self.build_optimizer(**optim_kwargs)
        return self.probabilities

    def encode(self, image):
        """Encodes an image into an embedding."""
        embedded_image, skip_connections = self.feature_extractor(image)
        return embedded_image, skip_connections

    def decode(self, embedded_image: tf.Tensor, skip_connections: List[tf.Tensor]) -> tf.Tensor:
        """Decodes an embedded image into segmentation probabilities."""
        skip_connection = skip_connections[1]
        with tf.variable_scope("decode"):
            if self.spatial_pyramid_pooling:
                embedded_image = self.aspp(embedded_image)

            if self.skip_decoding:  # DeepLab style decoding
                print("Refining decoding with skip connections.")
                with tf.variable_scope("decode_skip_connections"):
                    decoded = tf.image.resize_images(embedded_image, self.input_shape[1:3] // 4,
                                                     method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
                    # Allow twice as many features for encoder features in decoding:
                    decoded_skip_dim = self.aspp_dimension // 2
                    decoded_skip = tf.layers.conv2d(skip_connection, filters=decoded_skip_dim, kernel_size=[1, 1],
                                                    padding="SAME", use_bias=False)
                    decoded_skip = tf.layers.batch_normalization(decoded_skip, training=True)
                    decoded_skip = tf.nn.swish(decoded_skip)

                    decoded = tf.concat([decoded, decoded_skip], axis=-1)

                    # Apply depth-wise separable convolution and residual connections:
                    decoded = self.sep_conv(decoded, self.aspp_dimension + decoded_skip_dim, kernel_size=3)  # + decoded
                    decoded = self.sep_conv(decoded, self.aspp_dimension + decoded_skip_dim, kernel_size=3)  # + decoded
            else:
                decoded = embedded_image

            # Loop though list of reduction indices for building the rsd upsampling layers
            if self.rsd is not None and len(self.rsd) > 0:
                for i in sorted(self.rsd, reverse=True):
                    reduction_index = i - 1
                    print("Building residual skip decoder with reduction {}".format(i))
                    decoded = self.residual_skip_decoder(decoded, skip_connections[reduction_index],
                                                         num_output_filters=self.aspp_dimension, var_scope_index=reduction_index)

            if self._final_layer_dropout is not None:
                decoded = self._final_layer_dropout(decoded, training=self.is_training_ph)

            print("final feature tensor: {}".format(decoded))
            decoded = tf.layers.conv2d(decoded, self.n_output_channels, [1, 1], padding="SAME", activation=None,
                                       name=FINAL_LAYER_WEIGHTS_NAME, kernel_initializer=self.final_layer_weights_initializer,
                                       use_bias=True)

            # If using variable sized images in a batch, will have to implement something like this using map_fn:
            # https://stackoverflow.com/questions/48755945/resize-images-with-a-batch
            self.logits = tf.image.resize_images(decoded, self.input_shape[1:3], method=tf.image.ResizeMethod.BILINEAR,
                                                 align_corners=True)
            self.flat_logits = tf.reshape(self.logits, [-1, self.n_output_channels])
            probs = tf.nn.softmax(self.logits)

            self.predictions = self._probabilities_to_classes(probs)
            return probs

    def residual_skip_decoder(self, embedded_image: tf.Tensor, skip_connection: tf.Tensor,
                              conv_over_contatenated_features: bool = True, num_output_filters: int = 112, var_scope_index: int = 0,
                              ) -> tf.Tensor:
        print("Building residual skip decoding module.")
        print("Skip connection tensor: {}".format(skip_connection))

        def conv_nl_bn_branch(features, num_filters, kernel_size, dilation_rate=1):
            features = tf.layers.conv2d(features, filters=num_filters, kernel_size=[kernel_size, kernel_size],
                                        dilation_rate=[dilation_rate, dilation_rate], padding="SAME",
                                        use_bias=True)
            features = tf.nn.swish(features)
            return tf.layers.batch_normalization(features, training=self.is_training_ph)

        def pool_image_features(features):
            """Computes mean embedded image."""
            input_shape = tf.shape(features)
            features = tf.reduce_mean(features, axis=[1, 2], keep_dims=True)
            features = tf.tile(features, multiples=[1, input_shape[1], input_shape[2], 1])
            return features

        legacy_var_scope_name = False  # FIXME: delete these conditionals and use only one var scope naming pattern.
        if legacy_var_scope_name:
            var_scope_name = "decode_skip_connections"
        else:
            var_scope_name = "decode_skip_connections_{}".format(var_scope_index)
        with tf.variable_scope(var_scope_name):
            upsampled = tf.image.resize_images(embedded_image, tf.shape(skip_connection)[1:3],
                                             method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

            decoded = tf.concat([upsampled, skip_connection], axis=-1)
            print("Concatenated deep and skip feature maps: {}".format(decoded))

            if conv_over_contatenated_features:
                print("Convolving over concatenated feature maps in decoding layers.")
                if upsampled.get_shape().as_list()[-1] != num_output_filters:
                    print("Increasing upsampled skip connection number of filters with 1x1 conv.")
                    upsampled = conv_nl_bn_branch(upsampled, num_output_filters, 1, 1)

                num_decoded_filters = upsampled.get_shape().as_list()[-1]
                branch_0 = conv_nl_bn_branch(decoded, num_decoded_filters, 1)
                branch_1 = conv_nl_bn_branch(decoded, num_decoded_filters, 3, 2)
                branch_2 = pool_image_features(decoded)

                pyramid = tf.concat([branch_0, branch_1, branch_2], axis=-1)

                decoded = conv_nl_bn_branch(pyramid, num_output_filters, 3)

                if not self.disable_lsd_residual_connections:
                    print("Building residual connection in RSD module.")
                    # Residual connection
                    decoded += upsampled

            return decoded


    def build_efficientnet(self, image) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Returns EfficientNet embedded image and skip connections.

        Use endpoints['reduction_i'] for detection/segmentation, as the last intermediate feature with reduction level i. For example, if input image has resolution 224x224, then:
        endpoints['reduction_1'] has resolution 112x112
        endpoints['reduction_2'] has resolution 56x56
        endpoints['reduction_3'] has resolution 28x28
        endpoints['reduction_4'] has resolution 14x14
        endpoints['reduction_5'] has resolution 7x7
        """
        features, endpoints = efficientnet_builder.build_model_base(image, self.feature_extractor_name, training=self.is_training_ph, max_block_num=self.max_block_num)
        return endpoints['reduction_4'], [endpoints['reduction_1'], endpoints['reduction_2'], endpoints['reduction_3'], endpoints['reduction_4']]

    def aspp(self, inputs, dropout_rate=0.5, residual_connection=False):
        """
        Auto-DeepLab style Atrous Spatial Pyramid Pooling with dropout in place of batch norm, potentially with residual
        connection.
         """
        print("Building atrous spatial pyramid pooling layers.")
        with tf.variable_scope("spatial_pyramid_pooling", reuse=tf.AUTO_REUSE):
            input_shape = tf.shape(inputs)

            with tf.variable_scope("branch_0"):
                b0 = tf.layers.conv2d(inputs, self.aspp_dimension, [1, 1], use_bias=True, padding="SAME")
                b0 = tf.nn.swish(b0)
                b0 = tf.layers.dropout(b0, rate=dropout_rate, training=self.is_training_ph)

            with tf.variable_scope("branch_1"):
                # Following auto-deeplab, atrous convolution with rate = 96/(downsample_factor)
                # which equals 6 where downsample_factor = 16 (4 halvings).
                b1 = tf.layers.conv2d(inputs, self.aspp_dimension, 3, dilation_rate=6, use_bias=True, padding="SAME")
                b1 = tf.nn.swish(b1)
                b1 = tf.layers.dropout(b1, rate=dropout_rate, training=self.is_training_ph)

            with tf.variable_scope("branch_2"):
                # mean embedded image:
                b2 = tf.reduce_mean(inputs, axis=[1, 2])
                b2 = tf.expand_dims(b2, -1)
                b2 = tf.expand_dims(b2, -1)
                b2 = tf.layers.conv2d(b2, self.aspp_dimension, [1, 1], use_bias=True, padding="SAME")
                b2 = tf.layers.dropout(b2, rate=dropout_rate, training=self.is_training_ph)
                b2 = tf.nn.swish(b2)
                b2 = tf.image.resize_images(b2, input_shape[1:3], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

            x = tf.concat([b2, b1, b0], axis=-1)

            output_embedding_dim = self.aspp_dimension

            x = tf.layers.conv2d(x, output_embedding_dim, [1, 1], padding="SAME", use_bias=True)
            x = tf.nn.swish(x)
            x = tf.layers.dropout(x, rate=dropout_rate, training=self.is_training_ph)

            if residual_connection:
                x += inputs
            return x

    def _probabilities_to_classes(self, probabilities, thresh=0.5):
        return tf.cast(tf.to_float(probabilities > thresh), dtype=tf.float32)

    def build_optimizer(self, **optim_kwargs):
        print("Label smoothing epsilon: {}".format(optim_kwargs["label_smoothing"]))
        loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(self.flat_label_ph, logits=self.flat_logits, label_smoothing=optim_kwargs["label_smoothing"]))
        self.iou = self._iou(self.label_ph, self.probabilities)
        print("Defining optimizer with default learning rate: {}".format(optim_kwargs["learning_rate"]))
        self.lr_ph = tf.placeholder_with_default(optim_kwargs["learning_rate"], shape=[])
        self.optimizer = self.optimizer_class(learning_rate=self.lr_ph)
        if self.dice:
            loss = self._soft_dice(loss, self.iou)
        if self.darc1:
            print("Adding darc1 term to loss")
            loss += darc1_term(self.logits)
        if self.l2:
            print("Adding l2 weight decay term to loss")
            loss += l2_term()
        if self.l1:
            print("Adding l1 weight decay term to loss")
            loss += l1_term()
        # TODO: try l1
        self.loss = loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.minimize_op = self.optimizer.minimize(self.loss)

    def _soft_dice(self, cross_entropy_loss: tf.Tensor, iou: tf.Tensor):
        """
        Implements cross entropy minus the log of the dice.
        BCE - ln(dice)
        ref: http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/
        """
        print("Defining soft dice loss = CE - ln(dice)")
        dice = (2. * iou) / (iou + 1.)
        return cross_entropy_loss - tf.log(dice)

    def _binary_iou(self, Y_true, Y_hat, epsilon=1e-7, true_class_in_first_channel: bool = False):
        """
        Returns the intersection over union score of the batch for binary segmentation problems

        intesection = Y_hat.flatten() * Y_true.flatten()
        IOU = (intersection + epsilon) / (Y_hat.sum() + Y_true.sum() - intersection + epsilon)

        :param Y_true: (4-D array): (N, H, W, 2)
        :param Y_hat: (4-D array): (N, H, W, 2)
        :return: floating point IoU score
        """
        if true_class_in_first_channel:
            Y_true_single_channel = Y_true[:, :, :, 0]
            Y_hat_single_channel = Y_hat[:, :, :, 0]
        else:
            Y_true_single_channel = Y_true[:, :, :, 1]
            Y_hat_single_channel = Y_hat[:, :, :, 1]

        print("_iou tensors:")
        height, width, _ = Y_hat.get_shape().as_list()[1:]

        pred_flat = tf.reshape(Y_hat_single_channel, [-1, height * width])
        true_flat = tf.reshape(Y_true_single_channel, [-1, height * width])

        return self._compute_iou(true_flat, pred_flat, epsilon)

    def _iou(self, Y_true, Y_hat, epsilon=1e-7, exclude_bg_channel: bool = False) -> tf.Tensor:
        """
        Returns the intersection over union score of the batch for image segmentation problems

        intesection = Y_hat.flatten() * Y_true.flatten()
        IOU = (intersection + epsilon) / (Y_hat.sum() + Y_true.sum() - intersection + epsilon)

        :param Y_true: (4-D array): (N, H, W, C)
        :param Y_hat: (4-D array): (N, H, W, C)
        :return: floating point IoU score
        """
        if self.binary_iou_loss:
            return self._binary_iou(Y_true, Y_hat, epsilon)

        # multi_class_iou
        if self.seperate_background_channel and exclude_bg_channel:
            Y_true = Y_true[:, :, :, 1:]  # Background class is assumed to be in first channel, so we skip it
            Y_hat = Y_hat[:, :, :, 1:]

        print("_iou tensors:")
        height, width, depth = Y_hat.get_shape().as_list()[1:]
        print("height: {}, width: {}, depth: {}".format(height, width, depth))

        pred_flat = tf.reshape(Y_hat, [-1, height * width * depth])
        true_flat = tf.reshape(Y_true, [-1, height * width * depth])
        # -> shape=(batch, height * width * depth)

        return self._compute_iou(true_flat, pred_flat, epsilon)


    def _compute_iou(self, true, pred, epsilon=1e-7):
        """
        Compute the IoU between two unrolled tensors of shape=(batch, d),
        where d is the number of scalars to compare.
        """
        print('pred_flat: {}'.format(pred))
        print('true_flat: {}'.format(true))

        intersection = tf.reduce_sum(pred * true, axis=1)
        denominator = tf.reduce_sum(pred, axis=1) + tf.reduce_sum(true, axis=1) - intersection

        return tf.reduce_mean((intersection + epsilon) / (denominator + epsilon))

    def restore_model(self, sess, ckpt_dir, enable_ema=False, export_ckpt=None, filter_to_scopes: Optional[List[str]] = None, filter_out_scope: Optional[str] = None, convert_ckpt_to_rel_path: bool = False):
        """Restore variables from checkpoint dir."""
        assert isinstance(filter_to_scopes, list) or filter_to_scopes is None
        assert isinstance(filter_out_scope, str) or filter_out_scope is None
        sess.run(tf.global_variables_initializer())
        if convert_ckpt_to_rel_path:
            checkpoint = latest_checkpoint(ckpt_dir, return_relative=True)
        else:
            checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if enable_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
            for v in tf.global_variables():
                if 'moving_mean' in v.name or 'moving_variance' in v.name:
                    ema_vars.append(v)
            ema_vars = list(set(ema_vars))
            var_dict = ema.variables_to_restore(ema_vars)
            ema_assign_op = ema.apply(ema_vars)
        else:
            var_dict = {}
            for v in list(set(tf.global_variables())):
                var_dict[v.op.name] = v

            ema_assign_op = None

        sess.run(tf.global_variables_initializer())

        if filter_to_scopes is None and filter_out_scope is not None:
            var_dict = {key: value for key, value in var_dict.items() if not key.startswith(filter_out_scope)}
        elif filter_to_scopes is not None and filter_out_scope is not None:
            var_dict = {key: value for key, value in var_dict.items() if any([key.startswith(x) for x in filter_to_scopes]) and not key.startswith(filter_out_scope)}
        elif filter_to_scopes is not None:
            var_dict = {key: value for key, value in var_dict.items() if any([key.startswith(x) for x in filter_to_scopes])}

        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, checkpoint)

        self.variables_initialized = True
        print("Variables initialized")

        if export_ckpt:
            if ema_assign_op is not None:
                sess.run(ema_assign_op)
            saver = tf.train.Saver(max_to_keep=2)
            saver.save(sess, export_ckpt)
        print("{} variables restored".format(len(var_dict)))

    def sep_conv(self, inputs: tf.Tensor,
                 filters: int,
                 kernel_size: int,
                 strides: int = 1,
                 dilation_rate: int = 1,
                 depth_multiplier: int = 1
                 ) -> tf.Tensor:
        """Depth-wise separable convolution"""
        # Depth-wise convolution phase:
        x = tf.keras.layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            depth_multiplier=depth_multiplier,
            strides=strides,
            dilation_rate=dilation_rate,
            padding="SAME",
            use_bias=False,
            depthwise_initializer=self.weights_initializer)(inputs)
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.nn.swish(x)

        # Output phase:
        x = tf.layers.conv2d(
            x,
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=self.weights_initializer,
            padding="SAME",
            use_bias=False)
        return tf.nn.swish(tf.layers.batch_normalization(x, training=True))


def glorot_uniform_initializer(shape, dtype=None, partition_info=None):
    """
    The Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    """
    print("Initializing tensor with Glorot uniform.")
    del partition_info
    kernel_height, kernel_width, in_filters, out_filters = shape
    fan_in = int(kernel_height * kernel_width * in_filters)
    fan_out = int(kernel_height * kernel_width * out_filters)
    print("fan_in {}".format(fan_in))
    print("fan_out {}".format(fan_out))
    limit = np.sqrt(6. / fan_in + fan_out)
    return tf.random_uniform(shape, -limit, limit, dtype=dtype)


def glorot_normal_initializer(shape, dtype=None, partition_info=None):
    """
    The Glorot normal initializer, also called Xavier normal initializer.

    Draws samples from a normal distribution centered on 0
    with standard deviation given by
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number
    of input units in the weight tensor and `fan_out` is the number of
    output units in the weight tensor.
    Args:
        shape:
        dtype:
        partition_info:

    Returns:
        An initialization for the variable
    """
    print("Initializing tensor with Glorot normal.")
    del partition_info
    kernel_height, kernel_width, in_filters, out_filters = shape
    fan_in = int(kernel_height * kernel_width * in_filters)
    fan_out = int(kernel_height * kernel_width * out_filters)
    print("fan_in {}".format(fan_in))
    print("fan_out {}".format(fan_out))
    return tf.random_normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / (fan_in + fan_out)), dtype=dtype)

