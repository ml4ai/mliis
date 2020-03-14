import tensorflow as tf


def l2_term(weight_decay=0.0005):
    """
    L2 loss on trainable variables.
    Adapted from tensorflow tpu efficientnet implementation
    """
    return tf.identity(weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                        if 'batch_normalization' not in v.name]), name="l2")


def l1_term(weight_decay=0.0005):
    """
    L1 loss on trainable variables.
    """
    return tf.identity(weight_decay * tf.add_n([tf.reduce_sum(tf.math.abs(v)) for v in tf.trainable_variables()
                                        if 'batch_normalization' not in v.name]), name="l1")

def darc1_term(logits, weight=0.0005):
    """Assumes batch dim is first."""
    return tf.identity(weight * tf.reduce_max(tf.reduce_sum(tf.abs(logits), axis=0), name="darc1"))
