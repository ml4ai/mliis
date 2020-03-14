from typing import Optional

import numpy as np
import tensorflow as tf


def debug(dataset):
    """Debugging utility for tf.data.Dataset."""
    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()

    ds_init_op = iterator.make_initializer(dataset)

    with tf.Session() as sess:
        sess.run(ds_init_op)
        viz(sess, next_element)
        import pdb; pdb.set_trace()
        res = sess.run(next_element)
        # for i in range(len(res)):
        #     print("IoU of label with itself:")
        #     print(Gecko._iou(res[i][1], res[i][1], class_of_interest_channel=None))
        print(res)


def plot_mask(mask_j: np.ndarray, figure_index=0, channel_index: Optional[int] = None, test_iou_of_label: bool = False):
    if test_iou_of_label:
        from meta_learners.supervised_reptile.supervised_reptile.reptile import Gecko
    import matplotlib.pyplot as plt
    plt.figure(figure_index)
    if channel_index is None:
        for k in range(mask_j.shape[2]):
            if np.sum(mask_j[:, :, k]) == 0:
                continue
            break
        print("class at channel {}".format(k))
    else:
        k = channel_index
    plt.imshow(mask_j[:, :, k])
    plt.show()
    if test_iou_of_label:
        print("IoU of label with itself:")
        print(Gecko._iou(mask_j.copy(), mask_j.copy(), class_of_interest_channel=None, round_labels=True))
    return k


def viz(sess, next_element, num_to_viz=20):
    try:
        import matplotlib.pyplot as plt

        for i in range(num_to_viz):
            res = sess.run(next_element)
            image = res[0].astype(int)
            mask = res[1]
            if len(image.shape) == 4:
                for j in range(image.shape[0]):
                    plt.figure(i + j)
                    plt.imshow(image[j])
                    plt.show()
                    mask_j = mask[j]
                    plot_mask(mask_j, i + j)
            else:
                plt.figure(i)
                plt.imshow(image)
                plt.show()
                plot_mask(mask, i )
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()