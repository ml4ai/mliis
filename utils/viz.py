"""Visualization utils for image segmentation."""
import copy
import os

import numpy as np


def plot_mask_on_image(image: np.ndarray, mask: np.ndarray, truth_value: int = 1.0, alpha=0.75, scale_to_0_1: bool = True) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if scale_to_0_1:
        image /= 255.
    masked = np.ma.masked_where(mask != truth_value, mask)
    fig, ax1 = plt.subplots()
    ax1.imshow(image)
    ax1.imshow(masked, interpolation='none', alpha=alpha, cmap=cm.jet)
    plt.show()


def _plot_two_images(A, B):
    import matplotlib.pyplot as plt

    plt.figure()

    plt.subplot(121)
    plt.imshow(A)
    plt.subplot(122)
    plt.imshow(B)
    plt.show()


def _save_plot_two_images(A, B, fname):
    import matplotlib.pyplot as plt

    fig = plt.figure()

    plt.subplot(121)
    plt.imshow(A)
    plt.subplot(122)
    plt.imshow(B)

    plt.savefig(fname)

    plt.close(fig)


def savefig_mask_on_image(image: np.ndarray, mask: np.ndarray, truth_value: int = 1, alpha=0.5, save_path = None) -> None:
    """Plot mask on image for binary segmentation."""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    from matplotlib import pyplot as plt, cm as cm  # Import locally to avoid errors when matplotlib is not available
    image = image.copy()
    mask = mask.copy()
    if truth_value not in {1, 255}:
        raise ValueError("Foreground class should be 1 or 255")
    print("mask.shape: {}".format(mask.shape))
    if mask.shape[2] == 2:  # Get the second channel
        mask = mask[:, :, 1]
    if truth_value == 1:
        mask *= 255
        truth_value = 255
    image, mask = image.astype(int), mask.astype(int)
    masked = np.ma.masked_where(mask != truth_value, mask)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    fig, ax1 = plt.subplots()
    ax1.imshow(image)
    ax1.imshow(masked, interpolation='none', alpha=alpha, cmap=cm.autumn)  # cm.jet
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches=0,
            pad_inches=0)
        print("saved figure to {}".format(save_path))
        plt.close()
    else:
        print("No path speced to save to.")
        plt.show()


def savefig_batch_mask_on_image(images, masks, truth_value: int = 1, alpha=0.75, save_path_bn = None, ext=".png") -> None:
    """Plot mask on image for binary segmentation."""
    from matplotlib import pyplot as plt, cm as cm  # Import locally to avoid errors when matplotlib is not available
    for i, image_mask in enumerate(zip(images, masks)):
        image, mask = image_mask
        image = image.copy()
        mask = mask.copy()
        if truth_value not in {1, 255}:
            raise ValueError("Foreground class should be 1 or 255")
        print("mask.shape: {}".format(mask.shape))
        if mask.shape[2] == 2:  # Get the second channel
            mask = mask[:, :, 1]
        if truth_value == 1:
            mask *= 255
            _truth_value = 255
        image, mask = image.astype(int), mask.astype(int)
        masked = np.ma.masked_where(mask != _truth_value, mask)
        fig, ax1 = plt.subplots()
        ax1.imshow(image)
        ax1.imshow(masked, interpolation='none', alpha=alpha, cmap=cm.jet)
        if save_path_bn:
            save_path = save_path_bn + "_" + str(i) + ext
            plt.savefig(save_path)
            print("saved figure to {}".format(save_path))
            plt.close()
        else:
            print("No path speced to save to.")
