"""Image augmentations in numpy with support for dense-labels. Input features (images) should be in range [0, 255]."""
import random
from random import shuffle
import numpy as np
from scipy.ndimage import rotate
from typing import Optional, Union, List


def additive_gaussian_noise(image, mask, mean_sd=5.1):
    sd = np.abs(np.random.normal(mean_sd, 1, 1))
    noise = np.random.normal(0, sd, image.shape)
    return np.clip(image + noise, 0., 255.).astype(np.float32), mask.astype(np.float32)


def exposure(image, mask, mean_sd=12.75):
    sd = np.abs(np.random.normal(mean_sd, 1, 1))
    noise = np.random.normal(0, sd, 1)
    return np.clip(image + noise, 0., 255.).astype(np.float32), mask.astype(np.float32)


def random_eraser(input_img, mask, s_l=0.02, s_h=0.10, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    """
    Random eraser https://arxiv.org/pdf/1708.04896.pdf
    Adapted for image segmentation and speed from: https://github.com/yu4u/mixup-generator/blob/master/random_eraser.py
    """
    img_h, img_w, _ = input_img.shape
    s = np.random.uniform(s_l, s_h) * img_h * img_w
    r = np.random.uniform(r_1, r_2)
    w = int(np.sqrt(s / r))
    h = int(np.sqrt(s * r))
    top = np.random.randint(0, img_h)
    left = np.random.randint(0, img_w)
    c = np.random.uniform(v_l, v_h)
    input_img[top:top + h, left:left + w, :] = c
    mask[top:top + h, left:left + w, :] = [1, 0]  # Set the background to true, foreground to false
    return input_img.astype(np.float32), mask.astype(np.float32)


def fliplr(image, mask):
    image = np.fliplr(image)
    mask = np.fliplr(mask)
    return image.astype(np.float32), mask.astype(np.float32)


def shift_img_lr(image, shift, roll, right, fill: Optional[Union[int, List[int]]] = None):
    if right:
        image = np.roll(image, shift, 0)
        if not roll:
            if fill is not None:
                left_fill = fill
            else:
                left_fill = np.random.uniform(0, 255, image.shape[2])
            image[:, :shift] = left_fill
    else:
        image = np.roll(image, -shift, 0)
        if not roll:
            if fill is not None:
                right_fill = fill
            else:
                right_fill = np.random.uniform(0, 255, image.shape[2])
            image[:, -shift:] = right_fill
    return image


def shift_img_ud(image, shift, roll, up, fill: Optional[Union[int, List[int]]] = None):
    if up:
        image = np.roll(image, shift, 1)
        if not roll:
            if fill is not None:
                low_fill = fill
            else:
                low_fill = np.random.uniform(0, 255, image.shape[2])
            image[-shift:, :] = low_fill
    else:
        image = np.roll(image, -shift, 1)
        if not roll:
            if fill is not None:
                top_fill = fill
            else:
                top_fill = np.random.uniform(0, 255, image.shape[2])
            image[:shift, :] = top_fill
    return image


def translate(image, mask, max_shift=23, mask_fill=[1, 0]):  # TODO: try larger max_shift
    """Randomly jitter an image horizontally or vertically."""
    vert = random.getrandbits(1)
    direction = random.getrandbits(1)
    shift = np.random.randint(1, max_shift + 1, 1)[0]
    roll = random.getrandbits(1)
    if vert:
        image = shift_img_ud(image, shift, roll, direction)
        mask = shift_img_ud(mask, shift, roll, direction, fill=mask_fill)
    else:
        image = shift_img_lr(image, shift, roll, direction)
        mask = shift_img_lr(mask, shift, roll, direction, fill=mask_fill)
    return image.astype(np.float32), mask.astype(np.float32)


def rotate_img_mask(image, mask, max_angle: int = 45, mask_fill=[1, 0]):
    angle = np.random.randint(-max_angle, max_angle)
    mode = random.sample(['reflect', 'constant', 'mirror', 'wrap'], 1)[0]
    reshape = False

    fill_with_noise = False

    if mode == "constant":
        if random.getrandbits(1):
            cval = -256
            fill_with_noise = True
        else:
            cval = np.random.randint(0, 256)
    else:
        cval = 0

    image = rotate(image, angle=angle, reshape=reshape, mode=mode, cval=cval)

    if mode == "constant" and fill_with_noise:
        bg = image == -256
        noise = np.random.randint(0, 256, size=image.shape)
        image[bg] = noise[bg]

    cval = -256
    mask = rotate(mask, angle=angle, reshape=reshape, mode=mode, cval=cval, order=0)
    if mode == "constant":
        bg = mask[:, :, 0] == -256
        mask[bg] = mask_fill

    return image, mask


cur_aug_funcs = [random_eraser, translate, fliplr, additive_gaussian_noise, exposure, rotate_img_mask]


class Augmenter:
    """Image segmentation augmenter."""
    def __init__(self, aug_funcs=None):
        if aug_funcs is None:
            aug_funcs = cur_aug_funcs
        self.aug_funcs = aug_funcs
        self.prob_to_return_original = 1. / (len(aug_funcs) + 1)
        print("Initialized image segmentation augmenter.")

    def apply_augmentations(self, image, mask, prob_to_return_original=0.0, return_image_mask_in_list: bool = True):  # 0.5
        if prob_to_return_original is not None:
            prob = prob_to_return_original
        else:
            prob = self.prob_to_return_original
        if np.random.rand() <= prob:
            return image, mask
        image, mask = image.copy(), mask.copy()
        shuffle(self.aug_funcs)
        # Apply some or all of them in the shuffled order
        num_to_apply = np.random.randint(1, len(self.aug_funcs) + 1)
        for fn in self.aug_funcs[:num_to_apply]:
            image, mask = fn(image, mask)
        if return_image_mask_in_list:
            return [image, mask]
        else:
            return image, mask
