"""
Some augmentations / attacks for image
"""
from random import choice

import tensorflow as tf
from imgaug.augmenters import (
    AdditiveGaussianNoise,
    AverageBlur,
    Dropout,
    MedianBlur,
    Rotate,
    SaltAndPepper,
)

from src.addons.images import from_4D_image, get_ndims, to_4D_image
from src.addons.utils import TensorLike


def random_rotate(images: TensorLike) -> tf.Tensor:
    """
    Apply random rotation

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images`, rotated
    """
    # convert to 4D image
    image_or_images = tf.convert_to_tensor(images)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    # value of rotation
    angle = choice([15 * i for i in range(1, 6)])

    aug_func = Rotate(rotate=angle)
    return from_4D_image(
        tf.convert_to_tensor(aug_func(images=images.numpy())), original_ndims
    )


def random_dropout(images: TensorLike) -> tf.Tensor:
    """
    Apply random dropout

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images` with dropout
    """
    # convert to 4D image
    image_or_images = tf.convert_to_tensor(images)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    # value of rotation
    value = choice([0.1, 0.3, 0.5])

    aug_func = Dropout(p=value)
    return from_4D_image(
        tf.convert_to_tensor(aug_func(images=images.numpy())), original_ndims
    )


def random_average_blur(images: TensorLike) -> tf.Tensor:
    """
    Apply random average blur

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images`, blurred
    """
    # convert to 4D image
    image_or_images = tf.convert_to_tensor(images)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    # value of rotation
    kernel = choice([3, 5])

    aug_func = AverageBlur(k=kernel)
    return from_4D_image(
        tf.convert_to_tensor(aug_func(images=images.numpy())), original_ndims
    )


def random_median_blur(images: TensorLike) -> tf.Tensor:
    """
    Apply random dropout

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images`, blurred
    """
    # convert to 4D image
    image_or_images = tf.convert_to_tensor(images)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    # value of rotation
    kernel = choice([3, 5])

    aug_func = MedianBlur(k=kernel)
    return from_4D_image(
        tf.convert_to_tensor(aug_func(images=images.numpy())), original_ndims
    )


def random_gaussian_noise(images: TensorLike) -> tf.Tensor:
    """
    Apply random gaussian noise

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images`, noised
    """
    # convert to 4D image
    image_or_images = tf.convert_to_tensor(images)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    # value of rotation
    sigma = choice([0.01, 0.03])

    aug_func = AdditiveGaussianNoise(scale=sigma * 225)
    return from_4D_image(
        tf.convert_to_tensor(aug_func(images=images.numpy())), original_ndims
    )


def random_salt_pepper(images: TensorLike) -> tf.Tensor:
    """
    Apply random gaussian noise

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images`, noised
    """
    # convert to 4D image
    image_or_images = tf.convert_to_tensor(images)
    images = to_4D_image(image_or_images)
    original_ndims = get_ndims(image_or_images)

    # value of rotation
    value = choice([0.01, 0.03, 0.05])

    aug_func = SaltAndPepper(p=value)
    return from_4D_image(
        tf.convert_to_tensor(aug_func(images=images.numpy())), original_ndims
    )


def random_jpeg_quality(images: TensorLike) -> tf.Tensor:
    """
    Randomly reduce JPEG quality

    Parameters
    ----------
    images: TensorLike
        A tensor of shape
        `(num_images, num_rows, num_columns, num_channels) or
        `(num_rows, num_columns, num_channels)`

    Returns
    -------
        Image(s) with the same type and shape as `images`, reduced
    """
    # value of rotation
    value = choice([50, 70, 90])

    if len(images.shape) == 3:
        return tf.image.random_jpeg_quality(images, value - 1, value)

    return tf.map_fn(
        fn=lambda image: tf.image.random_jpeg_quality(image, value - 1, value),
        elems=images,
    )


def random_augmentation(
    images: tf.Tensor, attack: str = "", percentage: float = 0.6
) -> tf.Tensor:
    """
    Apply random augmentation if one is not specify

    Parameters
    ----------
    images: tf.Tensor
        Image to augment
    attack: str
        Attack to apply
    percentage: float
        Proportion on which to apply an attack

    Returns
    -------
    tf.Tensor:
        Augmented images
    """

    if attack == "none":
        return images

    augmentation = {
        "rotation": random_rotate,
        "dropout": random_dropout,
        "average_blur": random_average_blur,
        "median_blur": random_median_blur,
        "gaussian_noise": random_gaussian_noise,
        "salt_pepper": random_salt_pepper,
        "image_quality": random_jpeg_quality,
    }

    if attack:
        return augmentation[attack](images)

    attacked = choice(list(augmentation))
    if tf.random.uniform([], 0, 1) > percentage:
        return augmentation[attacked](images)
    return images
