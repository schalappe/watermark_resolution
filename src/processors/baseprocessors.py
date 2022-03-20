# -*- coding: utf-8
"""
Base preprocessor
"""
import tensorflow as tf


@tf.function
def load_image(image_path: str, height: int, width: int, extension: str) -> tf.Tensor:
    """
    Load an image

    Parameters
    ----------
    image_path: str
        path of image
    height: int
        New height of image
    width: int
        New width of images
    extension: str
        Extension of images

    Returns
    ------
    tf.Tensor: Image as tensor

    """
    # read the image from disk, decode it
    image = tf.io.read_file(image_path)
    if extension in ["jpeg", "jpg"]:
        image = tf.image.decode_jpeg(image)
    elif extension == "png":
        image = tf.image.decode_png(image)
    elif extension == "gif":
        image = tf.image.decode_gif(image)
    else:
        image = tf.image.decode_image(image)

    # expands if necessary
    if len(tf.shape(image)) < 3:
        image = tf.expand_dims(image, axis=-1)

    # resize image
    image = tf.image.resize(image, [height, width])

    # return the image
    return image
