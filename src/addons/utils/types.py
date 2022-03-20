# -*- coding: utf-8 -*-
"""
Types for typing functions signatures
"""
from typing import List, Union

import numpy as np
import tensorflow as tf

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
]


FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
