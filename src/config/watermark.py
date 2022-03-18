# -*- coding: utf-8 -*-
"""
    Contains all necessary paths
"""
from os.path import abspath, dirname, join

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))

DATA_PATH = join(ROOT_PATH, "data")
MODEL_PATH = join(ROOT_PATH, "models")

TRAIN_PATH = join(DATA_PATH, "Training")
TEST_PATH = join(DATA_PATH, "Testing")

DIMS_IMAGE = (128, 128, 1)
DIMS_MARK = (8, 8, 1)
