# !pip download efficientnet
# from keras import applications

import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml import PCA
from cuml.neighbors import NearestNeighbors
import tensorflow as tf
import efficientnet.tfkeras as efn
from tqdm.notebook import tqdm
import math
from shutil import copyfile
copyfile(src = "../input/tokenization/tokenization.py", dst = "../working/tokenization.py")
import tokenization
import tensorflow_hub as hub

# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Configuration
BATCH_SIZE = 8
IMAGE_SIZE = [512, 512]
# Seed
SEED = 42
# Verbosity
VERBOSE = 1
# Number of classes
N_CLASSES = 11011

# RESTRICT TENSORFLOW TO 2GB OF GPU RAM
# SO THAT WE HAVE 14GB RAM FOR RAPIDS
LIMIT = 2.0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
print('We will restrict TensorFlow to max %iGB GPU RAM'%LIMIT)
print('then RAPIDS can use %iGB GPU RAM'%(16-LIMIT))