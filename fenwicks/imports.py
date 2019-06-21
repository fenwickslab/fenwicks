from absl import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import functools
import itertools
import re
import math
import random
import gc
import json
import datetime
import collections

from typing import List, Callable, Union, Dict, Tuple, Optional, Iterator, Generator

from . import core
from . import functional as F

gfile = tf.io.gfile
logging.set_verbosity(logging.INFO)
