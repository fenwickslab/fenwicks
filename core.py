import functools
from typing import List


def apply_transforms(x, tfms: List):
    update_func = lambda x, y: y(x)
    return functools.reduce(update_func, tfms, x)
