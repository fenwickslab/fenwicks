import tensorflow as tf
import math


def exp_decay_lr(init_lr: float, decay_steps: int, base_lr: float = 0, decay_rate: float = 1 / math.e):
    """
    Get exponential learning rate decay schedule function.

    Learning rate schedule:

    ```python
    lr = base_lr + init_lr * decay_rate ^ (global_step / decay_steps)
    ```

    :param init_lr: initial learning rate, also the highest value.
    :param decay_steps: number of steps for the learning rate to reduce by a full decay_rate.
    :param base_lr: smallest learning rate. Default: 0.
    :param decay_rate: the decay rate. Default 1/e.
    :return: learning rate schedule function satisfying the above descriptions. The function has one optional parameter:
             the training step count `step`. `step` defaults to `None`, in which case the function gets or creates
             Tensorflow's `global_step`.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.train.get_or_create_global_step()
        return base_lr + tf.train.exponential_decay(init_lr, step, decay_steps, decay_rate)

    return lr_func


def triangular_lr(init_lr: float, total_steps: int, warmup_steps: int):
    """
    One cycle triangular learning rate schedule.

    :param init_lr: peak learning rate.
    :param total_steps: total number of training steps.
    :param warmup_steps: number of steps in the warmup phase, during which the learning rate increases linearly.
    :return: learning rate schedule function satisfying the above descriptions. The function has one optional parameter:
             the training step count `step`. `step` defaults to `None`, in which case the function gets or creates
             Tensorflow's `global_step`.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.train.get_or_create_global_step()
        step = tf.cast(step, tf.float32)
        warmup_sched = lambda: step * init_lr / warmup_steps
        decay_sched = lambda: (total_steps - step) * init_lr / (total_steps - warmup_steps)
        return tf.cond(tf.less_equal(step, warmup_steps), warmup_sched, decay_sched)

    return lr_func


def cosine_lr(init_lr: float, total_steps: int):
    """
    Get Adam optimizer function with one-cycle SGD with Warm Restarts, a.k.a. cosine learning rate decay.

    :param init_lr: initial learning rate, also the highest value.
    :param total_steps: total number of training steps.
    :return: learning rate schedule function satisfying the above descriptions. The function has one optional parameter:
             the training step count `step`. `step` defaults to `None`, in which case the function gets or creates
             Tensorflow's `global_step`.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.train.get_or_create_global_step()
        return tf.train.cosine_decay_restarts(init_lr, step, total_steps)

    return lr_func


def adam_optimizer(lr_func):
    """
    Adam optimizer with a given learning rate schedule.

    :param lr_func: learning rate schedule function.
    :return: optimizer function satisfying the above descriptions.
    """

    def opt_func():
        lr = lr_func()
        return tf.train.AdamOptimizer(lr)

    return opt_func


class SGD(tf.train.MomentumOptimizer):
    def __init__(self, lr: tf.Tensor, mom: float, wd: float):
        super().__init__(lr, momentum=mom, use_nesterov=True)
        self.wd = wd

    def compute_gradients(self, loss):
        grads_and_vars = super().compute_gradients(loss)

        l = len(grads_and_vars)
        for i in range(l):
            g, v = grads_and_vars[i]
            g += v * self.wd
            grads_and_vars[i] = (g, v)

        return grads_and_vars


def sgd_optimizer(lr_func, mom: float = 0.9, wd: float = 0.0):
    """
    SGD with Nesterov momentum optimizer with a given learning rate schedule.

    :param lr_func: learning rate schedule function.
    :param mom: momentum for SGD. Default: 0.9
    :param wd: weight decay factor. Default: no weight decay.
    :return: optimizer function satisfying the above descriptions.
    """

    def opt_func():
        lr = lr_func()
        return SGD(lr, mom=mom, wd=wd)

    return opt_func
