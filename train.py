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


def triangle_lr_one_cycle(lr: float, step: tf.Tensor, total_steps: int, warmup_steps: int) -> tf.Tensor:
    """
    One cycle triangular learning rate schedule.

    :param lr: peak learning rate.
    :param step: global_step Tensor.
    :param total_steps: total number of training steps.
    :param warmup_steps: number of steps in the warmup phase, in which the learning rate increases linearly.
    :return: learning rate Tensor.
    """
    step = tf.cast(step, tf.float32)
    warmup_sched = lambda: step * lr / warmup_steps
    decay_sched = lambda: (total_steps - step) * lr / (total_steps - warmup_steps)
    lr_sched = tf.cond(tf.less_equal(step, warmup_steps), warmup_sched, decay_sched)
    return lr_sched


def adam_optimizer(lr_func):
    def opt_func():
        lr = lr_func()
        return tf.train.AdamOptimizer(lr)

    return opt_func


def sgd_optimizer(lr_func, mom: float = 0.9):
    def opt_func():
        lr = lr_func()
        return tf.train.MomentumOptimizer(lr, momentum=mom, use_nesterov=True)

    return opt_func


def adam_sgdr_one_cycle(total_steps: int, lr: float = 0.001):
    """
    Get Adam optimizer function with one-cycle SGD with Warm Restarts, a.k.a. cosine learning rate decay.

    :param total_steps: total number of training steps.
    :param lr: initial learning rate, also the highest value.
    :return: optimizer function satisfying the above descriptions.
    """

    def opt_func():
        step = tf.train.get_or_create_global_step()
        # todo: use tf.train.cosine_decay for one cycle
        lr_func = tf.train.cosine_decay_restarts(lr, step, total_steps)
        return tf.train.AdamOptimizer(learning_rate=lr_func)

    return opt_func


def sgd_triangle_one_cycle(total_steps: int, lr: float, warmup_steps: int, mom: float = 0.9):
    """
    SGD+momentum with triangular learning rate schedule.

    :param total_steps: total number of training steps.
    :param lr: peak learning rate.
    :param warmup_steps: number of steps in the warmup phase, in which the learning rate increases linearly.
    :param mom: momentum for SGD
    :return: optimizer function satisfying the above descriptions.
    """

    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = lambda: triangle_lr_one_cycle(lr, step, total_steps, warmup_steps)
        return tf.train.MomentumOptimizer(lr_func, momentum=mom, use_nesterov=True)

    return opt_func
