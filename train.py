import tensorflow as tf
import math


def adam_exp_decay(base_lr: float, init_lr: float, decay_steps: int, decay_rate: float = 1 / math.e):
    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = base_lr + tf.train.exponential_decay(init_lr, step, decay_steps, decay_rate)
        return tf.train.AdamOptimizer(lr_func)

    return opt_func


def adam_sgdr_one_cycle(total_steps: int, lr: float = 0.001):
    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = tf.train.cosine_decay_restarts(lr, step, total_steps)
        return tf.train.AdamOptimizer(learning_rate=lr_func)

    return opt_func


def triangle_lr_one_cycle(lr: float, step, total_steps: int, warmup_steps: int):
    step = tf.cast(step, tf.float32)
    warmup_sched = lambda: step * lr / warmup_steps
    decay_sched = lambda: (total_steps - step) * lr / (total_steps - warmup_steps)
    lr_sched = tf.cond(tf.less_equal(step, warmup_steps), warmup_sched, decay_sched)
    return lr_sched


def sgd_triangle_one_cycle(total_steps: int, lr: float, warmup_steps: int, mom: float = 0.9):
    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = lambda: triangle_lr_one_cycle(lr, step, total_steps, warmup_steps)
        return tf.train.MomentumOptimizer(lr_func, momentum=mom, use_nesterov=True)

    return opt_func
