import tensorflow as tf


def adam_sgdr_one_cycle(total_steps: int, lr: float = 0.001):
    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = tf.train.cosine_decay_restarts(lr, step, total_steps)
        return tf.train.AdamOptimizer(learning_rate=lr_func)

    return opt_func


def triangle_lr_one_cycle(lr: float, step, total_steps: int, warmup_steps: int):
    warmup_sched = lambda: step * lr / warmup_steps
    decay_sched = lambda: (total_steps - step) * lr / (total_steps - warmup_steps)
    return tf.cond(tf.less_equal(step, warmup_steps), warmup_sched, decay_sched)


def sgd_triangle_one_cycle(total_steps: int, lr: float = 0.4, warmup_steps: int = 490, mom: float = 0.9):
    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = triangle_lr_one_cycle(lr, step, total_steps, warmup_steps)
        return tf.train.MomentumOptimizer(lr_func, momentum=mom, use_nesterov=True)

    return opt_func
