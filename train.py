import tensorflow as tf


def adam_sgdr_one_cycle(total_steps: int, lr: float = 0.001):
    def opt_func():
        step = tf.train.get_or_create_global_step()
        lr_func = tf.train.cosine_decay_restarts(lr, step, total_steps)
        return tf.train.AdamOptimizer(learning_rate=lr_func)

    return opt_func


def triangle_lr_epoch(t: int, lr: float, warmup: int, epochs: int):
    warmup_sched = lambda: t * lr / warmup
    decay_sched = lambda: (epochs - t) * lr / (epochs - warmup)
    return tf.cond(tf.less_equal(t, warmup), warmup_sched, decay_sched)


def triangle_lr(steps_per_epoch: int, lr: float, warmup: int, epochs: int):
    step = tf.train.get_or_create_global_step()
    return triangle_lr_epoch(tf.cast(step, tf.float32) / steps_per_epoch, lr, warmup, epochs)


def sgd_davidnet(lr: float = 0.4, warmup: int = 5, epochs: int = 24, batch_size: int = 512, trn_size: int = 50000,
                 momentum: float = 0.9):
    def opt_func():
        steps_per_epoch = trn_size // batch_size
        lr_func = lambda: triangle_lr(steps_per_epoch, lr, warmup, epochs) / batch_size
        return tf.train.MomentumOptimizer(lr_func, momentum=momentum, use_nesterov=True)

    return opt_func
