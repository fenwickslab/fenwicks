from .imports import *

import datetime
import collections

from .utils.colab import TPU_ADDRESS
from .optim import SGD, Adam


def exp_decay_lr(init_lr: float, decay_steps: int, base_lr: float = 0, decay_rate: float = 1 / math.e) -> Callable:
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
        """
        Get the learning rate at the current global step.
        :param step: An optional tensor in place of the global step (used in visualization). Default: None, i.e.,
                     use global step (used in training).
        :return: Learning rate tensor.
        """
        if step is None:
            step = tf.train.get_or_create_global_step()
        return base_lr + tf.train.exponential_decay(init_lr, step, decay_steps, decay_rate)

    return lr_func


def warmup_lr_sched(step: tf.Tensor, warmup_steps: int, init_lr: float, lr) -> tf.Tensor:
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
    warmup_lr = init_lr * step / warmup_steps
    is_warmup = tf.cast(step < warmup_steps, tf.float32)
    return (1.0 - is_warmup) * lr + is_warmup * warmup_lr


def linear_decay() -> Callable:
    return functools.partial(tf.train.polynomial_decay, end_learning_rate=0.0, power=1.0, cycle=False)


def one_cycle_lr(init_lr: float, total_steps: int, warmup_steps: int, decay_sched: Callable) -> Callable:
    """
    One cycle learning rate schedule.

    :param init_lr: peak learning rate.
    :param total_steps: total number of training steps.
    :param warmup_steps: number of steps in the warmup phase, during which the learning rate increases linearly.
    :param decay_sched: learning rate decay function.
    :return: learning rate schedule function satisfying the above descriptions.
    """

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        """
        Get the learning rate at the current global step.
        :param step: An optional tensor in place of the global step (used in visualization). Default: None, i.e.,
                     use global step (used in training).
        :return: Learning rate tensor.
        """

        if step is None:
            step = tf.train.get_or_create_global_step()

        lr = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
        lr = decay_sched(lr, step - warmup_steps, total_steps - warmup_steps)
        return lr if warmup_steps == 0 else warmup_lr_sched(step, warmup_steps, init_lr, lr)

    return lr_func


def adam_optimizer(lr_func: Callable, wd: float = None, beta1=0.9, beta2=0.999, epsilon=1e-8,
                   exclude_from_wd: List[int] = None, clip_norm: float = None) -> Callable:
    def opt_func():
        return Adam(lr_func(), wd=wd, beta1=beta1, beta2=beta2, epsilon=epsilon, exclude_from_wd=exclude_from_wd,
                    clip_norm=clip_norm)

    return opt_func


def sgd_optimizer(lr_func: Callable, mom: float = 0.9, wd: float = 0.0) -> Callable:
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


def weight_decay_loss(wd: float = 0.0005) -> tf.Tensor:
    l2_loss = []
    for v in tf.trainable_variables():
        if 'BatchNorm' not in v.name and 'weights' in v.name:
            l2_loss.append(tf.nn.l2_loss(v))
    return wd * tf.add_n(l2_loss)


def get_tpu_estimator(steps_per_epoch: int, model_func, work_dir: str, ws_dir: str = None, ws_vars: List[str] = None,
                      trn_bs: int = 128, val_bs: int = 1, pred_bs: int = 1, use_tpu: bool = True,
                      use_time_in_work_dir: bool = True) -> tf.contrib.tpu.TPUEstimator:
    """
    Create a TPUEstimator object ready for training and evaluation.

    :param steps_per_epoch: Number of training steps for each epoch.
    :param model_func: Model function for TPUEstimator. Can be built with `get_clf_model_func'.
    :param work_dir: Directory for storing intermediate files (such as checkpoints) generated during training.
    :param ws_dir: Directory containing warm start files, usually a pre-trained model checkpoint.
    :param ws_vars: List of warm start variables, usually from a pre-trained model.
    :param trn_bs: Batch size for training.
    :param val_bs: Batch size for validation. Default: all validation records in a single batch.
    :param pred_bs: Batch size for prediction. Default: 1.
    :param use_tpu: Whether to use TPU. Default: True.
    :param use_time_in_work_dir: Whether to use a subdirectory of `work_dir` using current time. Default: True.
    :return: A TPUEstimator object, for training, evaluation and prediction.
    """

    use_tpu = use_tpu and (TPU_ADDRESS is not None)
    cluster = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS) if use_tpu else None

    tpu_cfg = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=steps_per_epoch,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2)

    if use_time_in_work_dir:
        now = datetime.datetime.now()
        time_str = f'{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}:{now.minute:02d}:{now.second:02d}'
        work_dir = os.path.join(work_dir, time_str)

    trn_cfg = tf.contrib.tpu.RunConfig(cluster=cluster, model_dir=work_dir, tpu_config=tpu_cfg)

    ws = None if ws_dir is None else tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir,
                                                                    vars_to_warm_start=ws_vars)

    return tf.contrib.tpu.TPUEstimator(use_tpu=use_tpu, model_fn=model_func, model_dir=work_dir,
                                       train_batch_size=trn_bs, eval_batch_size=val_bs, predict_batch_size=pred_bs,
                                       config=trn_cfg, warm_start_from=ws)


def get_clf_model_func(model_arch: Callable, opt_func: Callable, reduction: str = tf.losses.Reduction.MEAN,
                       use_tpu: bool = True, init_ckpt: str = None) -> Callable:
    """
    Build a model function for a classification task to be used in a TPUEstimator, based on a given model architecture
    and an optimizer. Both the model architecture and optimizer must be callables, not model or optimizer objects. The
    reason for this design is to ensure that all variables are created in the same Tensorflow graph, which is created
    by the TPUEstimator.

    :param model_arch: Model architecture: a callable that builds a neural net model.
    :param opt_func: Optimization function: a callable that returns an optimizer.
    :param reduction: Whether to average (`tf.losses.Reduction.MEAN`) or sum (`tf.losses.Reduction.SUM`) losses
                      for different training records. Default: average.
    :param use_tpu: Whether to use TPU. Default: True.
    :param init_ckpt: Path to a pre-trained checkpoint. Default: None.
    :return: Model function ready for TPUEstimator.
    """

    use_tpu = use_tpu and (TPU_ADDRESS is not None)

    def model_func(features, labels, mode, params):
        phase = 1 if mode == tf.estimator.ModeKeys.TRAIN else 0
        tf.keras.backend.set_learning_phase(phase)

        model = model_arch()
        logits = model(features)
        y_pred = tf.math.argmax(logits, axis=-1)
        train_op = None
        loss = None

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, reduction=reduction)

        if mode == tf.estimator.ModeKeys.TRAIN:
            step = tf.train.get_or_create_global_step()

            opt = opt_func()
            if use_tpu:
                opt = tf.contrib.tpu.CrossShardOptimizer(opt, reduction=reduction)

            var = model.trainable_variables  # this excludes frozen variables
            grads_and_vars = opt.compute_gradients(loss, var_list=var)
            with tf.control_dependencies(model.get_updates_for(features)):
                train_op = opt.apply_gradients(grads_and_vars, global_step=step)

        metric_func = lambda y_pred, labels: {'accuracy': tf.metrics.accuracy(y_pred, labels)}
        tpu_metrics = (metric_func, [y_pred, labels])

        scaffold_func = None
        if init_ckpt:
            tvars = tf.trainable_variables()
            assignment_map = ckpt_assignment_map(tvars, init_ckpt)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_ckpt, assignment_map)
                    return tf.train.Scaffold()

                scaffold_func = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_ckpt, assignment_map)

        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, predictions={"y_pred": y_pred},
                                               train_op=train_op, scaffold_fn=scaffold_func, eval_metrics=tpu_metrics)

    return model_func


def ckpt_assignment_map(tvars: List, ckpg: str) -> Dict:
    """
    Compute the union of the current variables and checkpoint variables.

    :param tvars: List of trainable variables.
    :param ckpg: Path to checkpoint file.
    :return: Assignment map and list of initialized variable names.
    """
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(ckpg)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map
