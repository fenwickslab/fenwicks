import tensorflow as tf
from .utils.colab_tpu import TPU_ADDRESS
import datetime
import os


def get_tpu_estimator(n_trn, n_val, model_func, work_dir, ws_dir=None, ws_vars=None, trn_bs=128, val_bs=None):
    steps_per_epoch = n_trn // trn_bs
    if val_bs is None:
        val_bs = n_val

    cluster = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

    tpu_cfg = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=steps_per_epoch,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2)

    now = datetime.datetime.now()
    work_dir = os.path.join(work_dir,
                            f'{now.year}-{now.month:02d}-{now.day:02d}-{now.hour:02d}:{now.minute:02d}:{now.second:02d}')

    trn_cfg = tf.contrib.tpu.RunConfig(cluster=cluster, model_dir=work_dir, tpu_config=tpu_cfg)

    ws = None if ws_dir is None else tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir,
                                                                    vars_to_warm_start=ws_vars)

    return tf.contrib.tpu.TPUEstimator(model_fn=model_func, model_dir=work_dir, train_batch_size=trn_bs,
                                       eval_batch_size=val_bs, config=trn_cfg, warm_start_from=ws)


def get_clf_model_func(model_arch, opt_func, reduction=tf.losses.Reduction.MEAN):
    def model_func(features, labels, mode, params):
        phase = 1 if mode == tf.estimator.ModeKeys.TRAIN else 0
        tf.keras.backend.set_learning_phase(phase)

        model = model_arch()
        logits = model(features)

        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, reduction=reduction)
        step = tf.train.get_or_create_global_step()

        opt = opt_func()
        opt = tf.contrib.tpu.CrossShardOptimizer(opt, reduction=reduction)

        # train_op = tf.contrib.training.create_train_op(loss, optimizer)
        var = model.trainable_variables
        grads = tf.gradients(loss, var)
        with tf.control_dependencies(model.get_updates_for(features)):
            train_op = opt.apply_gradients(zip(grads, var), global_step=step)

        classes = tf.math.argmax(logits, axis=-1)
        metric_func = lambda classes, labels: {'accuracy': tf.metrics.accuracy(classes, labels)}
        tpu_metrics = (metric_func, [classes, labels])

        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, predictions={"predictions": classes},
                                               train_op=train_op, eval_metrics=tpu_metrics)

    return model_func
