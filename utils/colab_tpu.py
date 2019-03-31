from google.colab import auth
import tensorflow as tf
import os, json

TPU_ADDRESS = f'grpc://{os.environ["COLAB_TPU_ADDR"]}'

def setup_gcs():
  auth.authenticate_user()
  with tf.Session(TPU_ADDRESS) as sess:
    with open('/content/adc.json', 'r') as f:
      auth_info = json.load(f)
    tf.contrib.cloud.configure_gcs(sess, credentials=auth_info)

def get_tpu_estimator(n_trn, n_val, model_fn, model_dir,
  ws_dir, ws_vars, trn_bs=128, val_bs=None):

  steps_per_epoch = n_trn // trn_bs
  if val_bs is None:
    val_bs = n_val

  cluster = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

  tpu_cfg = tf.contrib.tpu.TPUConfig(
    iterations_per_loop=steps_per_epoch,
    per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2)

  trn_cfg = tf.contrib.tpu.RunConfig(
    cluster=cluster, model_dir=model_dir, tpu_config=tpu_cfg)

  ws = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from=ws_dir,
    vars_to_warm_start=ws_vars)

  return tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn, model_dir=model_dir,
    train_batch_size=trn_bs, eval_batch_size=val_bs,
    config=trn_cfg, warm_start_from=ws)

def get_clf_model_fn(model_arch, opt):
  def model_fn(features, labels, mode, params):
    phase = 1 if mode == tf.estimator.ModeKeys.TRAIN else 0
    tf.keras.backend.set_learning_phase(phase)

    model = model_arch()
    logits = model(features)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    step = tf.train.get_or_create_global_step()

    opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    with tf.control_dependencies(model.get_updates_for(features)):
      train_op = opt.minimize(loss, global_step=step)

    classes = tf.math.argmax(logits, axis=-1)
    metric_fn = lambda classes, labels: {'accuracy':
                                         tf.metrics.accuracy(classes, labels)}
    tpu_metrics = (metric_fn, [classes, labels])

    return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op,
                                             eval_metrics = tpu_metrics)

  return model_fn    
