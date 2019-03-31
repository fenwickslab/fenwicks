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
    
