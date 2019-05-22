from ..imports import *


def freeze_graph(model_dir: str, output_node_names: str, output_fn: str, overwrite: bool = False) -> Optional[str]:
    if not overwrite and gfile.exists(output_fn):
        return

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),
                                                                        output_node_names.split(","))
        with gfile.GFile(output_fn, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def


def load_graph(pb_fn: str) -> tf.Graph:
    with gfile.GFile(pb_fn, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph
