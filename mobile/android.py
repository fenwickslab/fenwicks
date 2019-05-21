from ..imports import *


def freeze_graph(model_dir, output_node_names):
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_graph = os.path.join(input_checkpoint, "frozen_model.pb")

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),
                                                                        output_node_names.split(","))
        with gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def
