import functools
import tensorflow as tf
from typing import Callable, Union, List, Tuple

from .. import layers
from .. import core
from .. import functional as F


def transformer(x: tf.Tensor, attn_mask: tf.Tensor = None, c: int = 768, num_hidden_layers=12, n_heads: int = 12,
                ff_c: int = 3072, ff_act: Callable = F.gelu, hidden_dropout_prob: float = 0.1,
                attn_dropout_prob: float = 0.1, initializer_range: float = 0.02,
                return_all_layers: bool = False) -> Union[List[tf.Tensor], tf.Tensor]:
    input_shape = core.get_shape_list(x)  # [bs, seq_len, c]
    x_2d = core.reshape_to_matrix(x)

    attn_c = c // n_heads
    bs, seq_len = input_shape[0], input_shape[1]

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope(f"layer_{layer_idx}"):
            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    attn_h = layers.attention(src=x_2d, dest=x_2d, mask=attn_mask, n_heads=n_heads, c=attn_c,
                                              dropout_prob=attn_dropout_prob, initializer_range=initializer_range,
                                              return_2d=True, bs=bs, src_len=seq_len, dest_len=seq_len)

                with tf.variable_scope("output"):
                    attn_h = tf.layers.dense(attn_h, c, kernel_initializer=tf.truncated_normal_initializer(
                        stddev=initializer_range))
                    attn_h = F.dropout(attn_h, hidden_dropout_prob)
                    attn_h = layers.layer_norm(attn_h + x_2d)

            with tf.variable_scope("intermediate"):
                ff_h = tf.layers.dense(attn_h, ff_c, activation=ff_act,
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))

            with tf.variable_scope("output"):
                h = tf.layers.dense(ff_h, c, kernel_initializer=tf.truncated_normal_initializer(
                    stddev=initializer_range))
                h = F.dropout(h, hidden_dropout_prob)
                h = layers.layer_norm(h + attn_h)
                x_2d = h
                all_layer_outputs.append(h)

    reshape_func = functools.partial(core.reshape_from_matrix, orig_shape_list=input_shape)
    return list(map(reshape_func, all_layer_outputs)) if return_all_layers else reshape_func(x_2d)


def embedding_lookup(x: tf.Tensor, vocab_size: int, c: int = 128, initializer_range: float = 0.02,
                     word_embedding_name: str = "word_embeddings", use_one_hot_embeddings: bool = False) -> Tuple[
    tf.Tensor, tf.Variable]:
    if x.shape.ndims == 2:
        x = tf.expand_dims(x, axis=[-1])  # todo: change input_shape instead of reshape
    input_shape = core.get_shape_list(x)
    x_flat = tf.reshape(x, [-1])

    embedding_table = tf.get_variable(name=word_embedding_name, shape=[vocab_size, c],
                                      initializer=tf.truncated_normal_initializer(stddev=initializer_range))

    x = tf.matmul(tf.one_hot(x_flat, depth=vocab_size), embedding_table) if use_one_hot_embeddings else tf.gather(
        embedding_table, x_flat)

    x = tf.reshape(x, input_shape[0:-1] + [input_shape[-1] * c])
    return x, embedding_table


def embedding_postprocessor(x, use_token_type: bool = False, token_type_ids=None, token_type_vocab_size: int = 16,
                            token_type_emb_name: str = "token_type_emb", use_position_embeddings: bool = True,
                            pos_emb_name: str = "pos_emb", initializer_range: float = 0.02, max_seq_len: int = 512,
                            dropout_prob: float = 0.1):
    input_shape = core.get_shape_list(x)
    bs, seq_len, c = input_shape[0], input_shape[1], input_shape[2]

    if use_token_type:
        token_type_table = tf.get_variable(name=token_type_emb_name, shape=[token_type_vocab_size, c],
                                           initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_emb = tf.matmul(one_hot_ids, token_type_table)
        token_type_emb = tf.reshape(token_type_emb, [bs, seq_len, c])
        x += token_type_emb

    if use_position_embeddings:
        full_pos_emb = tf.get_variable(name=pos_emb_name, shape=[max_seq_len, c],
                                       initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        pos_emb = tf.slice(full_pos_emb, [0, 0], [seq_len, -1])
        x += pos_emb
    return layers.layer_norm_and_dropout(x, dropout_prob)


# todo: only thing we use from src is its shape
def create_attention_mask_from_input_mask(src: tf.Tensor, dest_mask: tf.Tensor):
    src_shape = core.get_shape_list(src)  # [bs, src_len, ...]
    desk_shape = core.get_shape_list(dest_mask)  # [bs, dest_len], int32
    bs, src_len, dest_len = src_shape[0], src_shape[1], desk_shape[1]

    dest_mask = tf.cast(tf.reshape(dest_mask, [bs, 1, dest_len]), tf.float32)
    return tf.ones(shape=[bs, src_len, 1], dtype=tf.float32) * dest_mask  # [bs, src_len, dest_len]
