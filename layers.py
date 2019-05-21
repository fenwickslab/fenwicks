from .imports import *


class Parallel(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.fw_layers = []

    def add(self, layer: tf.keras.layers.Layer):
        self.fw_layers.append(layer)

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        outputs = core.parallel_transforms(x, self.fw_layers)
        return tf.keras.layers.concatenate(outputs)


# todo: SequentialLayer
class Sequential(tf.keras.Model):
    """
    A sequential model (or composite layer), which executes its internal layers sequentially in the same order they are
    added. Sequential can be initialized as an empty model / layer. More layers can be added later on.
    """

    def __init__(self):
        super().__init__()
        self.fw_layers = []

    def add(self, layer: tf.keras.layers.Layer):
        self.fw_layers.append(layer)

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        return core.sequential_transforms(x, self.fw_layers)


class Scaling(tf.keras.layers.Layer):
    """
    Scaling layer, commonly used right before a Softmax activation, since Softmax is sensitive to scaling. It simply
    multiplies its input by a constant weight (not trainable), which is a hyper-parameter.
    """

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        return x * self.weight


class GlobalPools2D(Parallel):
    """
    A concatenation of GlobalMaxPooling2D and GlobalAveragePooling2D.
    """

    def __init__(self):
        super().__init__()
        self.add(tf.keras.layers.GlobalMaxPooling2D())
        self.add(tf.keras.layers.GlobalAveragePooling2D())


class DenseBN(Sequential):
    """
    A Dense layer followed by BatchNormalization, ReLU activation, and optionally Dropout.
    """

    def __init__(self, c: int, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001, drop_rate: float = 0.0, bn_before_relu=True):
        """
        :param c: number of neurons in the Dense layer.
        :param kernel_initializer: initialization method for the Dense layer.
        :param drop_rate: Dropout rate, i.e., 1-keep_probability. Default: no dropout.
        """
        super().__init__()
        self.add(tf.keras.layers.Dense(c, kernel_initializer=kernel_initializer, use_bias=False))
        bn = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps)
        relu = tf.keras.layers.Activation('relu')

        if bn_before_relu:
            self.add(bn)
            self.add(relu)
        else:
            self.add(relu)
            self.add(bn)

        if drop_rate > 0.0:
            self.add(tf.keras.layers.Dropout(drop_rate))


class Classifier(Sequential):
    def __init__(self, n_classes: int, kernel_initializer: Union[str, Callable] = 'glorot_uniform',
                 weight: float = 1.0):
        super().__init__()
        self.add(tf.keras.layers.Dense(n_classes, kernel_initializer=kernel_initializer, use_bias=False))
        self.add(Scaling(weight))


class ConvBN(Sequential):
    """
    A Conv2D followed by BatchNormalization and ReLU activation.
    """

    def __init__(self, c: int, kernel_size=3, strides=(1, 1), kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, strides=strides,
                                        kernel_initializer=kernel_initializer, padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps))
        self.add(tf.keras.layers.Activation('relu'))


class ConvBlk(Sequential):
    """
    A block of `ConvBN` layers, followed by a pooling layer.
    """

    def __init__(self, c, pool=None, convs=1, kernel_size=3, kernel_initializer='glorot_uniform', bn_mom=0.99,
                 bn_eps=0.001):
        super().__init__()
        for i in range(convs):
            self.add(
                ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom, bn_eps=bn_eps))
        self.add(pool or tf.keras.layers.MaxPooling2D())


class ConvResBlk(ConvBlk):
    """
    A `ConvBlk` with additional residual `ConvBN` layers.
    """

    def __init__(self, c, pool=None, convs=1, res_convs=2, kernel_size=3, kernel_initializer='glorot_uniform',
                 bn_mom=0.99, bn_eps=0.001):
        super().__init__(c, pool=pool, convs=convs, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                         bn_mom=bn_mom, bn_eps=bn_eps)
        self.res = []
        for i in range(res_convs):
            conv_bn = ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer, bn_mom=bn_mom,
                             bn_eps=bn_eps)
            self.res.append(conv_bn)

    def call(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        h = super().call(x)
        hh = core.sequential_transforms(h, self.res)
        return h + hh


def init_pytorch(shape, dtype=tf.float32, partition_info=None) -> tf.Tensor:
    """
    Initialize a given layer, such as Conv2D or Dense, in the same way as PyTorch.

    Args:
    :param shape: Shape of the weights in the layer to be initialized.
    :param dtype: Data type of the initial weights.
    :param partition_info: Required by Keras. Not used.
    :return: Random weights for a the given layer.
    """
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


PYTORCH_PARAMS = {'kernel_initializer': init_pytorch, 'bn_mom': 0.9, 'bn_eps': 1e-5}


class FastAiHead(Sequential):
    def __init__(self, n_classes: int):
        super().__init__()
        self.add(GlobalPools2D())
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.BatchNormalization(momentum=PYTORCH_PARAMS['bn_mom'],
                                                    epsilon=PYTORCH_PARAMS['bn_eps']))
        self.add(tf.keras.layers.Dropout(0.25))
        self.add(DenseBN(512, bn_before_relu=False, **PYTORCH_PARAMS))
        self.add(tf.keras.layers.Dropout(0.5))
        self.add(Classifier(n_classes, kernel_initializer=PYTORCH_PARAMS['kernel_initializer']))


def check_model(build_nn: Callable, h: int, w: int) -> tf.Tensor:
    model = build_nn()
    shape = [1, h, w, 3]
    test_input = tf.random.uniform(shape, minval=0, maxval=1)
    test_output = model(test_input)
    return test_output


def attention(src: tf.Tensor, dest: tf.Tensor, mask: tf.Tensor = None, n_heads: int = 1, c: int = 512,
              dropout_prob: float = 0.0, initializer_range: float = 0.02, return_2d: bool = False, bs: int = None,
              src_len: int = None, dest_len: int = None) -> tf.Tensor:
    def qkv(x, name: str, seq_len):
        x = tf.layers.dense(x, n_heads * c, name=name,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        return tf.transpose(tf.reshape(x, [bs, seq_len, n_heads, c]), [0, 2, 1, 3])

    from_shape = core.get_shape_list(src)
    to_shape = core.get_shape_list(dest)

    if len(from_shape) == 3:
        bs, src_len, dest_len = from_shape[0], from_shape[1], to_shape[1]

    from_tensor_2d = core.reshape_to_matrix(src)  # [B*F, src_c]
    to_tensor_2d = core.reshape_to_matrix(dest)  # [B*T, dest_c]

    query = qkv(from_tensor_2d, 'query', src_len)
    key = qkv(to_tensor_2d, 'key', dest_len)
    value = qkv(to_tensor_2d, 'value', dest_len)

    attention_scores = tf.matmul(query, key, transpose_b=True) / math.sqrt(float(c))  # [B, N, F, T]

    if mask is not None:  # `mask`: [B, F, T]
        mask = tf.expand_dims(mask, axis=[1])  # [B, 1, F, T]
        attention_scores += (1.0 - tf.cast(mask, tf.float32)) * -10000.0

    attention_probs = tf.nn.softmax(attention_scores)  # [B, N, F, T]
    attention_probs = F.dropout(attention_probs, dropout_prob)

    context = tf.matmul(attention_probs, value)  # [B, N, F, c]
    context = tf.transpose(context, [0, 2, 1, 3])  # [B, F, N, c]

    ret_shape = [bs * src_len, n_heads * c] if return_2d else [bs, src_len, n_heads * c]
    return tf.reshape(context, ret_shape)


# todo: tf 2.0
def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    return F.dropout(layer_norm(input_tensor, name), dropout_prob)
