from tensorflow.python.keras.layers import Conv2D, Activation, concatenate, Input, MaxPool2D, GlobalAvgPool2D
from tensorflow.python.keras.models import Model


def fire_module(x, c_small=16, c_large=64):
    channel_axis = 3
    x = Conv2D(c_small, (1, 1), padding="valid", activation='relu')(x)
    h1 = Conv2D(c_large, (1, 1), padding="valid", activation='relu')(x)
    h2 = Conv2D(c_large, (3, 3), padding="same", activation='relu')(x)
    return concatenate([h1, h2], axis=channel_axis)


def fire_blk(h, c_small, c_large, max_pool):
    h = fire_module(h, c_small, c_large)
    h = fire_module(h, c_small, c_large)
    return MaxPool2D(pool_size=(3, 3), strides=(2, 2))(h) if max_pool else h


def SqueezeNet(input_shape=(224, 224, 3)):
    image_input = Input(shape=input_shape)

    h = Conv2D(64, (3, 3), strides=(2, 2), padding="valid", activation='relu')(image_input)
    h = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(h)

    h = fire_blk(h, 16, 64, True)
    h = fire_blk(h, 32, 128, True)
    h = fire_blk(h, 48, 192, False)
    h = fire_blk(h, 64, 256, False)

    # h = Dropout(0.5)(h)

    h = Conv2D(1000, kernel_size=(1, 1), padding="valid", name="last_conv", activation='relu')(h)
    h = GlobalAvgPool2D()(h)
    h = Activation("softmax", name="output")(h)

    return Model(inputs=image_input, outputs=h)
