from keras.layers import Conv2D, Activation, concatenate, Input, MaxPool2D, GlobalAvgPool2D
from keras.models import Model


def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):
    channel_axis = 3

    input = Conv2D(input_channel_small, (1, 1), padding="valid")(input)
    input = Activation("relu")(input)

    input_branch_1 = Conv2D(input_channel_large, (1, 1), padding="valid")(input)
    input_branch_1 = Activation("relu")(input_branch_1)

    input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
    input_branch_2 = Activation("relu")(input_branch_2)

    input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)

    return input


def SqueezeNet(input_shape=(224, 224, 3)):
    image_input = Input(shape=input_shape)

    network = Conv2D(64, (3, 3), strides=(2, 2), padding="valid")(image_input)
    network = Activation("relu")(network)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)

    # Remove layers like Dropout and BatchNormalization, they are only needed in training
    # network = Dropout(0.5)(network)

    network = Conv2D(1000, kernel_size=(1, 1), padding="valid", name="last_conv")(network)
    network = Activation("relu")(network)

    network = GlobalAvgPool2D()(network)
    network = Activation("softmax", name="output")(network)

    model = Model(inputs=image_input, outputs=network)

    return model
