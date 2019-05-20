import torch
import torch.nn as nn
from torch.autograd import Variable

import tensorflow as tf


class PytorchToKeras:
    def __init__(self, pModel, kModel):
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel
        tf.keras.backend.set_learning_phase(0)

    def __retrieve_k_layers(self):
        for i, layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self, input_size):
        input = torch.randn(input_size)
        input = Variable(input.unsqueeze(0))
        hooks = []

        def add_hooks(module):
            def hook(module, input, output):
                if hasattr(module, "weight"):
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module,
                                                                        nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)

        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self, input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for source_layer, target_layer in zip(self.__source_layers, self.__target_layers):
            weight_size = len(source_layer.weight.data.size())
            transpose_dims = []

            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)

            self.kModel.layers[target_layer].set_weights(
                [source_layer.weight.data.numpy().transpose(transpose_dims), source_layer.bias.data.numpy()])
