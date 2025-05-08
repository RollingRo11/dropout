import torch
import torch.nn.functional as F
import torch.nn as nn


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


# call after embedding
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(
            B, T // self.n, C * self.n
        )  # C * n is the amount of consecutive elements we're looking for

        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


# how we run the model
class Sequential:
    def __init__(self, layers):  # pass in a list of layers
        self.layers = layers

    def __call__(self, x):  # call all the layers sequentially
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [
            p for layer in self.layers for p in layer.parameters()
        ]  # all the parameters of the child modules (hence the layer (singular) .parameters)


class Linear:
    def __init__(self, fan_in, fan_out, bias=True, scale=1):
        self.weight = (
            torch.randn((fan_in, fan_out)) / fan_in**0.5
        )  # kaiming initialization
        self.bias = torch.zeros(fan_out) if bias else None
        self.scale = scale

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out * self.scale

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class Dropout:
    def __init__(self, p=0.5, training_mode=True):
        self.p = p
        self.training = training_mode

    def __call__(self, x):
        if self.training:
            # create a random boolean matrix, and
            self.mask = torch.rand(x.shape) < self.p
            self.out = x * self.mask / self.p
        else:
            self.out = x
        return self.out

    def parameters(self):
        return []
