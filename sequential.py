import numpy as np
from conv import Conv_layer
from relu import ReLU_layer
from flatten import Flatten_layer
from dense import Dense_layer

class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.built = False

    def build(self, input_shape):
        if self.built:
            raise RuntimeError("Model already built")
        
        for layer in self.layers:
            input_shape = layer.build(input_shape)

        self.built = True

        return input_shape


    def forward(self, x):
        if not self.built:
            raise RuntimeError("Model must be built before forward")
        
        for layer in self.layers:
            x = layer.forward(x)

        return x        # Logits

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout     # dLogits

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params



def main():
    model = Sequential([
    Conv_layer(filters=32),
    ReLU_layer(),
    Flatten_layer(),
    Dense_layer(neurons=10)
])

    model.build((32,32,3))

    x = np.random.randn(4,32,32,3)

    out = model.forward(x)

    print(out.shape)

    dout = np.random.randn(4,10)

    dx = model.backward(dout)

    print(dx.shape)

    params = model.parameters()

    print(len(params))


if __name__ == "__main__":
    main()