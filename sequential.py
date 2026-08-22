import numpy as np
from conv import Conv_layer
from relu import ReLU_layer
from flatten import Flatten_layer
from dense import Dense_layer

class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.built = False
        self.output_shape = None


    def build(self, input_shape):
        if self.built:
            raise RuntimeError("Model already built")
        
        for layer in self.layers:
            input_shape = layer.build(input_shape)

        self.built = True

        self.output_shape = input_shape

        return self.output_shape


    def forward(self, x):
        if not self.built:
            raise RuntimeError("Sequential: Model must be built before forward")
        
        for layer in self.layers:
            x = layer.forward(x)

        return x        # Logits


    def backward(self, dout):
        if not self.built:
            raise RuntimeError("Sequential: Model must be built before backward")
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout   
    

    def parameters(self):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built")
        
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params


    def get_weights(self):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built")
        
        weights = []

        for layer in self.layers:
            weights.append(layer.get_weights())

        return weights


    def set_weights(self, weights):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built")
        
        if len(weights) != len(self.layers):
            raise ValueError("Sequential: Number of weight groups must match number of layers")

        for layer, layer_weights in zip(self.layers, weights):
            layer.set_weights(layer_weights)


    def train(self):
        for layer in self.layers:
            layer.train()


    def eval(self):
        for layer in self.layers:
            layer.eval()


    def regularizable_parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.regularizable_parameters())

        return params


    def l2_loss(self, l2_lambda):
        loss = 0

        for w in self.regularizable_parameters():
            loss += np.sum(w ** 2)

        return l2_lambda * loss


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