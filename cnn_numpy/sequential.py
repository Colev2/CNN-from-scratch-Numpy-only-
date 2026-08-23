import numpy as np

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
    

    def parameters_grads(self):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built")
        
        parameters_grads = []
        for layer in self.layers:
            parameters_grads.extend(layer.parameters_grads())

        return parameters_grads


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


    def decayable_parameters(self):
        decayable_params = []

        for layer in self.layers:
            decayable_params.extend(layer.decayable_parameters())

        return decayable_params




