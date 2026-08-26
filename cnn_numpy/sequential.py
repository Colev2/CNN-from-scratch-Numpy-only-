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


    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("Sequential: Model must be built before forward")
        
        for layer in self.layers:
            x = layer.forward(x)

        return x        # Logits


    def backward(self, dout: np.ndarray) -> np.ndarray:
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
            weights.append(layer.get_weights())  # weights: [[conv1_weights, conv1_bias], [conv2_weights, conv2_bias],...]

        return weights


    def set_weights(self, weights):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built")
        
        if len(weights) != len(self.layers):
            raise ValueError("Sequential: Number of weight groups must match number of layers")

        for layer, layer_weights in zip(self.layers, weights):      # https://docs.python.org/3/library/functions.html#zip
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



    def save_weights(self, filepath):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built before saving")

        model_weights = self.get_weights()

        save_dict = {}

        for layer_idx, layer_weights in enumerate(model_weights):
            for weight_idx, weight in enumerate(layer_weights):
                save_dict[f"layer_{layer_idx}_weight_{weight_idx}"] = weight    # {layer_0_weight_0: conv1.weights, 
                                                                                # layer_0_weight_1: conv1.nias,....}
        # https://numpy.org/doc/stable/reference/generated/numpy.savez.html
        np.savez(filepath, **save_dict)     # Dict unpacking to kwargs: layer_0_weight_0=conv1.weights, ...


    def load_weights(self, filepath):
        if not self.built:
            raise RuntimeError("Sequential: Model needs to be built before loading weights")

        saved_data = np.load(filepath, allow_pickle=False)  # https://numpy.org/doc/stable/reference/generated/numpy.load.html

        model_weights = []

        for layer_idx, layer in enumerate(self.layers):
            current_layer_weights = layer.get_weights()     # e.g: [con1.weights, conv1.bias]

            layer_weights = []

            for weight_idx in range(len(current_layer_weights)):
                key = f"layer_{layer_idx}_weight_{weight_idx}"

                if key not in saved_data:
                    raise ValueError(f"Sequential: Missing '{key}' in saved model")

                if saved_data[key].shape != current_layer_weights[weight_idx].shape:
                    raise ValueError(f"Sequential: Shape mismatch for '{key}': "
                        f"expected {current_layer_weights[weight_idx].shape}, got {saved_data[key].shape}"
                    )

                layer_weights.append(saved_data[key])

            model_weights.append(layer_weights)

        saved_data.close()

        self.set_weights(model_weights)
