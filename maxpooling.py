import numpy as np
import math
from Conv import Conv_layer
from relu import ReLU_layer

class MaxPooling2D_layer:
    def __init__(self, in_channels, pool_size=(2,2), stride=2):
        if in_channels < 1:
            raise ValueError("MaxPooling in_channels arg must be 1 or greater")

        if pool_size[0] < 1 or pool_size[1] < 1:
            raise ValueError("pooling window must contain at least 1 element")

        if stride < 1:
            raise ValueError("stride must be 1 or greater")
        
        self.in_channels = in_channels
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = None
        self.output_shape = None
        self.max_element_idx = None
        self.dtype = np.float32

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != 4:
            raise ValueError("Input in MaxPooling layer must be of shape (B,H,W,C)")

        if input.shape[-1] != self.in_channels:
            raise ValueError("MaxPooling: in_channels kwarg must be equal to input's channels")

        self.input_shape = input.shape
        output = np.empty((input.shape[0], math.floor((input.shape[1] - self.pool_size[0]) / self.stride) + 1, 
                          math.floor((input.shape[2] - self.pool_size[1]) / self.stride) + 1, self.in_channels), dtype=self.dtype)
        self.output_shape = output.shape
        self.max_element_idx = np.empty_like(output, dtype=np.intp)
        
        for row in range(0, input.shape[1] - self.pool_size[0] + 1, self.stride):
            for col in range(0, input.shape[2] - self.pool_size[1] + 1, self.stride):
                output_row = row // self.stride
                output_col = col // self.stride
                window = input[:, row:row + self.pool_size[0], col:col + self.pool_size[1], :]
                max_element = np.max(window, axis=(1,2))
                self.max_element_idx[:, output_row, output_col, :] = np.argmax(window, axis=(1,2))
                output[:, output_row, output_col, :] = max_element

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise ValueError("input_shape attribute wasn't initialized. You need to call MaxPooling forward method to do so.")

        if self.output_shape is None:
            raise ValueError("output_shape attribute wasn't initialized. You need to call MaxPooling forward method to do so.")

        if self.max_element_idx is None:
            raise ValueError("max_element_idx attribute wasn't initialized. You need to call MaxPooling forward method to do so.")

        if dout.shape != self.output_shape:
            raise ValueError("Dout shape must be equal to forward's output's shape")

        dout = np.asarray(dout, dtype=self.dtype)
        
        din = np.zeros(self.input_shape, dtype=self.dtype)

        for channel in range(dout.shape[2]):
            for output_row in range(dout.shape[0]):
                for output_col in range(dout.shape[1]):
                    input_row = output_row * self.stride
                    input_col = output_col * self.stride
                    flat_window_shape_idx = self.max_element_idx[output_row, output_col, channel]
                    window_shape_row, window_shape_col = np.unravel_index(flat_window_shape_idx, self.pool_size)
                    input_idx = (input_row + window_shape_row, input_col + window_shape_col, channel)
                    din[input_idx] += dout[output_row, output_col, channel]

        return din


def main():
    input_img = np.ones((32,32,3))
    rng = np.random.default_rng(42)

    layer1 = Conv_layer(in_channels=3, filters=16, filter_shape=(3,3), padding=1, stride=1, rng=rng)
    output1 = layer1.forward(input_img)

    layer2 = ReLU_layer()
    output2 = layer2.forward(output1)

    layer3 = MaxPooling2D_layer(in_channels=output2.shape[2], pool_size=(2,2), stride=2)
    output3 = layer3.forward(output2)

    print(output3.shape)

if __name__ == "__main__":
    main()