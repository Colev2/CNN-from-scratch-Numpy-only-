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

        if input.shape[0] <= 0 or input.shape[1] <= 0 or input.shape[2] <= 0:
            raise ValueError("MaxPooling input dimensions' size must be 1 or greater ")

        if input.shape[-1] != self.in_channels:
            raise ValueError("MaxPooling: in_channels kwarg must be equal to input's channels")

        if self.pool_size[0] > input.shape[1] or self.pool_size[1] > input.shape[2]:
            raise ValueError("MaxPooling window dimensions can't be greater than input dimensions")

        self.input_shape = input.shape
        output = np.empty((input.shape[0], math.floor((input.shape[1] - self.pool_size[0]) / self.stride) + 1, 
                          math.floor((input.shape[2] - self.pool_size[1]) / self.stride) + 1, self.in_channels), dtype=self.dtype)
        self.output_shape = output.shape
        self.max_element_idx = np.empty_like(output, dtype=np.intp)
        
        for row in range(0, input.shape[1] - self.pool_size[0] + 1, self.stride):
            for col in range(0, input.shape[2] - self.pool_size[1] + 1, self.stride):
                output_row = row // self.stride
                output_col = col // self.stride
                window = input[:, row:row + self.pool_size[0], col:col + self.pool_size[1], :]  # (B,Kh,Kw,C)
                max_element = np.max(window, axis=(1,2))    # shape: (B,C). Element max[b,c] is the maximum of the 2D pooling 
                                                            # window of image b on channel c
                flat_window = np.reshape(window, (window.shape[0], window.shape[1] * window.shape[2], window.shape[3]))   # shape: (B,Kh*Kw,C)
                self.max_element_idx[:, output_row, output_col, :] = np.argmax(flat_window, axis=1)
                output[:, output_row, output_col, :] = max_element

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise ValueError("MaxPooling: input_shape attribute wasn't initialized. Forward method needs to be called")

        if self.output_shape is None:
            raise ValueError("MaxPooling: output_shape attribute wasn't initialized. Forward method needs to be called")

        if self.max_element_idx is None:
            raise ValueError("MaxPooling: max_element_idx attribute wasn't initialized. Forward method needs to be called")

        if dout.shape != self.output_shape:
            raise ValueError("MaxPooling: Dout shape must be same as forward's output shape")

        dout = np.asarray(dout, dtype=self.dtype)
        
        din = np.zeros(self.input_shape, dtype=self.dtype)

        batch_idx = np.arange(self.input_shape[0])[:, np.newaxis]
        channel_idx = np.arange(self.input_shape[3])[np.newaxis, :]

        for output_row in range(dout.shape[1]):
            for output_col in range(dout.shape[2]):
                input_row = output_row * self.stride
                input_col = output_col * self.stride
                flat_window_idxs = self.max_element_idx[:, output_row, output_col, :]  # Shape: (B,C). Each element is an index from 0 to Kh*Kw-1 representing the maximum 
                                                                                        # element inside the 2D window of image b and channel c. 
                window_rows, window_cols = np.unravel_index(flat_window_idxs, self.pool_size)  # (B,C) arrays. window_row(b,c) is the row index of maximum elemenet of image b and 
                                                                                            # channel c, when the window was at the spatial position (input_row, input_col)                                                                                            
                global_rows = input_row + window_rows   # Broadcasting scalar input_row and (B,C) arrays. (input_row, input_col) is the position the
                global_cols = input_col + window_cols   # window started, so to get the index of the maximum element inside that window, we need to add window_row, window_col
                                                        # to window's starting index. For instance, if window started at [H, W] = [3, 4], and the maximum
                                                        # element inside it was at the window position [1,1] (starting indexing from 0 as if its a new array)
                                                        # then the maximum element of image b and channel c is at [3 + 1, 4 + 1] = [4,5] when window starts at [3, 4]
                din[batch_idx, global_rows, global_cols, channel_idx] += dout[:, output_row, output_col, :]

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