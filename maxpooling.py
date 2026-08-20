import numpy as np
import time

class MaxPooling2D_layer:
    def __init__(self, pool_size=(2,2), stride=2):
        if pool_size[0] < 1 or pool_size[1] < 1:
            raise ValueError("pooling window must contain at least 1 element")

        if stride < 1:
            raise ValueError("stride must be 1 or greater")
        
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = None
        self.output_shape = None
        self.max_element_idxs = None
        self.dtype = np.float32


    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != 4:
            raise ValueError("Input in MaxPooling layer must be of shape (B,H,W,C)")
        
        batch_size = input.shape[0]
        height = input.shape[1]
        width = input.shape[2]
        channels = input.shape[3]
        Ph = self.pool_size[0]
        Pw = self.pool_size[1] 

        if batch_size <= 0 or height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("MaxPooling input dimensions' size must be 1 or greater ")

        if Ph > height or Pw > width:
            raise ValueError("MaxPooling window dimensions can't be greater than input dimensions")


        self.input_shape = input.shape
        self.output_height = int(np.floor((height - Ph) / self.stride)) + 1
        self.output_width = int(np.floor((width - Pw) / self.stride)) + 1
        self.output_shape = (batch_size, self.output_height, self.output_width, channels)  # (B,Hout,Wout,Ch)
        
        all_adjacent_windows = np.lib.stride_tricks.sliding_window_view(input, window_shape=self.pool_size, axis=(1,2))  
        pooling_windows = all_adjacent_windows[:, ::self.stride, ::self.stride, :]   # (B,Hout,Wout,Ch,Ph,Pw)
        output = np.max(pooling_windows, axis=(4,5))    # (Β,Hout,Wout,Ch)

        reshaped_windows = np.reshape(pooling_windows, shape=(batch_size,self.output_height,self.output_width,channels,-1))    # (B,Hout,Wout,Ch,Ph*Pw)
        self.max_element_idxs = np.argmax(reshaped_windows, axis=4)     # (B,Hout,Wout,Ch)

        return output



    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise ValueError("MaxPooling: input_shape attribute wasn't initialized. Forward method needs to be called")

        if self.output_shape is None:
            raise ValueError("MaxPooling: output_shape attribute wasn't initialized. Forward method needs to be called")

        if self.max_element_idxs is None:
            raise ValueError("MaxPooling: max_element_idx attribute wasn't initialized. Forward method needs to be called")

        if dout.shape != self.output_shape:
            raise ValueError("MaxPooling: Dout shape must be same as forward's output shape")

        dout = np.asarray(dout, dtype=self.dtype)
        
        din = np.zeros(self.input_shape, dtype=self.dtype)

        for pr in range(self.pool_size[0]):
            for pc in range(self.pool_size[1]):
                mask = self.max_element_idxs == pr * self.pool_size[1] + pc     # (B,Hout,Wout,Ch)
                contribution = dout * mask      # (B,Hout,Wout,Ch)
                din[:, pr:pr + self.output_height * self.stride:self.stride, pc:pc + self.output_width * self.stride:self.stride, :] += contribution

        return din


