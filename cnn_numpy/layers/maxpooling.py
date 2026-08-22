import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=(2,2), stride=2):
        if pool_size[0] < 1 or pool_size[1] < 1:
            raise ValueError("MaxPooling: Pooling window must contain at least 1 element")

        if stride < 1:
            raise ValueError("MaxPooling: Stride must be greater than 0")
        
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = None
        self.output_shape = None
        self.max_element_idxs = None
        self.built = False
        self.training = True
        self.dtype = np.float32


    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("MaxPool: Build expects input shape (H,W,C)")

        height, width, channels = input_shape

        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("MaxPool: Input dimensions must be greater than 0")

        Hout = int(np.floor((height - self.pool_size[0]) / self.stride)) + 1

        Wout = int(np.floor((width - self.pool_size[1]) / self.stride)) + 1

        if Hout <= 0 or Wout <= 0:
            raise ValueError("MaxPool: Pool dimensions must not be greater than input dimensions")

        self.built_shape = input_shape

        self.built = True

        return (Hout, Wout, channels)
    

    def forward(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=self.dtype)

        if input.ndim != 4:
            raise ValueError("MaxPooling: Input must be of shape (B,H,W,C)")
        
        self.input_shape = input.shape
        if self.built_shape != self.input_shape[1:]:
            raise ValueError("MaxPooling: Layer was built with different shape than forward's input")
        
        batch_size = input.shape[0]
        height = input.shape[1]
        width = input.shape[2]
        channels = input.shape[3]
        Ph = self.pool_size[0]
        Pw = self.pool_size[1] 

        if batch_size <= 0 or height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("MaxPooling: Input dimensions' size must be greater than 0")

        if Ph > height or Pw > width:
            raise ValueError("MaxPooling: Window dimensions can't be greater than input dimensions")
        
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


    def parameters(self):

        return []


    def get_weights(self):

        return []


    def set_weights(self, weights):
        pass


    def train(self):
        self.training = True


    def eval(self):
        self.training = False


    def regularizable_parameters(self):
        return []