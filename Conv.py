import numpy as np
import math

"""
Convolutional Layer Class:

Parameters:
in_channels: channels of the input image
filters: Integer, indicating the number of filters this layer has.
filter_shape: Tuple, indicating the shape of the filters in the convolutional layer. For example: (3,3) means a 3x3 filter.
input_img: Tuple, indicating the shape of the input image on the convolutional layer. For example, (256,256,3) means a 256x256x3 image.
"""
class Conv_layer:
    def __init__(self, in_channels=1, filters=1, filter_shape=(3,3), padding=1, stride=1, rng=None):
        if in_channels < 1:
            raise ValueError("Channels must be 1 or greater")
        if not isinstance(in_channels, int):
            raise ValueError("Input Channels must be integer")
        
        if not isinstance(filters, int):
            raise ValueError("Number of Filters must be integer")
        if filters < 1:
            raise ValueError("Filters must be 1 or greater")
        
        if len(filter_shape) != 2:
            raise ValueError("filter_shape argument must be a tuple of length 2, for instance: (3,3)")
        if filter_shape[0] < 1 or filter_shape[1] < 1:
            raise ValueError("Filter height and width must be 1 or greater")
        if not isinstance(filter_shape[0], int) or not isinstance(filter_shape[1], int):
            raise ValueError("Filter_shape must be a tuple of integers")
        
        if padding < 0:
            raise ValueError("Padding must be 0 or greater")
        if not isinstance(padding, int):
            raise ValueError("Padding must be integer")
        
        if stride < 1:
            raise ValueError("Stride must be 1 or greater")
        if not isinstance(stride, int):
            raise ValueError("Stride must be integer")

        if rng is None:
            rng = np.random.default_rng()

        self.in_channels = in_channels
        self.filters = filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.dtype = np.float32

        # Weights: He Normal Initialization
        fan_in = filter_shape[0] * filter_shape[1] * in_channels
        std = math.sqrt(2 / fan_in)
        self.filter_weights = rng.normal(loc=0, scale=std, size=(filters, filter_shape[0], filter_shape[1], in_channels)).astype(self.dtype)    # (F,Kh,Kw,Cin)

        self.bias = np.zeros((filters), dtype=self.dtype)

    def forward(self, input_img: np.ndarray) -> np.ndarray:
        if input_img.ndim == 2:
            input_img = np.expand_dims(input_img, axis=2)   # (H,W) -> (H,W,1)
        elif input_img.ndim != 3:
            raise ValueError("Image must be of shape: (H,W,C)")
        if input_img.shape[0] <= 0 or input_img.shape[1] <= 0:
            raise ValueError("Image Height and Width must be 1 or greater")
        self.input_img = input_img.astype(np.float32)
        if self.input_img.shape[2] != self.in_channels:
            raise ValueError(f"Convolutional Layer was initialized with {self.in_channels} input channels")

        # Padding
        arr = np.zeros((self.input_img.shape[0] + self.padding * 2, self.input_img.shape[1] + self.padding * 2, self.input_img.shape[2]), 
                       dtype=self.dtype)   # (2P+img_rows)x(2P+img_col)xChannels
        arr[self.padding:self.input_img.shape[0]+self.padding, self.padding:self.input_img.shape[1]+self.padding, :] = self.input_img    # Create padded image
        self.padded_input_img = arr

        if self.filter_shape[0] > self.padded_input_img.shape[0] or self.filter_shape[1] > self.padded_input_img.shape[1]:
            raise ValueError("Filter dimensions must not be greater than padded image dimensions")

        output_row, output_col = 0, 0
        window = np.empty((self.filter_shape[0], self.filter_shape[1], self.in_channels), dtype=self.dtype)
        self.output = np.empty((math.floor((self.input_img.shape[0] + 2*self.padding - self.filter_shape[0]) / self.stride) + 1, 
                           math.floor((self.input_img.shape[1] + 2*self.padding - self.filter_shape[1]) / self.stride) + 1, self.filters), dtype=self.dtype)  # H x W x F

        for filter_idx in range(self.filters):
            for row in range(0, self.padded_input_img.shape[0] - self.filter_shape[0] + 1, self.stride):
                for col in range(0, self.padded_input_img.shape[1] - self.filter_shape[1] + 1, self.stride):
                    window = self.padded_input_img[row:row+self.filter_shape[0], col:col+self.filter_shape[1], :]
                    result = np.sum(window * self.filter_weights[filter_idx, :, :, :])      
                    self.output[output_row, output_col, filter_idx] = result + self.bias[filter_idx]
                    output_col += 1
                output_row += 1
                output_col = 0
            output_row = 0

        return self.output

    def backward(self) -> np.ndarray:
        pass


