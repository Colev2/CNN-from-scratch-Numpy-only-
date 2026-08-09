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
    def __init__(self, in_channels=1, filters=1, filter_shape=(3,3), padding=1, stride=1, rng=None, initialization="he", distribution="normal"):
        # in_channels kwarg
        if in_channels < 1:
            raise ValueError("Channels must be 1 or greater")
        if not isinstance(in_channels, int):
            raise ValueError("Input Channels must be integer")

        # filters kwarg
        if not isinstance(filters, int):
            raise ValueError("Number of Filters must be integer")
        if filters < 1:
            raise ValueError("Filters must be 1 or greater")

        # filter_shape kwarg
        if len(filter_shape) != 2:
            raise ValueError("filter_shape keyword argument must be a tuple of length 2, for instance: (3,3)")
        if filter_shape[0] < 1 or filter_shape[1] < 1:
            raise ValueError("Filter height and width must be 1 or greater")
        if not isinstance(filter_shape[0], int) or not isinstance(filter_shape[1], int):
            raise ValueError("Filter_shape must be a tuple of integers")

        # padding kwarg
        if padding < 0:
            raise ValueError("Padding must be 0 or greater")
        if not isinstance(padding, int):
            raise ValueError("Padding must be integer")

        # stride kwarg
        if stride < 1:
            raise ValueError("Stride must be 1 or greater")
        if not isinstance(stride, int):
            raise ValueError("Stride must be integer")

        # rng kwarg
        if rng is None:
            rng = np.random.default_rng()

        # initialization kwarg
        if initialization not in ["he", "xavier"]:
            raise ValueError("initialization argument must be either 'he' or 'xavier'")

        # distribution kwarg
        if distribution not in ["normal", "uniform"]:
            raise ValueError("distribution argument must be either 'normal' or 'uniform'")

        self.in_channels = in_channels
        self.filters = filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.dtype = np.float32

        fan_in = filter_shape[0] * filter_shape[1] * in_channels
        fan_out = filter_shape[0] * filter_shape[1] * filters

        # Weights: He initialization
        if initialization == "he":
            if distribution == "normal":
                std = (2 / fan_in) ** 0.5
                self.filter_weights = rng.normal(loc=0, scale=std, size=(filters, filter_shape[0], filter_shape[1], in_channels))   # (F,Kh,Kw,Cin)
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)
            elif distribution == "uniform":
                limit = (6 / fan_in) ** 0.5
                self.filter_weights = rng.uniform(low=-limit, high=limit, size=(filters, filter_shape[0], filter_shape[1], in_channels))
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)

        # Weights: Xavier initialization
        elif initialization == "xavier":
            if distribution == "normal":
                std = (2 / (fan_in + fan_out)) ** 0.5
                self.filter_weights = rng.normal(loc=0, scale=std, size=(filters, filter_shape[0], filter_shape[1], in_channels))
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)
            elif distribution == "uniform":
                limit = (6 / (fan_in + fan_out)) ** 0.5
                self.filter_weights = rng.uniform(low=-limit, high=limit, size=(filters, filter_shape[0], filter_shape[1], in_channels))
                self.filter_weights = np.asarray(self.filter_weights, dtype=self.dtype)

        self.bias = np.zeros((filters), dtype=self.dtype)

    def forward(self, input_img: np.ndarray) -> np.ndarray:
        if input_img.ndim != 4:
            raise ValueError("Input image must have shape: (B,H,W,C) where B is the batch size, H and W are image's height and width, and C the channels of the image.")

        self.input_img = np.asarray(input_img, np.float32)
        batch_size = self.input_img.shape[0]
        img_height = self.input_img.shape[1]
        img_width = self.input_img.shape[2]
        img_channels = self.input_img.shape[3]
        
        if img_height <= 0 or img_width <= 0:
            raise ValueError("Image Height and Width must be 1 or greater")

        if batch_size <= 0 :
            raise ValueError("Batch size must be greater than 0")
        
        if img_channels != self.in_channels:
            raise ValueError(f"Convolutional Layer was initialized with {self.in_channels} input channels")

        # Padding
        arr = np.zeros((batch_size, img_height + self.padding * 2, img_width + self.padding * 2, img_channels), dtype=self.dtype)   # B x (2P+img_rows) x (2P+img_col) x Channels
        arr[:, self.padding:img_height + self.padding, self.padding:img_width + self.padding, :] = self.input_img    # Create padded image
        self.padded_input_img = arr

        if self.filter_shape[0] > self.padded_input_img.shape[1] or self.filter_shape[1] > self.padded_input_img.shape[2    ]:
            raise ValueError("Filter dimensions must not be greater than padded image dimensions")

        output_row, output_col = 0, 0
        window = np.empty((self.filter_shape[0], self.filter_shape[1], self.in_channels), dtype=self.dtype)     # H x W x C
        self.output = np.empty((batch_size, math.floor((img_height + 2*self.padding - self.filter_shape[0]) / self.stride) + 1, 
                           math.floor((img_width + 2*self.padding - self.filter_shape[1]) / self.stride) + 1, self.filters), dtype=self.dtype)  # B x H x W x F
        
        for row in range(0, self.padded_input_img.shape[1] - self.filter_shape[0] + 1, self.stride):
            for col in range(0, self.padded_input_img.shape[2] - self.filter_shape[1] + 1, self.stride):
                output_row = row // self.stride
                output_col = col // self.stride
                window = self.padded_input_img[:, row:row + self.filter_shape[0], col:col + self.filter_shape[1], :]    # Shape: (B, Kh, Kw, C)
                window = window[:, np.newaxis, :, :, :]   # Shape: (B,1,Kh,Kw,C)
                product = window * self.filter_weights  # Broadcasting: (B, 1, Kh, Kw, C) x (F, Kh, Kw, C) -> (B, F, Kh, Kw, C)    
                                                        # Image b: window is multiplied with each filter f element-wise. 
                                                        # Product[0,0]: image 0 window * filter 0. Product[0,1]: image 0 window * filter 1, and so on.
                result = np.sum(product, axis=(2,3,4))  # Shape: (B,F). Element result[b,f] is the convolution result of image b and filter f, at a specific spatial position
                self.output[:, output_row, output_col, :] = result + self.bias   # Broadcast result: (B,F) + bias: (F,)

        return self.output


    def backward(self) -> np.ndarray:
        pass


def main():
    window = np.array([
    [[[1], [2]],
     [[3], [4]]],

    [[[5], [6]],
     [[7], [8]]]
    ], dtype=np.float32)

    window = window[:, np.newaxis, :, :, :]

    filter_weights = np.array([
    [[[1], [1]],
     [[1], [1]]],

    [[[1], [0]],
     [[0], [-1]]]
    ], dtype=np.float32)

    product = window * filter_weights


    print("window after newaxis:", window.shape)
    print("filters shape:", filter_weights.shape)
    print("product shape:", product.shape)
    print("\nImage 0 × Filter 0:")
    print(product[0, 0, :, :, 0])

    print("\nImage 0 × Filter 1:")
    print(product[0, 1, :, :, 0])

    print("\nImage 1 × Filter 0:")
    print(product[1, 0, :, :, 0])

    print("\nImage 1 × Filter 1:")
    print(product[1, 1, :, :, 0])

    result = np.sum(product, axis=(2, 3, 4))

    print("\nresult shape:", result.shape)
    print(result)

if __name__ == "__main__":
    main()
