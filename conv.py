import numpy as np
import time
"""
Convolutional Layer Class:

Parameters:
in_channels: channels of the input image
filters: Integer, indicating the number of filters this layer has.
filter_shape: Tuple, indicating the shape of the filters in the convolutional layer. For example: (3,3) means a 3x3 filter.
input_img: Tuple, indicating the shape of the input image on the convolutional layer. For example, (256,256,3) means a 256x256x3 image.
"""
class Conv_layer:
    def __init__(self, filters=1, filter_shape=(3,3), padding=1, stride=1, rng=None, initialization="he", distribution="normal"):
        # filters kwarg
        if not isinstance(filters, int):
            raise ValueError("Conv: Number of Filters must be integer")
        if filters < 1:
            raise ValueError("Conv: Filters must be 1 or greater")

        # filter_shape kwarg
        if not isinstance(filter_shape, tuple):
            raise ValueError("Conv: Filter shape must be a tuple")
        if not isinstance(filter_shape[0], int) or not isinstance(filter_shape[1], int):
            raise ValueError("Conv: Filter_shape must be a tuple of integers")
        if len(filter_shape) != 2:
            raise ValueError("Conv: filter_shape keyword argument must be a tuple of length 2, for instance: (3,3)")
        if filter_shape[0] < 1 or filter_shape[1] < 1:
            raise ValueError("Conv: Filter height and width must be 1 or greater")

        # padding kwarg
        if not isinstance(padding, int):
            raise ValueError("Conv: Padding must be integer")
        if padding < 0:
            raise ValueError("Conv: Padding must be 0 or greater")

        # stride kwarg
        if not isinstance(stride, int):
            raise ValueError("Conv: Stride must be integer")
        if stride < 1:
            raise ValueError("Conv: Stride must be 1 or greater")

        # rng kwarg
        if rng is None:
            rng = np.random.default_rng()

        # initialization kwarg
        if not isinstance(initialization, str):
            raise ValueError("Conv: Initialization must be string")
        if initialization.strip().lower() not in ["he", "xavier"]:
            raise ValueError("Conv: initialization argument must be either 'he' or 'xavier'")

        # distribution kwarg
        if not isinstance(distribution, str):
            raise ValueError("Conv: Distribution must be string")
        if distribution.strip().lower() not in ["normal", "uniform"]:
            raise ValueError("Conv: distribution argument must be either 'normal' or 'uniform'")

        self.in_channels = None
        self.filters = filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.rng = rng
        self.initialization = initialization
        self.distribution = distribution
        self.padded_input = None
        self.output_shape = None
        self.dtype = np.float32

        self.weights = None


    def _initialize_weights(self):
        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1]
        
        fan_in = Kh * Kw * self.in_channels
        fan_out = Kh * Kw * self.filters

        # Weights: He initialization
        if self.initialization.strip().lower() == "he":
            if self.distribution.strip().lower() == "normal":
                std = (2 / fan_in) ** 0.5
                self.weights = self.rng.normal(loc=0, scale=std, size=(self.filters, Kh, Kw, self.in_channels))   # (F,Kh,Kw,Ch)
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif self.distribution.strip().lower() == "uniform":
                limit = (6 / fan_in) ** 0.5
                self.weights = self.rng.uniform(low=-limit, high=limit, size=(self.filters, Kh, Kw, self.in_channels))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        # Weights: Xavier initialization
        elif self.initialization.strip().lower() == "xavier":
            if self.distribution.strip().lower() == "normal":
                std = (2 / (fan_in + fan_out)) ** 0.5
                self.weights = self.rng.normal(loc=0, scale=std, size=(self.filters, Kh, Kw, self.in_channels))
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif self.distribution.strip().lower() == "uniform":
                limit = (6 / (fan_in + fan_out)) ** 0.5
                self.weights = self.rng.uniform(low=-limit, high=limit, size=(self.filters, Kh, Kw, self.in_channels))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        self.bias = np.zeros((self.filters), dtype=self.dtype)


    def forward(self, input_img: np.ndarray) -> np.ndarray:
        input_img = np.asarray(input_img, np.float32)

        if input_img.ndim != 4:
            raise ValueError("Conv: Input image must have shape: (B,H,W,C) where B is the batch size, H and W are image's height and width, and C the channels of the image.")

        batch_size = input_img.shape[0]
        img_height = input_img.shape[1]
        img_width = input_img.shape[2]
        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1]
        
        if img_height <= 0 or img_width <= 0:
            raise ValueError("Conv: Image Height and Width must be 1 or greater")

        if batch_size <= 0 :
            raise ValueError("Conv: Batch size must be greater than 0")

        if self.in_channels is None:
            self.in_channels = input_img.shape[3]
        elif self.in_channels != input_img.shape[3]:
            raise ValueError(f"Conv: Layer was initialized with {self.in_channels} input channels")
        
        if self.weights is None:
            self.weight_initialization()

        # Padding
        arr = np.zeros((batch_size, img_height + self.padding * 2, img_width + self.padding * 2, self.in_channels), dtype=self.dtype)   # (B, H + 2P, W + 2P, Ch)
        arr[:, self.padding:img_height + self.padding, self.padding:img_width + self.padding, :] = input_img    # Create padded image
        self.padded_input = arr

        if Kh > self.padded_input.shape[1] or Kw > self.padded_input.shape[2]:
            raise ValueError("Conv: Filter dimensions must not be greater than padded image dimensions")

        Hout = int(np.floor((img_height + 2*self.padding - Kh) / self.stride)) + 1
        Wout = int(np.floor((img_width + 2*self.padding - Kw) / self.stride)) + 1
        self.output_shape = (batch_size, Hout, Wout, self.filters) # (B,H,W,F)

        all_adjacent_windows = np.lib.stride_tricks.sliding_window_view(self.padded_input, window_shape=(Kh,Kw), axis=(1,2))   # sliding_window_view uses stride=1 by default  
        all_strided_windows = all_adjacent_windows[:, ::self.stride, ::self.stride, :]   # (B,Hout,Wout,C,Kh,Kw)
        windows = all_strided_windows.transpose(0, 1, 2, 4, 5, 3) # (B,Hout,Wout,Kh,Kw,C)
        flat_windows = np.reshape(windows, shape=(batch_size * windows.shape[1] * windows.shape[2], Kh * Kw * self.in_channels))  # (B*Hout*Wout, Kh*Kw*C)
        flat_weights = np.reshape(self.weights, shape=(self.filters, Kh * Kw * self.in_channels))  # (F, Kh*Kw*C)
        flat_output = flat_windows @ flat_weights.T  # (B*Hout*Wout,F)
        output = np.reshape(flat_output, shape=(batch_size, Hout, Wout, -1)) + self.bias    # Broadcasting: (B,Hout,Wout,F) + (F,)

        return output


    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.padded_input is None:
            raise ValueError("Conv: padded_input attribute wasn't initialized. Forward method needs to be called")

        if self.output_shape is None:
            raise ValueError("Conv: output_shape attribute wasn't initialized. Forward method needs to be called")
        
        if dout.shape != self.output_shape:
            raise ValueError("Conv: Dout shape must be same as forward output shape")

        dout = np.asarray(dout, dtype=self.dtype)
        batch_size = dout.shape[0]
        Hout = dout.shape[1]
        Wout = dout.shape[2]
        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1] 

        flat_dout = np.reshape(dout, shape=(batch_size * Hout * Wout, -1)) # (B*Hout*Wout,F)
        windows = np.lib.stride_tricks.sliding_window_view(self.padded_input, axis=(1,2), window_shape=(Kh,Kw))     
        windows = windows[:, ::self.stride, ::self.stride, :]   # (B,Hout,Wout,C,Kh,Kw)
        windows = windows.transpose(0, 1, 2, 4, 5, 3) # (B,Hout,Wout,Kh,Kw,C)
        flat_windows = np.reshape(windows, shape=(batch_size * windows.shape[1] * windows.shape[2], Kh * Kw * self.in_channels))  # (B*Hout*Wout, Kh*Kw*C)

        dL_db = np.sum(dout, axis=(0,1,2))  # (F,)

        dL_dw_flat = flat_dout.T @ flat_windows    # (F,Kh*Kw*C)
        dL_dw = np.reshape(dL_dw_flat, shape=(-1, Kh, Kw, self.in_channels))  # (F,Kh,Kw,C)

        dL_dx_padded = np.zeros_like(self.padded_input)
        flat_weights = np.reshape(self.weights, shape=(self.filters, -1))   # (F,Kh*Kw*C)
        dwindows_flat = flat_dout @ flat_weights # (B*Hout*Wout,F) @ (F,Kh*Kw*C) -> (B*Hout*Wout,Khw*Kw*C)
        dwindows = np.reshape(dwindows_flat, shape=(batch_size, Hout, Wout, Kh, Kw, self.in_channels))

        for kr in range(Kh):
            for kc in range(Kw):
                dL_dx_padded[:, kr:kr + Hout * self.stride:self.stride, kc:kc + Wout * self.stride:self.stride, :] += dwindows[:, :, :, kr, kc, :]

        dL_dx = dL_dx_padded[:, self.padding:self.padded_input.shape[1] - self.padding, self.padding:self.padded_input.shape[2] - self.padding, :]  # (B,H,W,C)

        self.dweights = dL_dw
        self.dbias = dL_db
        din = dL_dx

        return din


def benchmark_forward_backward(forward_func, backward_func, X, dout, repeats=10, warmup=3):
    for _ in range(warmup):
        forward_func(X)
        backward_func(dout)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        forward_func(X)
        backward_func(dout)

        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.min(times)

def main():
    rng = np.random.default_rng(42)

    # Small input so the test stays easy to inspect
    X = rng.normal(size=(32, 32, 32, 32)).astype(np.float32)

    conv = Conv_layer(in_channels=32, filters=64, filter_shape=(3, 3), padding=1, stride=1, rng=rng)
    output = conv.forward_new(X)
    dout = rng.normal(size=output.shape).astype(np.float32)


    old_mean, old_min = benchmark_forward_backward(conv.forward_old, conv.backward_old, X, dout)

    new_mean, new_min = benchmark_forward_backward(conv.forward_new, conv.backward_new, X, dout)

    speedup = old_mean / new_mean

    print(f"Old forward + backward: {old_mean * 1000:.3f} ms")
    print(f"New forward + backward: {new_mean * 1000:.3f} ms")
    print(f"Total speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
