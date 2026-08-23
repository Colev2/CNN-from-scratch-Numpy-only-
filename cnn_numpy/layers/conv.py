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
class Conv2D:
    def __init__(self, filters=1, filter_shape=(3,3), padding=1, stride=1, rng=None, initialization="he", distribution="normal", dtype=np.float32):
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
        self.initialization = initialization.strip().lower()
        self.distribution = distribution.strip().lower()
        self.padded_input = None
        self.output_shape = None
        self.built = False
        self.training = True

        self.weights = None
        self.dtype = np.dtype(dtype)


    def _initialize_parameters(self):
        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1]
        
        fan_in = Kh * Kw * self.in_channels
        fan_out = Kh * Kw * self.filters

        # Weights: He initialization
        if self.initialization == "he":
            if self.distribution == "normal":
                std = (2 / fan_in) ** 0.5
                self.weights = self.rng.normal(loc=0, scale=std, size=(self.filters, Kh, Kw, self.in_channels))   # (F,Kh,Kw,Ch)
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif self.distribution == "uniform":
                limit = (6 / fan_in) ** 0.5
                self.weights = self.rng.uniform(low=-limit, high=limit, size=(self.filters, Kh, Kw, self.in_channels))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        # Weights: Xavier initialization
        elif self.initialization == "xavier":
            if self.distribution == "normal":
                std = (2 / (fan_in + fan_out)) ** 0.5
                self.weights = self.rng.normal(loc=0, scale=std, size=(self.filters, Kh, Kw, self.in_channels))
                self.weights = np.asarray(self.weights, dtype=self.dtype)
            elif self.distribution == "uniform":
                limit = (6 / (fan_in + fan_out)) ** 0.5
                self.weights = self.rng.uniform(low=-limit, high=limit, size=(self.filters, Kh, Kw, self.in_channels))
                self.weights = np.asarray(self.weights, dtype=self.dtype)

        self.bias = np.zeros((self.filters), dtype=self.dtype)

        # Gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)


    def build(self, input_shape):
        # input_shape = (H,W,C)

        if len(input_shape) != 3:
            raise ValueError("Conv: Build expects input shape (H,W,C)")

        height, width, channels = input_shape

        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("Conv: Image Height, Width and Channels must be greater than 0")

        Hout = int(np.floor((height + 2 * self.padding - self.filter_shape[0]) / self.stride)) + 1
        Wout = int(np.floor((width + 2 * self.padding - self.filter_shape[1]) / self.stride)) + 1

        if Hout <= 0 or Wout <= 0:
            raise ValueError("Conv: Padded image's dimensions must be greater than kernel dimensions")
        
        self.in_channels = channels

        self.built_shape = input_shape

        self._initialize_parameters()

        self.built = True

        return (Hout, Wout, self.filters)


    def forward(self, input_img: np.ndarray) -> np.ndarray:
        input_img = np.asarray(input_img, np.float32)

        if input_img.ndim != 4:
            raise ValueError("Conv: Input image must have shape: (B,H,W,C)")

        if self.built_shape != input_img.shape[1:]:
            raise ValueError("Conv: Layer was built with different shape than forward's input")

        batch_size = input_img.shape[0]
        img_height = input_img.shape[1]
        img_width = input_img.shape[2]
        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1]
        
        if img_height <= 0 or img_width <= 0:
            raise ValueError("Conv: Image Height, Width must be greater than 0")

        if batch_size <= 0 :
            raise ValueError("Conv: Batch size must be greater than 0")

        if self.in_channels != input_img.shape[3]:
            raise ValueError(f"Conv: Layer was initialized with {self.in_channels} input channels")

        # Padding
        arr = np.zeros((batch_size, img_height + self.padding * 2, img_width + self.padding * 2, self.in_channels), dtype=self.dtype)   # (B, H + 2P, W + 2P, Ch)
        arr[:, self.padding:img_height + self.padding, self.padding:img_width + self.padding, :] = input_img    # Create padded image
        self.padded_input = arr

        if Kh > self.padded_input.shape[1] or Kw > self.padded_input.shape[2]:
            raise ValueError("Conv: Filter dimensions must not be greater than padded image dimensions")

        Hout = int(np.floor((img_height + 2 * self.padding - Kh) / self.stride)) + 1
        Wout = int(np.floor((img_width + 2 * self.padding - Kw) / self.stride)) + 1

        if Hout <= 0 or Wout <= 0:
            raise ValueError("Conv: Padded image's dimensions must be greater than kernel dimensions")
        
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

        # dL/db
        self.dbias[...] = np.sum(dout, axis=(0,1,2))  # (F,)

        # Contains as many rows as the number of input windows of size (Kh,Kw,C) and stride=stride
        flat_dout = np.reshape(dout, shape=(batch_size * Hout * Wout, -1)) # (B*Hout*Wout,F)

        # dL/dw
        # Get the B*Hout*Wout input windows used during the convolution.
        # Each window contains Kh*Kw*C elements.
        windows = np.lib.stride_tricks.sliding_window_view(self.padded_input, axis=(1,2), window_shape=(Kh,Kw))    
        windows = windows[:, ::self.stride, ::self.stride, :]   # Get windows with stride=stride. Shape: (B,Hout,Wout,C,Kh,Kw) -> B*Hout*Wout*C number of windows
        windows = windows.transpose(0, 1, 2, 4, 5, 3) # (B,Hout,Wout,Kh,Kw,C)

        # Widnow 0 -> All elements: [0, 1, 2, ..., Kh*Kw*C], window 1 -> All elements, window 2 ....
        flat_windows = np.reshape(windows, shape=(batch_size * windows.shape[1] * windows.shape[2], Kh * Kw * self.in_channels))  # (B*Hout*Wout, Kh*Kw*C)

        
        # Row f of flat_dout.T contains dL/dz for filter f. Each element of that row is the dout that corresponds to the window at b,h,w 
        # Column kh*kw*c of flat_windows contains the kh*kw*c -th element of each input window
        # For a specific weight w[f,kh,kw,c]:
        # dL/dw[f,kh,kw,c] = Σ_(b,h,w) dout[b,h,w,f] * windows[b,h,w,kh,kw,c]
        # The sum appears because the same weight is multiplied with every input pixel that it "sits on" during convolution, 
        # so its gradient receives one contribution from every output z that depends on it.
        # For example, the gradient of weight w[f=0, kh=0, kw=0, c=0] is:
        # Σ_(b,h,w) dout[b,h,w,0] * windows[b,h,w,0,0,0]
        # because that same weight of filter 0 is multiplied with every input pixel at kh=0,kw=0,c=0 of each window during convolution
        dL_dw_flat = flat_dout.T @ flat_windows    # (F,Kh*Kw*C)
        self.dweights[...] = np.reshape(dL_dw_flat, shape=(-1, Kh, Kw, self.in_channels))  # (F,Kh,Kw,C)

        # dL/dx
        # We first compute the gradient with respect to every element of every
        # convolution window. This is not yet dL/dx, because the same input pixel
        # may appear in multiple overlapping windows.
        dL_dx_padded = np.zeros_like(self.padded_input)

        # Each row contains all Kh*Kw*C weights of one filter.
        flat_weights = np.reshape(self.weights, shape=(self.filters, -1))   # (F,Kh*Kw*C)

        # For one specific window p and one specific element q=(kh,kw,c) inside it:
        # dL/dwindow[p,q] = Σ_f dL/dz[p,f] * w[f,q]
        # The sum over filters appears because the same input element inside this window contributes to the output 
        # of every filter at this spatial position. Therefore each row gives the gradients of all Kh*Kw*C elements of one convolution window.
        dwindows_flat = flat_dout @ flat_weights # (B*Hout*Wout,F) @ (F,Kh*Kw*C) -> (B*Hout*Wout,Khw*Kw*C)
        dwindows = np.reshape(dwindows_flat, shape=(batch_size, Hout, Wout, Kh, Kw, self.in_channels))  # (Β,Hout,Wout,Kh,Kw,C)


        # dwindows[:, :, :, kr, kc, :] has shape (B, Hout, Wout, C).
        # It contains the gradients of the window elements that are located at kernel position (kr, kc), for every possible convolution window.
        # The slice dL_dx_padded[:, kr:kr + Hout * self.stride:self.stride, kc:kc + Wout * self.stride:self.stride, :]
        # selects all padded-input positions that correspond to the (kr, kc) element of every convolution window, all at once.
        # It is basically the mapping from window coordinates (h, w, kr, kc) to input-pixel coordinates:
        # input_row = h * stride + kr
        # input_col = w * stride + kc
        # Operand += occurs because different window positions can refer to the same input element. For example, with stride=1:
        # window (h=0, w=0), element (kr=0, kc=1) and window (h=0, w=1), element (kr=0, kc=0) both refer to the same input pixel,
        # so their gradient contributions must be added together. Its the vectorized edition of:
        # for kr in range(Kh):
        #     for kc in range(Kw):
        #         for h in range(Hout):
        #             for w in range(Wout):
        #                 input_row = h * stride + kr
        #                 input_col = w * stride + kc
        #                 dL_dx_padded[:, input_row, input_col, :] += dwindows[:, h, w, kr, kc, :]
        for kr in range(Kh):
            for kc in range(Kw):
                # kr,kc positions of every window: [kr, kr + stride, kr + 2*stride, kr + 3*stride,...]
                # (B, Hout, Wout, C)
                dL_dx_padded[:, kr:kr + Hout * self.stride:self.stride, kc:kc + Wout * self.stride:self.stride, :] += dwindows[:, :, :, kr, kc, :]

        din = dL_dx_padded[:, self.padding:self.padded_input.shape[1] - self.padding, self.padding:self.padded_input.shape[2] - self.padding, :]  # (B,H,W,C)

        return din


    def parameters_grads(self):
        parameters_grads = [(self.weights, self.dweights), (self.bias, self.dbias)]

        return parameters_grads


    def get_weights(self):

        return [self.weights.copy(), self.bias.copy()]


    def set_weights(self, weights: list):
        self.weights[...] = weights[0]
        self.bias[...] = weights[1]


    def train(self):
        self.training = True


    def eval(self):
        self.training = False


    def decayable_parameters(self):
        return [self.weights]



