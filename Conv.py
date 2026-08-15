import numpy as np

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
        if not isinstance(in_channels, int):
            raise ValueError("Conv: Input Channels must be integer")
        if in_channels < 1:
            raise ValueError("Conv: Channels must be 1 or greater")

        # filters kwarg
        if not isinstance(filters, int):
            raise ValueError("Conv: Number of Filters must be integer")
        if filters < 1:
            raise ValueError("Conv: Filters must be 1 or greater")

        # filter_shape kwarg
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
        if initialization not in ["he", "xavier"]:
            raise ValueError("Conv: initialization argument must be either 'he' or 'xavier'")

        # distribution kwarg
        if not isinstance(distribution, str):
            raise ValueError("Conv: Distribution must be string")
        if distribution not in ["normal", "uniform"]:
            raise ValueError("Conv: distribution argument must be either 'normal' or 'uniform'")

        self.in_channels = in_channels
        self.filters = filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.padded_input = None
        self.output_shape = None
        self.dbias = None
        self.dweights = None
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
        input_img = np.asarray(input_img, np.float32)

        if input_img.ndim != 4:
            raise ValueError("Conv: Input image must have shape: (B,H,W,C) where B is the batch size, H and W are image's height and width, and C the channels of the image.")

        batch_size = input_img.shape[0]
        img_height = input_img.shape[1]
        img_width = input_img.shape[2]
        img_channels = input_img.shape[3]
        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1]
        
        if img_height <= 0 or img_width <= 0:
            raise ValueError("Conv: Image Height and Width must be 1 or greater")

        if batch_size <= 0 :
            raise ValueError("Conv: Batch size must be greater than 0")
        
        if img_channels != self.in_channels:
            raise ValueError(f"Conv: Layer was initialized with {self.in_channels} input channels")

        # Padding
        arr = np.zeros((batch_size, img_height + self.padding * 2, img_width + self.padding * 2, img_channels), dtype=self.dtype)   # B x (2P+img_rows) x (2P+img_col) x Channels
        arr[:, self.padding:img_height + self.padding, self.padding:img_width + self.padding, :] = input_img    # Create padded image
        self.padded_input = arr

        if Kh > self.padded_input.shape[1] or Kw > self.padded_input.shape[2]:
            raise ValueError("Conv: Filter dimensions must not be greater than padded image dimensions")

        window = np.empty((batch_size, Kh, Kw, self.in_channels), dtype=self.dtype)     # (B,Kh,Kw,C)
        output = np.empty((batch_size, int(np.floor((img_height + 2*self.padding - Kh) / self.stride)) + 1, 
                           int(np.floor((img_width + 2*self.padding - Kw) / self.stride)) + 1, self.filters), dtype=self.dtype)  # (B,H,W,F)
        self.output_shape = output.shape
        
        for row in range(0, self.padded_input.shape[1] - Kh + 1, self.stride):
            for col in range(0, self.padded_input.shape[2] - Kw + 1, self.stride):
                output_row = row // self.stride
                output_col = col // self.stride
                window = self.padded_input[:, row:row + Kh, col:col + Kw, :]    # Shape: (B, Kh, Kw, C)
                window = window[:, np.newaxis, :, :, :]   # Shape: (B,1,Kh,Kw,C)
                product = window * self.filter_weights  # Broadcasting: (B, 1, Kh, Kw, C) x (F, Kh, Kw, C) -> (B, F, Kh, Kw, C)    
                                                        # Image b: window is multiplied with each filter f element-wise. 
                                                        # Product[0,0]: image 0 window * filter 0. Product[0,1]: image 0 window * filter 1, and so on.
                result = np.sum(product, axis=(2,3,4))  # Shape: (B,F). Element result[b,f] is the convolution result of image b and filter f, at a specific spatial position
                output[:, output_row, output_col, :] = result + self.bias   # Broadcast result: (B,F) + bias: (F,)

        return output


    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.padded_input is None:
            raise ValueError("Conv: padded_input attribute wasn't initialized. Forward method needs to be called")

        if self.output_shape is None:
            raise ValueError("Conv: output_shape attribute wasn't initialized. Forward method needs to be called")
        
        if dout.shape != self.output_shape:
            raise ValueError("Conv: Dout shape must be same as forward output shape")

        dout = np.asarray(dout, dtype=self.dtype)

        Kh = self.filter_shape[0]
        Kw = self.filter_shape[1] 

        dL_db = np.sum(dout, axis=(0,1,2))  # (F,)
        dL_dw = np.zeros(self.filter_weights.shape, dtype=self.dtype)   # (F, Kh, Kw, C)
        dL_dx_padded = np.zeros(self.padded_input.shape, dtype=self.dtype)    # (B, H + 2P, W + 2P, C)

        for row in range(dout.shape[1]):
            for col in range(dout.shape[2]):
                window = self.padded_input[:, row * self.stride:row * self.stride + Kh, col * self.stride:col * self.stride + Kw, :]  # (B,Kh,Kw,C)
                # Broadcasting: (B,F,1,1,1) * (B,1,Kh,Kw,C) -> (B,F,Kh,Kw,C)
                dL_dw += np.sum(dout[:, row, col, :][:, :, np.newaxis, np.newaxis, np.newaxis] * window[:, np.newaxis, :, :, :], axis=0)

                dL_dx_padded[:, row * self.stride:row * self.stride + Kh, 
                            col * self.stride:col * self.stride + Kw, :] += np.sum(dout[:, row, col, :][:, :, np.newaxis, np.newaxis, np.newaxis] 
                            * self.filter_weights[np.newaxis, :, :, :, :], axis=1)  # Σ_f (B,F,1,1,1) * (1,F,Kh,Kw,C) -> (B,Kh,Kw,C)    

        dL_dx = dL_dx_padded[:, self.padding:self.padded_input.shape[1] - self.padding, self.padding: self.padded_input.shape[2] - self.padding, :]

        self.dweights = dL_dw
        self.dbias = dL_db
        din = dL_dx

        return din


def main():
    rng = np.random.default_rng(42)

    # Small input so the test stays easy to inspect
    X = rng.normal(size=(2, 4, 4, 2)).astype(np.float32)

    conv = Conv_layer(in_channels=2, filters=2, filter_shape=(3, 3), padding=1, stride=2, rng=rng)

    epsilon = 1e-3

    # First get analytical gradients at the original parameter/input values
    Z = conv.forward(X)
    # Artificial upstream gradient:
    # G = dL/dZ
    G = rng.normal(size=Z.shape).astype(np.float32)
    din = conv.backward(G)


    def relative_error(a, b):
        return abs(a - b) / max(1e-8, abs(a) + abs(b))


    # ============================================================
    # 1. Bias gradient check
    # ============================================================

    bias_idx = 1

    original_bias = conv.bias[bias_idx].copy()

    conv.bias[bias_idx] = original_bias + epsilon
    Z_plus = conv.forward(X)
    L_plus = np.sum(Z_plus * G)

    conv.bias[bias_idx] = original_bias - epsilon
    Z_minus = conv.forward(X)
    L_minus = np.sum(Z_minus * G)

    conv.bias[bias_idx] = original_bias

    numerical_db = (L_plus - L_minus) / (2 * epsilon)
    analytical_db = conv.dbias[bias_idx]

    print("\nBIAS CHECK")
    print("Index:", bias_idx)
    print("Analytical:", analytical_db)
    print("Numerical: ", numerical_db)
    print("Relative error:", relative_error(analytical_db, numerical_db))


    # ============================================================
    # 2. Another weight gradient check
    # ============================================================

    weight_idx = (1, 2, 1, 1)

    original_weight = conv.filter_weights[weight_idx].copy()

    conv.filter_weights[weight_idx] = original_weight + epsilon
    Z_plus = conv.forward(X)
    L_plus = np.sum(Z_plus * G)

    conv.filter_weights[weight_idx] = original_weight - epsilon
    Z_minus = conv.forward(X)
    L_minus = np.sum(Z_minus * G)

    conv.filter_weights[weight_idx] = original_weight

    numerical_dw = (L_plus - L_minus) / (2 * epsilon)
    analytical_dw = conv.dweights[weight_idx]

    print("\nWEIGHT CHECK")
    print("Index:", weight_idx)
    print("Analytical:", analytical_dw)
    print("Numerical: ", numerical_dw)
    print("Relative error:", relative_error(analytical_dw, numerical_dw))


    # ============================================================
    # 3. dX check — inner pixel
    # ============================================================

    x_idx = (0, 2, 2, 1)

    original_x = X[x_idx].copy()

    X[x_idx] = original_x + epsilon
    Z_plus = conv.forward(X)
    L_plus = np.sum(Z_plus * G)

    X[x_idx] = original_x - epsilon
    Z_minus = conv.forward(X)
    L_minus = np.sum(Z_minus * G)

    X[x_idx] = original_x

    numerical_dx = (L_plus - L_minus) / (2 * epsilon)
    analytical_dx = din[x_idx]

    print("\nINPUT CHECK — INNER")
    print("Index:", x_idx)
    print("Analytical:", analytical_dx)
    print("Numerical: ", numerical_dx)
    print("Relative error:", relative_error(analytical_dx, numerical_dx))


    # ============================================================
    # 4. dX check — border pixel
    # ============================================================

    x_idx = (0, 0, 0, 0)

    original_x = X[x_idx].copy()

    X[x_idx] = original_x + epsilon
    Z_plus = conv.forward(X)
    L_plus = np.sum(Z_plus * G)

    X[x_idx] = original_x - epsilon
    Z_minus = conv.forward(X)
    L_minus = np.sum(Z_minus * G)

    X[x_idx] = original_x

    numerical_dx = (L_plus - L_minus) / (2 * epsilon)
    analytical_dx = din[x_idx]

    print("\nINPUT CHECK — BORDER")
    print("Index:", x_idx)
    print("Analytical:", analytical_dx)
    print("Numerical: ", numerical_dx)
    print("Relative error:", relative_error(analytical_dx, numerical_dx))

if __name__ == "__main__":
    main()
