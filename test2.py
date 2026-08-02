import numpy as np
from Conv import Conv_layer
from relu import ReLU_layer

def main():
    rng = np.random.default_rng(42)
    image = np.ones((32,32,3))
    layer1 = Conv_layer(in_channels=3, filters=16, filter_shape = (3,3), padding=1, stride=1, rng=rng)
    output1 = layer1.forward(image)
    print(output1[:, :, 0])
    layer2 = ReLU_layer()
    output2 = layer2.forward(output1)
    print(layer2.positive_input_mask[:, :, 0])

if __name__ == "__main__":
    main()