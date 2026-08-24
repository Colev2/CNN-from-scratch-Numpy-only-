import numpy as np

def random_horizontal_flip(X, rng, probability=0.5):
    X = X.copy()    # To not change the original dataset

    flip_mask = rng.random(X.shape[0]) < probability    # ndarray([False, True, False,...]) with each element being true with prob=0.5
    X[flip_mask] = X[flip_mask, :, ::-1, :]     # swap columns

    return X


def random_crop(X, rng, padding=4):
    B, H, W, C = X.shape

    padded = np.pad(X,((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant")    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    output = np.empty_like(X)

    for b in range(B):
        top = rng.integers(0, 2 * padding + 1)  # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html
        left = rng.integers(0, 2 * padding + 1)

        output[b] = padded[b, top:top + H, left:left + W, :]

    return output


def augment_batch(X, rng, crop_padding=4, horizontal_flip=True):
    if crop_padding > 0:
        X = random_crop(X, rng, padding=crop_padding)
    
    if horizontal_flip:
        X = random_horizontal_flip(X, rng, probability=0.5)

    return X