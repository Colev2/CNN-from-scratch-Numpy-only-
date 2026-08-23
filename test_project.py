import numpy as np

from project import create_batches, train_test_split, create_optimizer_object
from cnn_numpy.optimizers import SGD, SGD_momentum, Adam
from cnn_numpy.layers.batchnorm import BatchNorm
from cnn_numpy.layers.dense import Dense
from cnn_numpy.layers.conv import Conv2D


def test_create_batches():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    batches = list(create_batches(X, y, batch_size=4, shuffle=False))

    # 10 samples with batch size 4 -> 4, 4, 2
    assert len(batches) == 3
    assert batches[0][0].shape[0] == 4
    assert batches[1][0].shape[0] == 4
    assert batches[2][0].shape[0] == 2

    # Without shuffle, concatenating the batches
    # should give us exactly the original dataset.
    X_reconstructed = np.concatenate([X_batch for X_batch, _ in batches], axis=0)

    y_reconstructed = np.concatenate([y_batch for _, y_batch in batches], axis=0)

    assert np.array_equal(X_reconstructed, X)
    assert np.array_equal(y_reconstructed, y)


def test_train_test_split():
    X = np.arange(20).reshape(10, 2)

    # 5 samples from class 0 and 5 from class 1
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    rng = np.random.default_rng(42)

    X_train, y_train, X_val, y_val = train_test_split(X, y, validation_size=0.2, rng=rng)

    # 80% train, 20% validation
    assert len(X_train) == 8
    assert len(X_val) == 2

    # Stratification should preserve one validation
    # sample from each class.
    assert np.count_nonzero(y_val == 0) == 1
    assert np.count_nonzero(y_val == 1) == 1

    assert np.count_nonzero(y_train == 0) == 4
    assert np.count_nonzero(y_train == 1) == 4

    # No sample should disappear during the split.
    combined = np.concatenate([X_train, X_val], axis=0)

    assert sorted(map(tuple, combined)) == sorted(map(tuple, X))


def test_create_optimizer_object():
    parameter = np.array([1.0, -2.0])
    gradient = np.zeros_like(parameter)

    parameters = [(parameter, gradient)]
    regularizable_parameters = [parameter]

    optimizer = create_optimizer_object(Adam, parameters, regularizable_parameters, learning_rate=0.001, l2_lambda=1e-4)

    assert isinstance(optimizer, Adam)

    optimizer = create_optimizer_object(SGD, parameters, regularizable_parameters, learning_rate=0.001, l2_lambda=1e-4)

    assert isinstance(optimizer, SGD)

    optimizer = create_optimizer_object(SGD_momentum, parameters, regularizable_parameters, learning_rate=0.001, l2_lambda=1e-4)

    assert isinstance(optimizer, SGD_momentum)


def test_batchnorm_backward_gradient():
    rng = np.random.default_rng(42)

    # Small Dense-like BatchNorm input
    x = rng.normal(size=(4, 3)).astype(np.float64)
    dout = rng.normal(size=(4, 3)).astype(np.float64)

    bn = BatchNorm(epsilon=1e-5, momentum=0.9, dtype=np.float64)

    bn.build((3,))

    # BatchNorm must be in training mode for this check.
    bn.training = True

    # -------- Analytical gradients --------

    bn.forward(x)

    dx = bn.backward(dout).copy()
    dgamma = bn.dgamma.copy()
    dbeta = bn.dbeta.copy()

    # -------- Numerical gradients --------

    epsilon = 1e-5

    def scalar_loss():
        output = bn.forward(x)

        # If dout = dL/doutput, then using
        #
        # L = sum(output * dout)
        #
        # gives exactly that upstream gradient.
        return np.sum(output * dout)

    # ----- dx -----

    numerical_dx = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        original_value = x[index]

        x[index] = original_value + epsilon
        loss_plus = scalar_loss()

        x[index] = original_value - epsilon
        loss_minus = scalar_loss()

        x[index] = original_value

        numerical_dx[index] = (loss_plus - loss_minus) / (2 * epsilon)

    # ----- dgamma -----

    numerical_dgamma = np.zeros_like(bn.gamma)

    for i in range(bn.gamma.size):
        original_value = bn.gamma[i]

        bn.gamma[i] = original_value + epsilon
        loss_plus = scalar_loss()

        bn.gamma[i] = original_value - epsilon
        loss_minus = scalar_loss()

        bn.gamma[i] = original_value

        numerical_dgamma[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # ----- dbeta -----

    numerical_dbeta = np.zeros_like(bn.beta)

    for i in range(bn.beta.size):
        original_value = bn.beta[i]

        bn.beta[i] = original_value + epsilon
        loss_plus = scalar_loss()

        bn.beta[i] = original_value - epsilon
        loss_minus = scalar_loss()

        bn.beta[i] = original_value

        numerical_dbeta[i] = (loss_plus - loss_minus) / (2 * epsilon)

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(dgamma, numerical_dgamma, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(dbeta, numerical_dbeta, rtol=1e-5, atol=1e-6)



def test_dense_backward_gradient():
    rng = np.random.default_rng(42)

    x = rng.normal(size=(3, 4)).astype(np.float64)

    dense = Dense(neurons=2, rng=rng, initialization="xavier", distribution="normal", dtype=np.float64)

    dense.build((4,))

    out = dense.forward(x)

    dout = rng.normal(size=out.shape).astype(np.float64)

    # ---------- Analytical gradients ----------

    dx = dense.backward(dout).copy()
    dweights = dense.dweights.copy()
    dbias = dense.dbias.copy()

    epsilon = 1e-5

    def loss():
        out = dense.forward(x)
        return np.sum(out * dout)

    # ---------- Numerical dx ----------

    dx_num = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        old_value = x[idx]

        x[idx] = old_value + epsilon
        loss_plus = loss()

        x[idx] = old_value - epsilon
        loss_minus = loss()

        x[idx] = old_value

        dx_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    # ---------- Numerical dweights ----------

    dweights_num = np.zeros_like(dense.weights)

    for idx in np.ndindex(dense.weights.shape):
        old_value = dense.weights[idx]

        dense.weights[idx] = old_value + epsilon
        loss_plus = loss()

        dense.weights[idx] = old_value - epsilon
        loss_minus = loss()

        dense.weights[idx] = old_value

        dweights_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    # ---------- Numerical dbias ----------

    dbias_num = np.zeros_like(dense.bias)

    for idx in np.ndindex(dense.bias.shape):
        old_value = dense.bias[idx]

        dense.bias[idx] = old_value + epsilon
        loss_plus = loss()

        dense.bias[idx] = old_value - epsilon
        loss_minus = loss()

        dense.bias[idx] = old_value

        dbias_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    np.testing.assert_allclose(dx, dx_num, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(dweights, dweights_num, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(dbias, dbias_num, rtol=1e-5, atol=1e-6)



def test_conv_backward_gradient():
    rng = np.random.default_rng(42)

    x = rng.normal(size=(1, 3, 3, 1)).astype(np.float64)

    conv = Conv2D(filters=2, filter_shape=(2, 2), padding=0, stride=1, rng=rng, initialization="xavier", distribution="normal", dtype=np.float64)

    conv.build((3, 3, 1))

    out = conv.forward(x)

    dout = rng.normal(size=out.shape).astype(np.float64)

    # ---------- Analytical gradients ----------

    dx = conv.backward(dout).copy()
    dweights = conv.dweights.copy()
    dbias = conv.dbias.copy()

    epsilon = 1e-5

    def loss():
        out = conv.forward(x)
        return np.sum(out * dout)

    # ---------- Numerical dx ----------

    dx_num = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        old_value = x[idx]

        x[idx] = old_value + epsilon
        loss_plus = loss()

        x[idx] = old_value - epsilon
        loss_minus = loss()

        x[idx] = old_value

        dx_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    # ---------- Numerical dweights ----------

    dweights_num = np.zeros_like(conv.weights)

    for idx in np.ndindex(conv.weights.shape):
        old_value = conv.weights[idx]

        conv.weights[idx] = old_value + epsilon
        loss_plus = loss()

        conv.weights[idx] = old_value - epsilon
        loss_minus = loss()

        conv.weights[idx] = old_value

        dweights_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    # ---------- Numerical dbias ----------

    dbias_num = np.zeros_like(conv.bias)

    for idx in np.ndindex(conv.bias.shape):
        old_value = conv.bias[idx]

        conv.bias[idx] = old_value + epsilon
        loss_plus = loss()

        conv.bias[idx] = old_value - epsilon
        loss_minus = loss()

        conv.bias[idx] = old_value

        dbias_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    np.testing.assert_allclose(dx, dx_num, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(dweights, dweights_num, rtol=1e-5, atol=1e-6)

    np.testing.assert_allclose(dbias, dbias_num, rtol=1e-5, atol=1e-6)