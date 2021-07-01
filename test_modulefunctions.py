import pytest
from neuralnetmodules import *
import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy


def sigmoid(input_value):
    """Calculates the sigmoid function at an input value.

    :param input_value: Any int or float
    :return: Resultant float
    """
    return 1 / (1 + (np.exp(-input_value)))


def softmax(input_value):
    """Calculates the softmax function of an input vector.

    :param input_value:
    :return: Resultant softmax'd vector
    """
    return scipy.special.softmax(input_value)


def test_linear_forward():
    m = LinearLayer(3, 3, 0)
    m.weights = np.array([[1, 2, 3],
                          [1, 1, 1],
                          [2, 1, 2]])
    input_matrix = np.array([1, 0, 2])
    assert np.array_equal(m.linear_forward(input_matrix), np.array([7, 3, 6]))


def test_sigmoid_forward():
    m = SigmoidLayer(3)
    input_matrix = np.array([-1, 0, 1])
    assert np.array_equal(m.sigmoid_forward(input_matrix), np.array([sigmoid(-1),
                                                                     sigmoid(0),
                                                                     sigmoid(1)]))


def test_softmax_forward():
    m = SoftmaxLayer(3)
    input_vector = np.array([8, 5, 0])
    assert np.allclose(m.softmax_forward(input_vector), softmax(input_vector))


def test_crossentropy_forward():
    m = CrossEntropyLayer(3)
    real_vector = np.array([0, 1, 0])
    predicted_vector = np.array([0.25, 0.5, 0.25])
    assert np.array_equal(m.crossentropy_forward(real_vector, predicted_vector), scipy.stats.entropy(real_vector, predicted_vector))