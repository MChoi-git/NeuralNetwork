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


def test_softmax_backward():
    m = SoftmaxLayer(3)
    diagonal_predicted_vector = np.array([[0.25, 0, 0],
                                         [0, 0.25, 0],
                                         [0, 0, 0.5]])
    cross_product_vector = np.array([0.25**2, 0.25**2, 0.5**2])
    resultant = np.array([value - cross_product_vector for value in diagonal_predicted_vector])
    assert np.array_equal(2 * resultant, m.softmax_backward("Placeholder", np.array([0.25, 0.25, 0.5]), np.array([2, 2, 2])))


def test_cross_entropy_forward():
    m = CrossEntropyLayer(3)
    real_vector = np.array([0, 1, 0])
    predicted_vector = np.array([0.25, 0.5, 0.25])
    assert np.array_equal(m.cross_entropy_forward(real_vector, predicted_vector), scipy.stats.entropy(real_vector, predicted_vector))


def test_cross_entropy_backward():
    m = CrossEntropyLayer(3)
    real_vector = np.array([0, 1, 0])
    predicted_vector = np.array([0.25, 0.5, 0.25])
    resultant_vector = np.array([-1 * real_vector[i] / predicted_vector[i] for i in range(0, len(real_vector))])
    assert np.array_equal(resultant_vector, m.cross_entropy_backward(real_vector, predicted_vector, "placeholder", 1))