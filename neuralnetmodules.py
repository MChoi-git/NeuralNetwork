import math
import numpy as np

"""
    Note: These functions are written on a per example basis. 
"""


class LinearLayer:
    r"""Linearly transforms input data using weights, where the bias term is folded into the weights

    Args:
        input_features: Size of the input
        output_features: Size of the output

    Shape:

    Attributes:
        weights: Learnable weights applied to the input to the linear connections. Matrix is initialized based on the
            weight_init_flag condition.
        weight_init_flag: Weights are initialized to zero if True. Else they are initialized to random values [-0.1, 0.1].

    """
    def __init__(self, input_features: int, output_features: int, weight_init_flag: bool):
        self.input_features = input_features
        self.output_features = output_features
        if weight_init_flag:
            self.weights = np.zeros((self.output_features, self.input_features))    # Account for folded bias term here later
        else:
            self.weights = np.random.uniform(low=-0.1, high=0.1, size=(self.output_features, self.input_features))

    def linear_forward(self, input_example):
        """Linear transform of feature vector (size M) into output vector (size j) by weight (size jxM).

        :param input_example: Numpy vector of feature values for one example
        :return: Resultant vector
        """
        return np.dot(input_example, self.weights.T)


class SigmoidLayer:
    r"""Sigmoid transformation of feature vector

    Args:
        input_features: Size of the input

    Shape:

    Attributes:


    """
    def __init__(self, input_features: int):
        self.input_features = input_features

    def sigmoid_forward(self, input_example):
        """Applies the sigmoid function to all elements of input vector. Note that an implementation with scipy.special.expit may be faster.

        :param input_example: Vector containing inputs to each hidden layer node
        :return: Vector of post-sigmoid values
        """
        return 1 / (1 + (np.exp(-input_example)))


class SoftmaxLayer:
    r"""Softmax transformation of feature vector

        Args:
            input_features: Size of the input

        Shape:

        Attributes:


        """
    def __init__(self, input_features: int):
        self.input_features = input_features

    def softmax_forward(self, input_example):
        """Calculates a softmax probability distribution over the input vector, where the probability always sums to 1.

        :param input_example: Vector of size D neural network outputs which do not sum to 1.
        :return: Vector of size D containing the resultant softmax probabilities
        """
        numerator_vector = np.exp(input_example)
        denominator_value = np.sum(np.exp(input_example))
        collection_vector = [value / denominator_value for value in numerator_vector]
        return collection_vector
