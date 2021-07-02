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

    def softmax_backward(self, softmax_forward_output_vector, predicted_vector, previous_grad):
        """Calculates the gradient wrt. the predicted vector for the backward computation of the softmax.

        :param softmax_forward_output_vector: Scalar softmax output vector from the output of the forward computation.
        :param predicted_vector: Vector containing the probability of each class for one example.
        :param previous_grad: Vector containing the previous gradient.
        :return: Gradient matrix, where the Jacobian of the softmax is scaled by the previous gradient vector.
        """
        diagonal_predicted_vector = np.diagflat(predicted_vector)
        squared_predicted_vector = np.square(predicted_vector)
        return np.multiply(previous_grad, np.subtract(diagonal_predicted_vector, squared_predicted_vector))


class CrossEntropyLayer:
    r"""Cross-entropy calculation module

            Args:
                input_features: Size of the input

            Shape:

            Attributes:


            """
    def __init__(self, input_features: int):
        self.input_features = input_features

    def cross_entropy_forward(self, real_vector, predicted_vector):
        """Calculates the cross-entropy between two vectors

        :param real_vector: One-hot vector where only the real label is 1.
        :param predicted_vector: Vector containing the probability of each class for one example.
        :return: Scalar cross-entropy value
        """
        return -np.dot(real_vector.T, np.log(predicted_vector))

    def cross_entropy_backward(self, real_vector, predicted_vector, x_entropy_forward_output_vector, previous_grad):
        """Calculates the gradient wrt. the predicted vector for the backward computation of the cross-entropy.

        :param real_vector: One-hot vector where only the real label is 1.
        :param predicted_vector: Vector containing the probability of each class for one example.
        :param x_entropy_forward_output_vector: Scalar cross-entropy value from the output of the forward computation.
        :param previous_grad: Scalar starting gradient. Assumed to be 1, from dJ/dJ = 1.
        :return: Gradient wrt. the predicted vector. Size will be equal to that of the real and predicted vectors.
        """
        # Make sure that the vectors are of the same length.
        if len(real_vector) != len(predicted_vector):
            print(f"X-Entropy backward calculation error: Vector sizes do not match.\n{len(real_vector)} != {len(predicted_vector)}")
        return -previous_grad * np.true_divide(real_vector, predicted_vector)
