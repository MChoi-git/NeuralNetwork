import numpy as np
import neuralnetmodules as nn


def init_test(filename):
    """Separates labels and features from dataset. Same as train.init_train

    :param filename: Path to the dataset file
    :return: Arrays containing the labels and data
    """
    with open(filename, "r") as f:
        raw_data = np.genfromtxt(f, delimiter=',')

        print(f"Number of examples: {len(raw_data)}")
        labels = np.array([entry[0] for entry in raw_data])
        data = np.array([entry[1:] for entry in raw_data])
        return labels, data


def one_hot_vector(input_class, size):
    """Creates a one hot vector from the given index

    :param input_class: Index where one-hot value is high
    :param size: Overall size of the one-hot vector
    :return: One-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[int(input_class) - 1] = 1
    return one_hot


def predict(filename, net):
    # Initialize dataset
    labels, data = init_test(filename)
    # Initialize one-hot vector labels for cross-entropy
    one_hot_labels = np.array([one_hot_vector(label, 9) for label in labels])
    # Predict on all examples
    soft_m = nn.SoftmaxLayer(9)
    cross_e = nn.CrossEntropyLayer(9)
    predictions = []
    average_loss = 0
    for i in range(len(data)):
        forward = net.forward(data[i])
        pred = soft_m.softmax_forward(forward)
        loss = cross_e.cross_entropy_forward(one_hot_labels[i], pred)
        predictions += [pred.argmax() + 1]
        average_loss += loss
    average_loss /= len(data)
    prediction_loss = 0
    for i in range(len(labels)):
        if labels[i] != predictions[i]:
            prediction_loss += 1
    prediction_loss /= len(data)
    return predictions, average_loss, prediction_loss