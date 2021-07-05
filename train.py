import numpy as np
import model
import neuralnetmodules as nn

def init_train(filename):
    """Separate labels and features from dataset

    :param filename: Path to the dataset file
    :return: Arrays containing the labels and data
    """
    with open(filename, "r") as f:
        raw_data = np.genfromtxt(f, delimiter=',')

        print(f"Number of examples: {len(raw_data)}")
        labels = np.array([entry[0] for entry in raw_data])
        data = np.array([entry[1:] for entry in raw_data])
        return labels, data

def one_hot_vector(input, size):
    one_hot = np.zeros(size)
    one_hot[int(input) - 1] = 1
    return one_hot


def train(filename, weight_flag):
    # Initialize dataset
    labels, data = init_train(filename)

    # Initialize one-hot vector labels for cross-entropy
    onehot_labels = np.array([one_hot_vector(label, 9) for label in labels])

    # Create the net model
    net = model.Net(100, 9, weight_flag)

    # One epoch
    # Calculate softmax and cross entropy for one epoch
    output = np.array([net.forward(example) for example in data])
    soft_m = nn.SoftmaxLayer(9)
    preds = np.array([soft_m.softmax_forward(example) for example in output])
    cross_e = nn.CrossEntropyLayer(9)
    loss = np.array([cross_e.cross_entropy_forward(real, predicted) for real, predicted in zip(onehot_labels, preds)])
    print(f"Average loss across epoch: {loss.sum()/loss.size}")



train("handout/largeTrain.csv", 1)