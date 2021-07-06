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


def train_forward(net, example_label, example_features):
    # Get intermediate sigmoid and linear quantities
    lin1 = net.lin1.linear_forward(example_features)
    sig1 = net.sig1.sigmoid_forward(lin1)
    lin2 = net.lin2.linear_forward(sig1)
    # Get predictions from softmax
    soft_m = nn.SoftmaxLayer(9)
    pred = soft_m.softmax_forward(lin2)
    # Get cross entropy values
    cross_e = nn.CrossEntropyLayer(9)
    loss = np.array([cross_e.cross_entropy_forward(example_label, pred)])
    # print(f"Average loss across epoch: {loss.sum()/loss.size}")
    return [lin1, sig1, lin2, pred, loss]


def train_backward(net, example_label, example_features, forward_values):
    base_case_gradient = 1
    cross_e_grad = nn.CrossEntropyLayer(9).cross_entropy_backward(example_label, forward_values[3], base_case_gradient)
    soft_m_grad = nn.SoftmaxLayer(9).softmax_backward(forward_values[3], cross_e_grad)
    beta_grad, lin2_grad = net.lin2.linear_backward(forward_values[1], soft_m_grad)
    sig1_grad = net.sig1.sigmoid_backward(forward_values[1], lin2_grad)
    alpha_grad, lin1_grad = net.lin1.linear_backward(example_features, sig1_grad)
    return [alpha_grad, beta_grad]



def train_entire_net(filename, weight_flag, num_hidden_layers, num_epochs, learning_rate):
    # Initialize dataset
    labels, data = init_train(filename)
    # Initialize one-hot vector labels for cross-entropy
    one_hot_labels = np.array([one_hot_vector(label, 9) for label in labels])
    # Create the net model
    net = model.Net(num_hidden_layers, 9, weight_flag)
    # Train
    for i in range(num_epochs):
        print(f"Current epoch: {i}")
        avg_loss = 0
        for j in range(len(data)):
            forward_values = train_forward(net, one_hot_labels[j], data[j])
            weight_gradients = train_backward(net, one_hot_labels[j], data[j], forward_values)
            net.lin1.weights -= learning_rate * weight_gradients[0]
            net.lin2.weights -= learning_rate * weight_gradients[1]
            avg_loss += forward_values[4]
        avg_loss = avg_loss / len(data)
        print(f"Average loss for epoch {i} is {avg_loss}")



#Pseudo-main for debug purposes
train_entire_net("handout/largeTrain.csv", 0, 200, 100, 0.01)