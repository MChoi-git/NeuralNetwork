import neuralnetmodules as nn


class Net:
    def __init__(self, num_hidden_layers, num_classes, weight_flag):
        # Linear layers
        self.lin1 = nn.LinearLayer(128, num_hidden_layers, weight_flag)
        self.lin2 = nn.LinearLayer(num_hidden_layers, num_classes, weight_flag)

        # Sigmoid activations
        self.sig1 = nn.SigmoidLayer(128)

    def backward(self, example_label, example_features, forward_values, previous_grad):
        # Making forward quantities readable
        lin1 = forward_values[0]
        sig1 = forward_values[1]
        beta_grad, lin2_grad = self.lin2.linear_backward(sig1, previous_grad)
        sig1_grad = self.sig1.sigmoid_backward(lin1, lin2_grad)
        alpha_grad, lin1_grad = self.lin1.linear_backward(example_features, sig1_grad)
        return alpha_grad, beta_grad