import neuralnetmodules as nn


class Net:
    def __init__(self, num_hidden_layers, num_classes, weight_flag):
        # Linear layers
        self.lin1 = nn.LinearLayer(128, num_hidden_layers, weight_flag)
        self.lin2 = nn.LinearLayer(num_hidden_layers, num_classes, weight_flag)

        # Sigmoid activations
        self.sig1 = nn.SigmoidLayer(128)

    def forward(self, example_features):
        output = self.lin1.linear_forward(example_features)
        output = self.sig1.sigmoid_forward(output)
        output = self.lin2.linear_forward(output)
        return output