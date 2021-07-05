import neuralnetmodules as nn


class Net:
    def __init__(self, num_hidden_layers, num_classes, weight_flag):
        # Linear layers
        self.lin1 = nn.LinearLayer(128, num_hidden_layers, weight_flag)
        self.lin2 = nn.LinearLayer(num_hidden_layers, num_classes, weight_flag)

        # Sigmoid activations
        self.sig1 = nn.SigmoidLayer(128)
        self.sig2 = nn.SigmoidLayer(num_hidden_layers)

    def forward(self, dataset):
        output = self.lin1.linear_forward(dataset)
        output = self.sig1.sigmoid_forward(output)
        output = self.lin2.linear_forward(output)
        output = self.sig2.sigmoid_forward(output)
        return output