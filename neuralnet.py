import sys

import matplotlib.pyplot as plt
import numpy as np
import train as t
import predict as p
import seaborn as sns


def to_csv(filename, data):
    with open(filename, "w") as f:
        for entry in data:
            f.writelines(f"{entry},")


# Just a main script for neural net implementation. Captures command line arguments and calls necessary functions.
def main():
    # python neuralnet.py largeTrain.csv largeTest.csv largeTrainOutput.labels largeTestOutput.labels metrics.txt 100 200 0 0.01
    train_input = f"handout/{sys.argv[1]}"
    test_input = f"handout/{sys.argv[2]}"
    train_out = f"output/{sys.argv[3]}"
    test_out = f"output/{sys.argv[4]}"
    metrics_out = f"output/{sys.argv[5]}"
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    # Train
    neural_net, loss_list = t.train_entire_net(train_input, num_epoch, hidden_units, init_flag, learning_rate, metrics_out)
    # Predict
    trained_predictions, trained_average_loss, trained_error = p.predict(train_input, neural_net)
    tested_predictions, tested_average_loss, tested_error = p.predict(test_input, neural_net)
    # Send predictions to labels files
    to_csv(train_out, trained_predictions)
    to_csv(test_out, tested_predictions)
    # Print error and average loss metrics to the metrics file
    with open(metrics_out, "a") as f:
        f.writelines(f"Training set error: {trained_error}\n"
                     f"Training set last average loss: {trained_average_loss}\n"
                     f"Testing set error: {tested_error}\n"
                     f"Testing set average loss: {tested_average_loss}")
    # Graph the average loss across all epochs
    y_loss = np.array(loss_list)
    x_epochs = np.arange(1, len(y_loss) + 1)
    sns.set()
    loss_plot = sns.lineplot(x=x_epochs, y=y_loss)
    loss_plot.set(xlabel='Epoch', ylabel='Cross-entropy loss')
    plt.title('Cross-entropy Loss vs Epoch')
    plt.show()


if __name__ == "__main__":
    main()