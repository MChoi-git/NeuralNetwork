import sys
import numpy as np


# Just a main script for neural net implementation. Captures command line arguments and calls necessary functions.
def main():
    # python neuralnet.py largeTrain.csv largeTest.csv largeTrainOutput.labels largeTestOutput.labels metrics.txt 100 50 2 0.01
    train_input = f"handout/{sys.argv[1]}"
    test_input = f"handout/{sys.argv[2]}"
    train_out = f"output/{sys.argv[3]}"
    test_out = f"output/{sys.argv[4]}"
    metrics_out = f"output/{sys.argv[5]}"
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])


if __name__ == "__main__":
    main()