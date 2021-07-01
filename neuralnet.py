import sys
import numpy as np


# Just a main script for neural net implementation. Captures command line arguments and calls necessary functions.

train_input = f"handout/{sys.argv[1]}"
test_input = f"handout/{sys.argv[2]}"
train_out = f"output/{sys.argv[3]}"
test_out = f"output/{sys.argv[4]}"
metrics_out = f"output/{sys.argv[5]}"
num_epoch = sys.argv[6]
hidden_units = sys.argv[7]
init_flag = sys.argv[8]
learning_rate = sys.argv[9]