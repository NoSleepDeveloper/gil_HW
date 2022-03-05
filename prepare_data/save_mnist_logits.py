
import argparse
from utils import *
from tensorflow.keras.datasets import mnist

import MNISTvgg
# import lib.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--save_file_path', default='mnist_probs.dat', type=str,
                    help='Name of file to save probs, labels pair.')

if __name__ == "__main__":
	args = parser.parse_args()
	# save_test_probs_labels(mnist, MNISTvgg.mnistvgg(), args.save_file_path)
	M = MNISTvgg.mnistvgg(train=False)
	save_test_probs_labels_mnist(mnist, M, 'mnist_probs.dat')
