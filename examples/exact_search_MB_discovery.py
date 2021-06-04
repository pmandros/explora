import pandas as pd
import timeit
from explora.algorithms.greedy_markov_blanket_discovery import greedy_markov_blanket_discovery
from explora.algorithms.random_markov_blanket_discovery import random_markov_blanket_discovery
from explora.algorithms.stochastic_markov_blanket_discovery import stochastic_markov_blanket_discovery
from explora.algorithms.stochastic_random_markov_blanket_discovery import stochastic_random_markov_blanket_discovery
from explora.information_theory.estimators import fraction_of_information_permutation, \
    fraction_of_information_permutation_upper_bound, mutual_information_permutation, \
    mutual_information_permutation_upper_bound
from explora.optimization.exact_search import depth_first_search
import time

data_file_mnist = "../datasets/mnist_test_10k_binary.csv"
data_file_tic = "../datasets/tic_tac_toe.csv"

data_tic = pd.read_csv(data_file_tic)
data_mnist = pd.read_csv(data_file_mnist)

# exact DFS
result, score = depth_first_search(data_tic, fraction_of_information_permutation, new_pruning_rule=False)
