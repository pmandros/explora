import pandas as pd

from explora.algorithms.greedy_markov_blanket_discovery import greedy_markov_blanket_discovery
from explora.algorithms.random_markov_blanket_discovery import random_markov_blanket_discovery
from explora.algorithms.stochastic_markov_blanket_discovery import stochastic_markov_blanket_discovery
from explora.algorithms.stochastic_random_markov_blanket_discovery import stochastic_random_markov_blanket_discovery
from explora.algorithms.shrink import shrink
from explora.information_theory.estimators import fraction_of_information_permutation, \
    fraction_of_information_permutation_upper_bound, mutual_information_chi_square, mutual_information_permutation

data_file = "../datasets/tic_tac_toe.csv"
data = pd.read_csv(data_file)

# # standard greedy
# result, score = greedy_markov_blanket_discovery(fraction_of_information_permutation, data)
# print(f'greedy and permutation-based estimator: selected attributes {result} with score {score}')
# result, score = greedy_markov_blanket_discovery(fraction_of_information_permutation_upper_bound, data)
# print(f'greedy and permutation upper bound estimator: selected attributes {result} with score {score}')

# chi square estimator and shrink
result, score = greedy_markov_blanket_discovery(mutual_information_chi_square, data)
print(f'greedy and chi square estimator: selected attributes {result} with score {score}')
result = {x - 1 for x in result}
shrink_result, shrink_score = shrink(mutual_information_chi_square, result, data, shrink_threshold=0.01)
print(f'after shrinking with chi square estimator: selected attributes {shrink_result} with score {shrink_score}')

# # random greedy
# result, score = random_markov_blanket_discovery(fraction_of_information_permutation, data)
# print(f'random greedy and permutation-based estimator: selected attributes {result} with score {score}')
# result, score = random_markov_blanket_discovery(fraction_of_information_permutation_upper_bound, data)
# print(f'random greedy and permutation upper bound estimator: selected attributes {result} with score {score}')
#
# # stochastic standard greedy
# result, score = stochastic_markov_blanket_discovery(fraction_of_information_permutation, data)
# print(f'stochastic greedy and permutation-based estimator: selected attributes {result} with score {score}')
# result, score = stochastic_markov_blanket_discovery(fraction_of_information_permutation_upper_bound, data)
# print(f'stochastic greedy and permutation upper bound estimator: selected attributes {result} with score {score}')
#
# # stochastic random greedy
# result, score = stochastic_random_markov_blanket_discovery(fraction_of_information_permutation, data)
# print(f'stochastic random greedy and permutation-based estimator: selected attributes {result} with score {score}')
# result, score = stochastic_random_markov_blanket_discovery(fraction_of_information_permutation_upper_bound, data)
# print(f'stochastic random greedy and permutation upper bound estimator: selected attributes {result} with score {score}')

# shrinking
