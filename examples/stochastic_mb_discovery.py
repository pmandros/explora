

import pandas as pd

from explora.algorithms.stochastic_markov_blanket_discovery import stochastic_markov_blanket_discovery
from explora.information_theory.estimators import fraction_of_information_permutation, fraction_of_information_permutation_upper_bound

data_file = "../datasets/tic_tac_toe.csv"
data = pd.read_csv(data_file)

result, score = stochastic_markov_blanket_discovery(fraction_of_information_permutation,data)
print(f'selected attributes with permutation {result} with score {score}')

result, score = stochastic_markov_blanket_discovery(fraction_of_information_permutation_upper_bound,data)
print(f'selected attributes with permutation upper-bound {result} with score {score}')
