from explora.optimization.greedy_search import greedy_search


def stochastic_markov_blanket_discovery(estimator, data, target_variable=None, limit=None, control_variables=None):
    """
    Markov blanket discovery with stochastic greedy
    """
    selected, best_score = greedy_search(estimator, data, target_variable=target_variable, limit=limit,
                                         is_stochastic=True,
                                         control_variables=control_variables)

    return selected, best_score
