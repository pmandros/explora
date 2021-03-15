from explora.optimization.greedy_search import greedy_search


def random_markov_blanket_discovery(estimator, data, target_variable=None, limit=None, control_variables=None):
    """
    Markov blanket discovery with random greedy (uniformly selects from top-3)
    """
    selected, best_score = greedy_search(estimator, data, target_variable=target_variable, limit=limit,
                                         control_variables=control_variables, select_from_top_k=3)

    return selected, best_score
