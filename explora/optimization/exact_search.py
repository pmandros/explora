import numpy as np
import pandas as pd
import time

from explora.optimization.refinement_and_evaluation import refine_evaluate_get_optimistic_value_sort
from explora.information_theory.information_theory_basic import entropy_plugin


def depth_first_search(data, estimator, target_variable=None, new_pruning_rule=True):
    estimator_name = estimator.__name__
    if 'fraction' not in estimator_name.lower():
        raise ValueError('Optimistic estimator does not apply to non fraction of information scores. To use exact'
                         ' search specify a fraction of information estimator')
    start_time = time.time()

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # the number of attributes excluding the target_variable
    number_explanatory_variables = np.size(data, 1) - 1

    if target_variable is None:
        target_variable_index = number_explanatory_variables
    else:
        target_variable_index = target_variable - 1
    target_data_column = data[:, target_variable_index]
    entropy_target = entropy_plugin(target_data_column)

    candidate_variables_indices = list(range(number_explanatory_variables + 1))
    candidate_variables_indices.remove(target_variable_index)

    root_node = list()
    root_node_refinement_elements = candidate_variables_indices.copy()
    root_node_score = -float('inf')

    best_solution = root_node
    best_score = root_node_score

    queue = list()
    queue.append((root_node, root_node_refinement_elements, root_node_score))

    num_pruned = 0

    while queue:
        top_queue_element = queue.pop()
        candidate_to_refine = top_queue_element[0]  # a list of attribute indices
        candidate_refinement_elements = top_queue_element[1]  # a list of attribute indices
        candidate_score = top_queue_element[2]

        # create refinements, evaluate them, get their optimistic values
        # a tuple of (score, refinement_element_index, opt_value)
        refinements = refine_evaluate_get_optimistic_value_sort(estimator,
                                                                data, target_data_column, candidate_to_refine,
                                                                candidate_refinement_elements)

        # get best
        best_refinement = refinements[0]
        best_refinement_score = best_refinement[0]
        best_refinement_element = best_refinement[1]
        if best_score < best_refinement_score:
            best_score = best_refinement_score
            best_solution = candidate_to_refine.copy()
            best_solution.append(best_refinement_element)
            print(f'Updated best solution: {best_solution} with score: {best_score}')

        # prune
        if new_pruning_rule is True:
            # based on monotonicity bound and new pruning rule
            refinements[:] = [ref for ref in refinements if
                              not (ref[2] <= best_score)
                              and not (ref[0] < candidate_score)]
        else:
            # based only on monotonicity bound
            refinements[:] = [ref for ref in refinements if
                              not (ref[2] <= best_score)]
        # how much was pruned
        num_pruned = num_pruned + (len(candidate_refinement_elements) - len(refinements))

        # get the unpruned refinement elements
        unpruned_refinement_elements = [ref[1] for ref in refinements]

        # non-redundantly propagate refinement elements OPUS-style
        # pushes elements into stack in increasing score value, i.e., the top element will be the one with best score
        # propagates most refinement elements to lowest scoring nodes (and not according to lowest opt value)
        for i in range(1, len(unpruned_refinement_elements)):
            unpruned_refinement = candidate_to_refine.copy()
            unpruned_refinement.append(refinements[-i - 1][1])  # append the index of the next lowest scoring node
            refinement_elements_to_propagate = unpruned_refinement_elements[-i:]  # propagate most refinement elements
            new_queue_element = (unpruned_refinement, refinement_elements_to_propagate,
                                 refinements[-i - 1][0])
            queue.append(new_queue_element)

        # for i in range(len(unpruned_refinement_elements) - 1):
        #     unpruned_refinement = candidate_to_refine.copy()
        #     unpruned_refinement.append(refinements[-i - 1][1])  # append the index of the next lowest scoring node
        #     refinement_elements_to_propagate = unpruned_refinement_elements[:-1 - i]  # propagate most refinement elements
        #     new_queue_element = (unpruned_refinement, refinement_elements_to_propagate, refinements[-i - 1][0])
        #     queue.append(new_queue_element)

    best_solution = {x + 1 for x in best_solution}
    best_score = best_score
    print(f'Best solution found: {best_solution} with score {best_score}')
    print(f'Number of pruned: {num_pruned}')
    print(f'Time for completion: {time.time() - start_time}')
    return best_solution, best_score
