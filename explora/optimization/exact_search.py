import numpy as np
import pandas as pd
import time

from explora.optimization.refinement_and_evaluation import refine_evaluate_get_optimistic_value_sort
from explora.information_theory.information_theory_basic import entropy_plugin


def depth_first_search(data, estimator, target_variable=None, limit=None, new_pruning_rule=True):
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

    if limit is None:
        limit = number_explanatory_variables

    candidate_variables_indices = list(range(number_explanatory_variables + 1))
    candidate_variables_indices.remove(target_variable_index)

    root_node = list()
    root_node_refinements = candidate_variables_indices.copy()
    root_node_score = -float('inf')

    best_solution = root_node
    best_score = root_node_score

    queue = list()
    queue.append((root_node, root_node_refinements, root_node_score))

    num_pruned = 0
    num_pruned_rule = 0

    while queue:
        print(f'Size of queue: {len(queue)}')
        top_queue_element = queue.pop()
        candidate_to_refine = top_queue_element[0] #a list of attribute indices
        refinement_elements = top_queue_element[1]  #a list of attribute indices
        candidate_score = top_queue_element[2]

        # create refinements, evaluate them, get their optimistic values
        scores_and_refs_and_opt_values = refine_evaluate_get_optimistic_value_sort(estimator, data, target_data_column,
                                                                                   candidate_to_refine,
                                                                                   refinement_elements)

        # get best
        best_refinement = scores_and_refs_and_opt_values[0]
        best_refinement_score = best_refinement[0]
        best_refinement_index = best_refinement[1]
        if best_score < best_refinement_score:
            best_score = best_refinement_score
            best_solution = candidate_to_refine.copy()
            best_solution.append(best_refinement_index)
            print(f'Updated best solution: {best_solution} with score: {best_score / entropy_target}')

        # prune
        unpruned_refinements = list()

        # prune based on monotonicity bound
        if new_pruning_rule is True:
            [unpruned_refinements.append(ref) for ref in scores_and_refs_and_opt_values
             if not (ref[2] <= best_score) and not (ref[0] < candidate_score)]
        else:
            [unpruned_refinements.append(ref) for ref in scores_and_refs_and_opt_values if ref[2] > best_score]

        num_pruned = num_pruned + (len(refinement_elements)-len(unpruned_refinements))

        # get the indices of the unpruned refinements
        unpruned_refinement_elements = [refinement[1] for refinement in unpruned_refinements]

        # non-redundantly propagate refinement elements OPUS-style
        for i in range(len(unpruned_refinements) - 1):
            unpruned_refinement = candidate_to_refine.copy()
            unpruned_refinement.append(unpruned_refinements[-i - 1][1])
            refinement_elements = unpruned_refinement_elements[:-1 - i]
            new_queue_element = (unpruned_refinement, refinement_elements, unpruned_refinements[-i - 1][0])
            queue.append(new_queue_element)

        # for i in range(len(unpruned_candidate_indices)-1):
        #     unpruned_refinement=candidate.copy()
        #     unpruned_refinement.append(unpruned_candidate_indices[i])
        #     refinement_elements=unpruned_candidate_indices[i+1:]
        #     new_queue_element=(unpruned_refinement,refinement_elements)
        #     queue.append(new_queue_element)

    best_solution= {x + 1 for x in best_solution}
    best_score = best_score / entropy_target
    print(f'Best solution found: {best_solution} with score {best_score}')
    print(f'Number of pruned: {num_pruned}')
    print(f'Time for completion: {time.time() - start_time}')
    return best_solution, best_score

