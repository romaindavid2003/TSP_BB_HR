from typing import Generator

from tsp import Graph
from branch_and_bound import BranchAndBound
from tsp_lagrange_relaxation import TSPlagrangianRelaxation


class BBTSP(BranchAndBound):

    def __init__(self, problem_instance):
        super().__init__(problem_instance)

    def is_minimisation_problem(self):
        return True
    
    def separate(self, problem_sub_instance: Graph) -> Generator[Graph, None, None]:
        """ separation mechanism: 
        [depends on the evaluation]
        we choose a vertex with 3 edges (e1, e2, e3) in the evaluation and check his enforced edges:
        - 0:
            separate on:
                - enforce e1, e2 (ban all othes)
                - enforce e1 ban e2
                - ban e1
        -1 (suppose e3 is enforced):
        separate on:
                - enforce e1 (ban all othes)
                - ban e1
        -2 impossible

        Relies on the fact that some information has been stored during evaluation
        """
        to_split_vertex, chosen_neighbors = problem_sub_instance.get_evaluation_information
        enforced_neighbors = problem_sub_instance.get_enforced_neighbors(to_split_vertex)

        assert 0 <= len(enforced_neighbors) < 2
        assert enforced_neighbors in chosen_neighbors

        other_neighbors = set(chosen_neighbors)-set(enforced_neighbors)

        if len(enforced_neighbors) == 1:
            graph = problem_sub_instance.copy()
            neighbor1 = other_neighbors.pop()
            graph.enforce(to_split_vertex, neighbor1)
            yield graph
            graph = problem_sub_instance.copy()
            graph.ban(to_split_vertex, neighbor1)
            yield graph

        elif len(enforced_neighbors) == 0:
            graph = problem_sub_instance.copy()
            neighbor1 = other_neighbors.pop()
            neighbor2 = other_neighbors.pop()
            graph.enforce(to_split_vertex, neighbor1)
            graph.enforce(to_split_vertex, neighbor2)
            yield graph
            graph = problem_sub_instance.copy()
            graph.enforce(to_split_vertex, neighbor1)
            graph.ban(to_split_vertex, neighbor2)
            yield graph
            graph = problem_sub_instance.copy()
            graph.ban(to_split_vertex, neighbor1)
            yield graph


    def evaluate(self, problem_sub_instance: Graph, last_best_penalties) -> tuple[bool, float, bool, float|None]:
        """
        exists_feasible, bound, found_feasible, feasible_value
        """
        # first compute a minimum spanning tree with enforced edges
        # this may update the bestfeasible value
        # but mainly check if we enforce a small cycle in the tree
        # if this is the case, we can directly stop this
        feasible_solution_exists, heuristic_value, is_best_value = problem_sub_instance.compute_heuristic_for_constrained_graph()

        if not feasible_solution_exists:
            return False, 0, False, 0
        
        if is_best_value:
            return True, heuristic_value, True, heuristic_value

        lr = TSPlagrangianRelaxation(graph=self, upper_bound=self.best_solution_value, initial_penalties=last_best_penalties)
        bound, found_heuristic, heuristic_value2 = lr.find_uppper_bound()
        
        if found_heuristic:
            if heuristic_value2 > heuristic_value:
                heuristic_value = heuristic_value2
        return (True, bound, True, heuristic_value)
