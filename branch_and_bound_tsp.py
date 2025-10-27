from typing import Generator
import numpy as np

from tsp import Graph
from branch_and_bound import BranchAndBound, EvaluationResult
from tsp_lagrange_relaxation import TSPLagrangianRelaxation


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
        sepa_info = problem_sub_instance.get_separation_information()
        to_split_vertex, chosen_neighbors = sepa_info.to_split_vertex, sepa_info.chosen_neighbors
        enforced_neighbors = problem_sub_instance.get_enforced_neighbors(to_split_vertex)
        assert 0 <= len(enforced_neighbors) < 2, len(enforced_neighbors)
        assert set(enforced_neighbors) <= set(chosen_neighbors), f"{enforced_neighbors} {chosen_neighbors}"

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

    def compute_evaluation(self, problem_sub_instance: Graph, last_best_penalties) -> EvaluationResult:
        """
        exists_feasible, bound, found_feasible, feasible_value, next_evaluation_parameters
        """
        # first compute a minimum spanning tree with enforced edges
        # this may update the bestfeasible value
        # but mainly check if we enforce a small cycle in the tree
        # if this is the case, we can directly stop this
        feasible_solution_exists, heuristic_value, is_best_value = problem_sub_instance.compute_heuristic_for_constrained_graph()
        if not feasible_solution_exists:
            return EvaluationResult(exists_feasible=False, bound=0)
        
        if is_best_value:
            return EvaluationResult(bound=heuristic_value, found_feasible=True, feasible_value=heuristic_value)

        lr = TSPLagrangianRelaxation(graph=problem_sub_instance, upper_bound=self.best_solution_value, initial_penalties=last_best_penalties)
        bound, found_hamiltonian_cycle, hamiltonian_cycle_value2 = lr.find_lower_bound()

        if found_hamiltonian_cycle:  # stop search
            return EvaluationResult(bound=bound, found_feasible=found_hamiltonian_cycle, feasible_value=bound)

        problem_sub_instance.set_separation_information(lr.separation_information)
        return EvaluationResult(bound=bound, found_feasible=True, feasible_value=heuristic_value, next_evaluation_parameters=lr.best_lower_bound_penalties)


def test_tsp_branch_and_bound():
    def test_tsp_bb(graph: Graph):
        tsp_value = graph.solve_dynamic_programming()
        b_b = BBTSP(graph)
        value = b_b.find_best_value()
        #assert value == tsp_value, f"{tsp_value} (real) is not {value} (found)"
        print("graph size: ", len(graph), "found value: ", value, "real value: ", tsp_value)

    
    test_tsp_bb(Graph.from_points(np.array([[0, 0], [1, 1], [2, 0], [0, 3], [2, 3], [3, 3]])))
    test_tsp_bb(Graph.from_points(np.array([[0, 0], [1, 1], [2, 0], [0, 3], [2, 3]])))
    test_tsp_bb(Graph.from_points(np.array([[0, 0], [1, 1], [2, 0], [0, 1]])))
    test_tsp_bb(Graph.from_points(np.array([[0, 0], [1, 1], [2, 0], [0, 2]])))
    test_tsp_bb(Graph.from_points(np.array([[1, 0], [0, 1], [0, 0], [1, 1]])))
    test_tsp_bb(Graph(vertex_nb=3, weights=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))
    test_tsp_bb(Graph(vertex_nb=4, weights=np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])))

    for _ in range(5):
        test_tsp_bb(Graph.random_triangular_equality_abiding_graph(15, 10))
    return

    print("all tests success")

if __name__ == "__main__":
    test_tsp_branch_and_bound()