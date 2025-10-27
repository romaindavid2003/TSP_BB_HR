import numpy as np
from tsp import Graph, SeparationInfo

class TSPLagrangianRelaxation:

    def __init__(self, graph: Graph, upper_bound: float, initial_penalties: np.ndarray | None=None):
        self.graph = graph
        self.best_lower_bound: float | None = None

        self.upper_bound: float = upper_bound
        
        self.penalties: np.array
        self.old_penalties: np.array = np.zeros(len(graph))
        if initial_penalties is None:
            self.penalties = np.zeros(len(graph)).astype(np.float16)
        else:
            self.penalties = initial_penalties
            
        self.best_lower_bound_penalties = initial_penalties
        self.separation_information: SeparationInfo|None = None
    
    def find_lower_bound(self, accuracy_threshold: float=0.00001, max_iteration: int=1000) -> tuple[float, bool, float|None]:
        """
        returns lower_bound, found_hamiltonian_cycle, feasible value
        """

        for i in range(max_iteration):
            
            # too few changes
            if i > 2 and (np.sum(np.abs(self.penalties-self.old_penalties)) < accuracy_threshold):
                break
            
            penalized_graph = self.compute_penalized_graph()

            weight_one_tree, node_nb_per_vertex, separation_info = penalized_graph.compute_best_one_tree()

            subgradient = np.array(node_nb_per_vertex)-2
            lower_bound = weight_one_tree+2*np.sum(self.penalties)

            if np.sum(np.abs(subgradient)) < accuracy_threshold:  # best was a hamiltonian cycle
                return lower_bound, True, lower_bound  # the bound is actually the feasible value
            
            else:
              assert separation_info is not None, f"sepa info is not None if one tree is not HC {subgradient} {self.graph}"

            self.update_best_lower_bound(lower_bound, separation_info)

            self.update_penalties(subgradient, lower_bound)
        
        # bcs we relax equalities, we never find a feasible solution 
        # (except when the feasible solution is the solution to the relaxation)
        return self.best_lower_bound, False, 0

    def compute_penalized_graph(self) -> Graph:
        penalized_graph = self.graph.copy()

        for i in range(len(penalized_graph)):
            penalized_graph.weights[i] += self.penalties[i]
            penalized_graph.weights[:, i] += self.penalties[i]  # keep weight matrix symmetricity

        return penalized_graph
    
    def update_penalties(self, subgradient, lower_bound) -> None:
        step_size = self.get_step_size(subgradient, lower_bound)
        self.old_penalties = self.penalties#.copy()
        self.penalties += step_size*subgradient
    
    def get_step_size(self, subgradient, lower_bound) -> float:
        return 0.5*(self.upper_bound-lower_bound)/np.linalg.norm(subgradient)
    
    def update_best_lower_bound(self, lower_bound: float, separation_info: SeparationInfo) -> None:
        if self.best_lower_bound is None or self.best_lower_bound < lower_bound:
            self.best_lower_bound = lower_bound
            self.best_lower_bound_penalties = self.penalties
            self.separation_information = separation_info


def test_tsp_lagrangian_relaxation():

    def test_tsp_hr(graph: Graph):

        value = graph.solve_dynamic_programming()
        heuristc_value = graph.compute_heuristic()
        relaxed_tsp = TSPLagrangianRelaxation(graph=graph, upper_bound=heuristc_value)
        bound, found_feasible, feasible_value = relaxed_tsp.find_lower_bound()
        if found_feasible:
            assert bound <= value <= feasible_value, f"{bound} > {value} or {feasible_value} < {value}, {graph}"
            assert value <= heuristc_value, f"{heuristc_value} < {value}, {graph}"
            print(f"success, bound :{bound} <= {value} <= {feasible_value}")
        else:
            assert bound <= value <= heuristc_value, f"{bound} > {value} or {heuristc_value} < {value}, {graph}"
            print(f"success, bound :{bound} <= {value}")

    test_tsp_hr(Graph(vertex_nb=3, weights=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))
    test_tsp_hr(Graph(vertex_nb=4, weights=np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])))

    for _ in range(5):
        test_tsp_hr(Graph.random_triangular_equality_abiding_graph(15, 10))

    print("all tests success")


if __name__ == "__main__":
    test_tsp_lagrangian_relaxation()