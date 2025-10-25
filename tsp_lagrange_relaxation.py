import numpy as np
from tsp import Graph


class TSPlagrangianRelaxation:

    def __init__(self, graph: Graph, upper_bound: float, initial_penalties: np.array | None=None):
        self.graph = graph
        self.best_lower_bound: float | None = None
        self.upper_bound: float = upper_bound
        
        self.penalties: np.array
        if initial_penalties is None:
            self.penalties = np.zeros(len(graph))
        else:
            self.penalties = initial_penalties
    
    def find_best_lower_bound(self, accuracy_threshold: float=0.00001, max_iteration: int=1000) -> tuple[float, bool, float|None]:
        """
        returns lower_bound, found_feasible, feasible value
        """

        for i in range(max_iteration):

            if 0 < accuracy_threshold:
                break
            
            penalized_graph = self.compute_penalized_graph()

            best_one_tree = penalized_graph.compute_best_one_tree()
            lower_bound, subgradient = self.compute_result(best_one_tree)

            if np.sum(np.abs(subgradient)) < accuracy_threshold:  # best was a hamiltonian cycle
                return lower_bound, True, lower_bound  # the bound is actually the feasible value

            if self.best_lower_bound < lower_bound:
                self.best_lower_bound = lower_bound

            self.update_penalties(subgradient, lower_bound)
        
        # bcs we relax equalities, we never find a feasible solution 
        # (except when the feasible solution is the solution to the relaxation)
        return self.best_lower_bound, False, 0

    def compute_penalized_graph(self) -> Graph:
        penalized_graph = self.graph.copy()

        for i in range(len(penalized_graph)):
            penalized_graph.weights[i] += self.penalties[i]
            penalized_graph.weights[:, i] += self.penalties[i]  # keep weight ;arix symmetricity
        
        return penalized_graph
    
    def compute_result(self, tree) -> tuple[float, np.array]:

        node_nb_per_vertex = tree

        node_nb_per_vertex = np.array(node_nb_per_vertex)
        subgradient = node_nb_per_vertex-2
        lower_bound = get_tree_hc_length(tree)+2*np.sum(self.penalties)
        return lower_bound, subgradient
    
    def update_penalties(self, subgradient, lower_bound) -> None:
        step_size = self.get_step_size(lower_bound)
        self.penalties += step_size*subgradient
    
    def get_step_size(self, subgradient, lower_bound) -> float:
        return 0.5*(self.upper_bound-lower_bound)/np.linalg.norm(subgradient)