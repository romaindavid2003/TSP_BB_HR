import numpy as np
from tsp import Graph


class TSPlagrangianRelaxation:

    def __init__(self, graph: Graph, upper_bound: float, initial_penalties: np.array | None=None):
        self.graph = graph
        self.best_lower_bound: float | None = None
        self.upper_bound: float = upper_bound
        
        self.penalties: np.array
        self.old_penalties: np.array = np.zeros(len(graph))
        if initial_penalties is None:
            self.penalties = np.zeros(len(graph))
        else:
            self.penalties = initial_penalties
    
    def find_best_lower_bound(self, accuracy_threshold: float=0.00001, max_iteration: int=1000) -> tuple[float, bool, float|None]:
        """
        returns lower_bound, found_feasible, feasible value
        """

        for i in range(max_iteration):
            
            # too few changes
            if i > 2 and (np.sum(np.abs(self.penalties-self.old_penalties)) < accuracy_threshold):
                break
            
            penalized_graph = self.compute_penalized_graph()

            weight_one_tree, node_nb_per_vertex = penalized_graph.compute_best_one_tree()

            subgradient = np.array(node_nb_per_vertex)-2
            lower_bound = weight_one_tree+2*np.sum(self.penalties)

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
            penalized_graph.weights[:, i] += self.penalties[i]  # keep weight matrix symmetricity
        
        return penalized_graph
    
    def update_penalties(self, subgradient, lower_bound) -> None:
        step_size = self.get_step_size(lower_bound)
        self.old_penalties = self.penalties
        self.penalties += step_size*subgradient
    
    def get_step_size(self, subgradient, lower_bound) -> float:
        return 0.5*(self.upper_bound-lower_bound)/np.linalg.norm(subgradient)