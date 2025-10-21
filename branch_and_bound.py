from abc import ABC, abstractmethod

class BranchAndBound(ABC):
    def __init__(self, problem_instance, is_minimisation_problem=True):
        self.is_minimisation_problem: bool = is_minimisation_problem
        self.problem_instance = problem_instance
        self.best_solution_value = self.compute_heuristic(problem_instance)
    
    def better(self, value1, value2) -> bool:
        if self.is_minimisation_problem:
            return value1 <= value2
        return value1 >= value2
        

    def explore_tree(self, problem_sub_instance) -> int:
        found_solution, sub_instance_bound = self.evaluate(problem_sub_instance=problem_sub_instance)
        
        # no need to explore further, we have found better already
        if self.better(self.best_solution_value, sub_instance_bound):
            return self.best_solution_value  # we wont find better than that
        
        # no need to explore further, we have found a best solution of this sub instance
        if found_solution:
            self.best_solution_value = sub_instance_bound
            return sub_instance_bound

        for new_problem_sub_instance in self.separate(problem_sub_instance):
            self.explore_tree(self, new_problem_sub_instance)

        return self.best_solution_value  # the children will update this

    @abstractmethod
    def separate(self):
        """can be implemented as an iterable to save space (yield)"""

    @abstractmethod
    def evaluate(self, problem_sub_instance) -> tuple[bool, float]:
        """
        returns whether it found a real solution as a bound,
        and the bound value. 
        the best_solution_value attribute gets updated automatically
        """
        pass
    
    @abstractmethod
    def compute_heuristic(self, problem_sub_instance) -> float:
        pass
    