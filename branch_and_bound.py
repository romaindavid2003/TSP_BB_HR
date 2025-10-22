from abc import ABC, abstractmethod


class ProblemInstance(ABC):
    @abstractmethod
    def compute_heuristic(self) -> float:
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass


class BranchAndBound(ABC):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance: ProblemInstance = problem_instance
        self.best_solution_value = None  # gets updated in compute_heuristic
        self.compute_heuristic(problem_instance)

        self.visited_nodes: int = 0
        self.compute_heuristic_frequency: int = 100
        
    @abstractmethod
    def is_minimisation_problem(self) -> bool:
        pass

    def is_better(self, value1: float|None, value2: float|None) -> bool:
        if value1 is None:
            return False
        if value2 is None:
            return True
        if self.is_minimisation_problem():
            return value1 <= value2
        return value1 >= value2
    
    def find_best_value(self) -> float:
        value = self.explore_tree(self.problem_instance)
        print(f"visited nodes: {self.visited_nodes}")
        return value

    def explore_tree(self, problem_sub_instance: ProblemInstance) -> float:

        self.visited_nodes += 1
        if self.visited_nodes % self.compute_heuristic_frequency == 0:
            self.compute_heuristic(problem_sub_instance)

        sub_instance_bound, found_best_solution = self.evaluate(problem_sub_instance=problem_sub_instance)

        # no need to explore further, either we have found better already, or we have found the best of this subproblem
        if found_best_solution or self.is_better(self.best_solution_value, sub_instance_bound):
            return self.best_solution_value  # we wont find better than that

        for new_problem_sub_instance in self.separate(problem_sub_instance):
            self.explore_tree(new_problem_sub_instance)

        return self.best_solution_value  # the children will update this

    @abstractmethod
    def separate(self, problem_sub_instance: ProblemInstance):
        """can be implemented as an iterable to save space (yield)"""
        pass

    def evaluate(self, problem_sub_instance: ProblemInstance) -> tuple[float, bool]:
        """
        uses the evaluate method of the problem_sub_instance class
        this method should return:
        -the bound value
        -whether it found a feasible solution
        -the best feasible solution value
        
        Note that the best_solution_value attribute gets updated here.
        returns the bound, and whether the search should be stopped (if feasible is the bound)
        """
        bound, found_feasible, feasible_value = problem_sub_instance.evaluate()
        if found_feasible:
            self.update_best_solution_value(feasible_value)
            if feasible_value == bound:
                return bound, True
        return bound, False
        
    
    def compute_heuristic(self, problem_sub_instance: ProblemInstance) -> float:
        self.update_best_solution_value(problem_sub_instance.compute_heuristic())
        
    
    def update_best_solution_value(self, new_value: float) -> None:
        if self.is_better(new_value, self.best_solution_value):
            self.best_solution_value = new_value