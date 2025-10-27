from abc import ABC, abstractmethod
from typing import Any, Generator
from pydantic import BaseModel


class ProblemInstance(ABC):
    @abstractmethod
    def compute_heuristic(self) -> float:
        pass


class EvaluationResult(BaseModel):
    # stop search encoded by found_feasible=True and bound==feasible_value
    bound: float
    exists_feasible: bool = True
    found_feasible: bool = False
    feasible_value: float|None = None
    next_evaluation_parameters: Any = None


class BranchAndBound(ABC):
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance: ProblemInstance = problem_instance
        self.best_solution_value: float = None  # gets updated in compute_heuristic
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

    def explore_tree(self, problem_sub_instance: ProblemInstance, evaluation_parameters=None) -> float:

        self.visited_nodes += 1
        if self.visited_nodes % self.compute_heuristic_frequency == 0:
            self.compute_heuristic(problem_sub_instance)

        sub_instance_bound, stop_search, next_evaluation_parameters = self.evaluate(problem_sub_instance=problem_sub_instance, evaluation_parameters=evaluation_parameters)
        # no need to explore further, either we have found better already, or we have found the best of this subproblem
        if stop_search or self.is_better(self.best_solution_value, sub_instance_bound):
            return self.best_solution_value  # we wont find better than that

        for new_problem_sub_instance in self.separate(problem_sub_instance):
            self.explore_tree(new_problem_sub_instance, next_evaluation_parameters)

        return self.best_solution_value  # the children will update this

    @abstractmethod
    def compute_evaluation(self, problem_sub_instance: ProblemInstance, evaluation_parameters:Any=None) -> EvaluationResult:
        pass

    @abstractmethod
    def separate(self, problem_sub_instance: ProblemInstance) -> Generator[ProblemInstance, None, None]:
        """can be implemented as an iterable to save space (yield)"""
        pass

    def evaluate(self, problem_sub_instance: ProblemInstance, evaluation_parameters=None) -> tuple[float, bool, Any]:
        """
        uses the evaluate method of the problem_sub_instance class
        which should return:
        -the bound value
        -whether it found a feasible solution
        -the best feasible solution value
        
        Note that the best_solution_value attribute gets updated here.
        
        returns the bound, whether the search should be stopped (if feasible is the bound), and the next evaluation_results
        """
            
        eval_result = self.compute_evaluation(problem_sub_instance, evaluation_parameters)
        if not eval_result.exists_feasible:  # stop search
            return -1, True, eval_result.next_evaluation_parameters
        if eval_result.found_feasible:
            self.update_best_solution_value(eval_result.feasible_value)
            if eval_result.feasible_value == eval_result.bound:  # stop search
                return eval_result.bound, True, eval_result.next_evaluation_parameters
        return eval_result.bound, False, eval_result.next_evaluation_parameters
        
    
    def compute_heuristic(self, problem_sub_instance: ProblemInstance) -> None:
        self.update_best_solution_value(problem_sub_instance.compute_heuristic())
        
    
    def update_best_solution_value(self, new_value: float) -> None:
        if self.is_better(new_value, self.best_solution_value):
            self.best_solution_value = new_value
