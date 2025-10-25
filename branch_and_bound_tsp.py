from typing import Generator

from tsp import Graph
from branch_and_bound import BranchAndBound


class BBTSP(BranchAndBound):

    def is_minimisation_problem(self):
        return True
    
    def separate(self, problem_sub_instance: Graph) -> Generator[Graph, None, None]:
        """ separation mechanism: 
        
        """

        yield Graph()
