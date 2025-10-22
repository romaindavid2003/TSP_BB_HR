class BBTSP(BranchAndBound):

    def is_minimisation_problem(self):
        return True
    
    def separate(self, problem_sub_instance: ConstrainedGraph) -> Generator[ConstrainedGraph, None, None]:
        """ separation mechanism: 
        
        """

        yield ConstrainedGraph()
