
class Knapsack:
    def __init__(self, weights: list[int], values: list[int], max_weight: int):
        self.weights = weights
        self.values = values

        assert len(self.values) == len(self.weights)

        self.max_weight = max_weight
    
    def __len__(self) -> int:
        return len(self.weights)
    
    def solve_dynamic_programming(self, full_result: bool = False) -> int:

        vs = self.values
        ws = self.weights
        mw = self.max_weight+1
        
        item_nb = len(self)
        value_by_weight_by_subsack = [[0 for w in range(mw)] for s in range(item_nb)]

        value_by_weight_by_subsack[0] = [vs[0]*(ws[0]<=w) for w in range(mw)]
        for i in range(1, item_nb):
            wi = ws[i]
            for w in range(mw):
                if w < wi:
                    value_by_weight_by_subsack[i][w] = value_by_weight_by_subsack[i-1][w]
                else:
                    value_by_weight_by_subsack[i][w] = max(value_by_weight_by_subsack[i-1][w], value_by_weight_by_subsack[i-1][w-wi]+vs[i])
        if full_result:
            return value_by_weight_by_subsack
        else:
            return value_by_weight_by_subsack[-1][-1]



def test_knapsack_solve_dynamic_programming():

    knapsack = Knapsack(weights=[10, 6, 6], values=[11, 6, 6], max_weight=12)
    full_result = knapsack.solve_dynamic_programming(full_result=True)
    assert full_result == [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11], [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 11, 11, 11], [0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 11, 11, 12]], full_result
    print("success")


test_knapsack_solve_dynamic_programming()


class KnapsackHamiltonianRelaxation:

    def __init__(self, knapsack: Knapsack):
        self.knapsack = knapsack

    
    def find_uppper_bound(self) -> int:
        self.evaluate_full_knapsack()
        self.evaluate_empty_knapsack()
    
    def evaluate_full_knapsack(self):
        self.evaluate_knapsack([True for i in range(len(self.knapsack))])
        
    def evaluate_empty_knapsack(self):
        self.evaluate_knapsack([False for i in range(len(self.knapsack))])
    
    def solve_sub_problem(self, penalty_factor: float) -> list[bool]:
        return [self.knapsack.values[i]>=penalty_factor*self.knapsack.weights[i] for i in range(len(self.knapsack))]
    
    def evaluate_knapsack(self, chosen_items: list[bool]) -> None:
        return 