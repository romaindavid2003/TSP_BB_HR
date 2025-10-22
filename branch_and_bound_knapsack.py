from typing import Generator


class ValuedKnapsack(Knapsack):
    def __init__(self, weights: list[int], values: list[int], max_weight: int, added_value:int = 0, sorted_by_value_weight_ratio=False):
        super().__init__(weights=weights, values=values, max_weight=max_weight)
        self.added_value = added_value
        self.sorted_by_value_weight_ratio: bool = sorted_by_value_weight_ratio  # we need to do this only one time
    
    def from_knapsack(knapsack: Knapsack) -> "ValuedKnapsack":
      return ValuedKnapsack(knapsack.weights, knapsack.values, knapsack.max_weight)
    
    def evaluate(self) -> tuple[float, bool, float|None]:
        if len(self) == 0:
          return 0, True, 0
        hr = KnapsackHamiltonianRelaxation(self)
        evaluation_result = hr.find_uppper_bound()
        return (self.added_value+evaluation_result[0], evaluation_result[1], self.added_value+(evaluation_result[2] or 0))

    def compute_heuristic(self) -> float:
        if not self.sorted_by_value_weight_ratio:
            self.sorted_by_value_weight_ratio = True
            vs, ws = self.values, self.weights
            n = len(vs)
            value_weight_ratios = [vs[i]/ws[i] for i in range(n)]
            order = sorted([i for i in range(n)], key=lambda i:value_weight_ratios[i], reverse=True)
            self.weights = [ws[i] for i in order]
            self.values = [vs[i] for i in order]

        value = 0
        weight_left = self.max_weight
        for i in range(len(self.values)):
            v, w = self.values[i], self.weights[i]
            if weight_left >= w:
                weight_left -= w
                value += v
        return value


class BBKnapsack(BranchAndBound):
    def is_minimisation_problem(self):
        return False
    
    def separate(self, problem_sub_instance: ValuedKnapsack) -> Generator[ValuedKnapsack, None, None]:
        """ separation mechanism: first object either gets choosed or thrown away"""
        w1, v1 = problem_sub_instance.weights[0], problem_sub_instance.values[0]
        rest_w = problem_sub_instance.weights[1:]
        rest_v = problem_sub_instance.values[1:]
        wmax = problem_sub_instance.max_weight
        yield ValuedKnapsack(weights=rest_w, values=rest_v, max_weight=wmax, added_value=problem_sub_instance.added_value, sorted_by_value_weight_ratio=problem_sub_instance.sorted_by_value_weight_ratio)
        yield ValuedKnapsack(weights=rest_w, values=rest_v, max_weight=wmax-w1, added_value=problem_sub_instance.added_value+v1, sorted_by_value_weight_ratio=problem_sub_instance.sorted_by_value_weight_ratio)
def test_knapsack_branch_and_bound():
    def test_knapsack_bb(knapsack: Knapsack):
        knapsack_value = knapsack.solve_dynamic_programming()
        knapsack_instance = ValuedKnapsack.from_knapsack(knapsack)
        b_b = BBKnapsack(knapsack_instance)
        value = b_b.find_best_value()
        assert value == knapsack_value, f"{knapsack_value} is not {value}"
        print("instance size: ", 2**len(knapsack), "found value: ", value)

    test_knapsack_bb(Knapsack(weights=[10, 6, 6], values=[11, 6, 6], max_weight=12))
    test_knapsack_bb(Knapsack.get_random_knapsack(10, 100))
    test_knapsack_bb(Knapsack.get_random_knapsack(15, 100))
    test_knapsack_bb(Knapsack.get_random_knapsack(20, 100))
    test_knapsack_bb(Knapsack.get_random_knapsack(30, 100))
    for _ in range(0):
        test_knapsack_bb(Knapsack.get_random_knapsack(30, 100))
    print("all tests success")

test_knapsack_branch_and_bound()