import random


type affine = tuple[float, float]


class Knapsack:
    
    def __init__(self, weights: list[int], values: list[float], max_weight: int):
        self.weights = weights
        self.values = values

        assert len(self.values) == len(self.weights)

        self.max_weight = max_weight
    
    def __len__(self) -> int:
        return len(self.weights)
    
    def __str__(self) -> str:
      return f"w {self.weights} v {self.values} MW {self.max_weight}"
    
    @classmethod
    def get_random_knapsack(cls, knapsack_size: int, item_max_weight: int = 100) -> "Knapsack":
        weights = [random.randint(1, item_max_weight) for i in range(knapsack_size)]
        values = [random.randint(1, item_max_weight) for i in range(knapsack_size)]
        max_weight=int(abs(random.random())*item_max_weight*knapsack_size/3)

        return cls(weights=weights, values=values, max_weight=max_weight)

    
    def solve_dynamic_programming(self, full_result: bool = False) -> float|list[list[float]]:

        vs = self.values
        ws = self.weights
        mw = self.max_weight+1
        
        item_nb = len(self)
        value_by_weight_by_subsack: list[list[float]] = [[0 for w in range(mw)] for s in range(item_nb)]

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



def affine_intesection(affine1: affine, affine2: affine) -> float | None:
    if affine1[1] == affine2[1]:
        print(affine1, affine2, "same affine")
        return None
    return (affine2[0] - affine1[0])/(affine1[1] - affine2[1])


def affine_value(affine: affine, value: float) -> float:
    return affine[1]*value+affine[0]


class KnapsackLagrangianRelaxation:

    def __init__(self, knapsack: Knapsack):
        self.knapsack: Knapsack = knapsack
        self.best_penalized_value_by_penalty: list[tuple[int, int]] = []  # sorted [(lambda1 (eg 0), V_pen(lambda1)), ..., (lambda_n (eg +inf), V_pen(lambda_n))]
        self.piece_wise_affine_estimation: list[tuple[tuple[int, int], affine]] = []  # for each interval we store what the best upper bound is

        self.best_feasible_found_value = None
    
    def update_best_solution_value(self, new_value: float) -> None:
        if self.best_feasible_found_value is None or new_value > self.best_feasible_found_value:
            self.best_feasible_found_value = new_value
    
    def update_piece_wise_estimation(self, index:int, new_affine: affine, from_right: bool = True) -> None:

        self.piece_wise_affine_estimation[index] = (self.piece_wise_affine_estimation[index][0], new_affine)
                
        # merge if necessary
        if from_right:
            step = 1
            side = 1
        else:
            step = -1
            side = 0
        if self.piece_wise_affine_estimation[index+step][1] == new_affine:
            self.piece_wise_affine_estimation[index] = (tuple(sorted((self.piece_wise_affine_estimation[index][0][1-side], self.piece_wise_affine_estimation[index+step][0][side]))), new_affine)
            del self.piece_wise_affine_estimation[index+step]
    
    def find_uppper_bound(self, threshold=0.001, floating_accuracy_threshold=0.001) -> tuple[float, bool, float|None]:
        """ 
        returns upper_bound, found_feasible, feasible value
        threshold defines when a change in lambda (penalty factor) is not big enough to continue searching the min of w """

        best_upper_bound = sum(self.knapsack.values)

        affine1 = self.evaluate_full_knapsack()
        affine2 = self.evaluate_empty_knapsack()
        
        new_penalty_candidate: float = affine_intesection(affine1=affine1, affine2=affine2)
        penalty_candidate: float = new_penalty_candidate+threshold+1
        
        self.piece_wise_affine_estimation.append(((0, new_penalty_candidate), affine1))
        self.piece_wise_affine_estimation.append(((new_penalty_candidate, -1), affine2))

        piece_wise_cursor = 0

        lower_bound_w_candidate = affine_value(affine=affine1, value=new_penalty_candidate)  # we suppose this could be the minimum of the dual (best upper bound)
        
        while abs(new_penalty_candidate-penalty_candidate) > threshold:
            penalty_candidate = new_penalty_candidate
            best_solution_for_penalty = self.solve_sub_problem(penalty_candidate)
            # now find the new place where w (penalized value fct) seems minimal (our affine approximation)
            new_affine_lower_bound = self.get_affine_lower_bound(best_solution_for_penalty)
            w_candidate = affine_value(affine=new_affine_lower_bound, value=penalty_candidate)

            if new_affine_lower_bound[1] == 0:
                return w_candidate, True, w_candidate  # the bound is the best feasible 

            if new_affine_lower_bound[1] > 0:
                self.update_best_solution_value(new_affine_lower_bound[0])

            if abs(w_candidate - lower_bound_w_candidate) < floating_accuracy_threshold:  # found the minimum
                # print("real minimum of dual reached")
                return w_candidate, self.best_feasible_found_value is not None, self.best_feasible_found_value

            assert w_candidate > lower_bound_w_candidate, f"The impossible happened: {w_candidate}, {lower_bound_w_candidate}"
            
            if w_candidate < best_upper_bound:
                best_upper_bound = w_candidate

            # we find the left and right intersection (we know that here our affine is on top of the rest)
            # left
            while True:
                interval, affine = self.piece_wise_affine_estimation[piece_wise_cursor]
                if interval[0] == 0 or affine_value(affine, interval[0])>=affine_value(new_affine_lower_bound, interval[0]):
                    left_crossing = affine_intesection(affine1=affine, affine2=new_affine_lower_bound)
                    left_crossing_value = affine_value(affine, left_crossing)
                    left_piece_wise_cursor = piece_wise_cursor

                    left_affine_interval = self.piece_wise_affine_estimation[piece_wise_cursor]
                    self.piece_wise_affine_estimation[piece_wise_cursor] = ((left_affine_interval[0][0], left_crossing), left_affine_interval[1])
                    piece_wise_cursor += 1

                    self.piece_wise_affine_estimation.insert(piece_wise_cursor, ((left_crossing, interval[1]), new_affine_lower_bound))
                    self.update_piece_wise_estimation(index=piece_wise_cursor, new_affine=new_affine_lower_bound)
                    piece_wise_cursor += 1
                    break
                self.update_piece_wise_estimation(index=piece_wise_cursor, new_affine=new_affine_lower_bound)
                piece_wise_cursor -= 1
            while True:
                interval, affine = self.piece_wise_affine_estimation[piece_wise_cursor]
                if interval[1] == -1 or affine_value(affine, interval[1])>=affine_value(new_affine_lower_bound, interval[1]):
                    right_crossing = affine_intesection(affine1=affine, affine2=new_affine_lower_bound)
                    right_crossing_value = affine_value(affine, right_crossing)
                    right_piece_wise_cursor = piece_wise_cursor-1

                    right_affine_interval = self.piece_wise_affine_estimation[piece_wise_cursor]
                    self.piece_wise_affine_estimation[piece_wise_cursor] = ((right_crossing, right_affine_interval[0][1]), right_affine_interval[1])

                    self.piece_wise_affine_estimation.insert(piece_wise_cursor, ((interval[0], right_crossing), new_affine_lower_bound))
                    self.update_piece_wise_estimation(index=piece_wise_cursor, new_affine=new_affine_lower_bound, from_right=False)
                    break
                self.update_piece_wise_estimation(index=piece_wise_cursor, new_affine=new_affine_lower_bound, from_right=False)
                piece_wise_cursor += 1
            
            if left_crossing_value <= right_crossing_value:  # left looks more promising for a min
                piece_wise_cursor = left_piece_wise_cursor
                new_penalty_candidate = left_crossing
            else:
                piece_wise_cursor = right_piece_wise_cursor
                new_penalty_candidate = right_crossing
            lower_bound_w_candidate = affine_value(affine=new_affine_lower_bound, value=new_penalty_candidate)
        
        return best_upper_bound, self.best_feasible_found_value is not None, self.best_feasible_found_value

    
    def evaluate_full_knapsack(self) -> affine:
        return self.get_affine_lower_bound([i for i in range(len(self.knapsack))])
        
    def evaluate_empty_knapsack(self) -> affine:
        return self.get_affine_lower_bound([])
    
    def solve_sub_problem(self, penalty_factor: float) -> list[int]:
        chosen_items = []
        for i in range(len(self.knapsack)):
            if self.knapsack.values[i]>=penalty_factor*self.knapsack.weights[i]:
                chosen_items.append(i)
        return chosen_items
    
    def get_affine_lower_bound(self, chosen_items: list[int]) -> affine:
        return sum([self.knapsack.values[i] for i in chosen_items]), -sum([self.knapsack.weights[i] for i in chosen_items])+self.knapsack.max_weight



def test_knapsack_lagrangian_relaxation():

    def test_knapsack_hr(knapsack: Knapsack):

        value = knapsack.solve_dynamic_programming()
        relaxed_knapsack = KnapsackLagrangianRelaxation(knapsack)
        bound, found_feasible, feasible_value = relaxed_knapsack.find_uppper_bound()
        if found_feasible:
            assert bound >= value >= feasible_value, f"{bound} < {value} or {feasible_value} > {value}, {knapsack}"
            print(f"success, bound :{bound} >= {value} >= {feasible_value}")
        else:
            assert bound >= value, f"{bound} < {value}, {knapsack}"
            print(f"success, bound :{bound} >= {value}")

    test_knapsack_hr(Knapsack(weights=[2 ,1,1], values=[1, 3,2], max_weight=0))
    test_knapsack_hr(Knapsack(weights=[10, 6, 6], values=[11, 6, 6], max_weight=12))
    test_knapsack_hr(Knapsack(weights=[9, 2], values=[5, 2], max_weight=5))

    for _ in range(5):
        test_knapsack_hr(Knapsack.get_random_knapsack(30, 100))

    print("all tests success")


if __name__ == "__main__":
    test_knapsack_lagrangian_relaxation()
