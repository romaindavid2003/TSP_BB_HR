# TSP_BB_HR
The project contains an abstract Branch and Bound class, implemented on two problems, knapsack and TSP.
For the knapsack BB, the evaluation methods used is a lagrangian relaxation of the weight constraint, with column generation to (approximately) solve it.
For the TSP BB, the evaluation methods used is an approximate lagrangian relaxation of the 2 neighbors constraint, with gradient ascent and polyakov step to (approximately) solve it.
