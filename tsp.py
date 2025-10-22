class Tree(BaseModel):
    root: int
    children: list["Tree"]


class UnionFind:
    """Code by GPT"""
    def __init__(self, n: int):
        """Initialize Union-Find for elements 0..n-1."""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # optional: track number of disjoint sets

    def find(self, x: int) -> int:
        """Find the root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if merged, False if already in same set."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # already in same set

        # union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1  # optional
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y belong to the same set."""
        return self.find(x) == self.find(y)

    def size(self) -> int:
        """Return the number of disjoint sets."""
        return self.count



class Graph:
    """
    Weighted undirected graph, edge weights respect triangular inequality
    """
    def __init__(self, vertex_nb: int, weights: dict[tuple[int, int], float]):
        self.vertex_nb: int = vertex_nb
        self.weights: dict[tuple[int, int], float] = weights
    
    def __len__(self) -> int:
        return len(self.weights)
    
    def solve_dynamic_programming(self) -> float:
        return 0
    
    def compute_heuristic(self) -> float:
        best_spanning_tree, tree_weight = self.compute_kruskal()
        return tree_weight+0
    
    def compute_kruskal(self) -> tuple[Tree, float]:
        edges = [item for item in self.weights.items()]
        edges.sort(key=lambda item: item[1])  # sort by weight
        component_by_vertex = UnionFind(len(self))
        
        kept_edges = []
        total_weight = 0

        for item in edges:
            vertex1, vertex2 = item[0]
            if not component_by_vertex.connected(vertex1, vertex2):
                component_by_vertex.union(vertex1, vertex2)
                
                kept_edges.append((vertex1, vertex2))
                total_weight += item[1]
        
        # we know have a list of edges that represent a tree in a graph
        # we need to root it to get a tree
        tree = Tree()

        return tree, total_weight
    
