from functools import lru_cache



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


def get_tree_hc_length(tree: Tree, weights: list[list[float]]) -> float:
    def get_tree_hc_length_last_visited_node(tree: Tree) -> tuple[float, int]:
        length = 0
        last_node = tree.root
        for child in tree.children:
            length += weights[last_node][child.root]
            child_length, last_node = get_tree_hc_length_last_visited_node(child)
            length += child_length
        return length, last_node  # also deals with trivial leaf tree case
    
    length, last_visited_node = get_tree_hc_length_last_visited_node(tree)
    return length+weights[tree.root][last_visited_node]


class Graph:
    """
    Weighted undirected graph, edge weights respect triangular inequality
    """
    def __init__(self, vertex_nb: int, weights: list[list[float]]):
        self.vertex_nb: int = vertex_nb
        self.weights: list[list[float]] = weights  # weight matrix
    
    def __len__(self) -> int:
        return len(self.weights)
    
    def solve_dynamic_programming(self) -> float:

        vertices_no0 = set(range(len(1, self)))

        @lru_cache(maxsize=None)  # memoization here
        def minimal_chain(T:set[int], target_vertex: int) -> float:
            """
            returns minimal length of a path leading from vertex 0 to vertex target_vertex going once through every vertex of T
            """
            if len(T) == 0:  # we always start from 0 
                return self.weights[0][target_vertex]
            else:
                return min(minimal_chain(T-set(k), k)+self.weights[k][target_vertex] for k in T)

        return min(minimal_chain(vertices_no0-set(k))+self.weights[0][k] for k in vertices_no0)
    
    def compute_heuristic(self) -> float:
        best_spanning_tree = self.compute_kruskal()
        # now find the real length of this hamiltonian cycle by skipping the doubled edges
        return get_tree_hc_length(best_spanning_tree)
    
    def compute_kruskal(self) -> Tree:
        edges = [item for item in self.weights.items()]
        edges.sort(key=lambda item: item[1])  # sort by weight
        component_by_vertex = UnionFind(len(self))
        
        kept_edges = []

        neighbors_in_tree: dict[int, list[int]] = {i:[] for i in range(len(self))}

        for item in edges:
            vertex1, vertex2 = item[0]
            if not component_by_vertex.connected(vertex1, vertex2):
                component_by_vertex.union(vertex1, vertex2)
                
                kept_edges.append((vertex1, vertex2))

                neighbors_in_tree[vertex1].append(vertex2)
                neighbors_in_tree[vertex2].append(vertex1)
        
        # we know have a list of edges that represent a tree in a graph
        # we need to root it to get a tree
        visited_vertices = set([1])
        def construct_tree_from_neighbors(root: int) -> Tree:
            neighbors = []
            for neighbor in neighbors_in_tree[root]:
                if not neighbor in visited_vertices:
                    neighbors.append(neighbor)
                    visited_vertices.add(neighbor)

            return Tree(root=root, children=[construct_tree_from_neighbors(neighbor) for neighbor in neighbors])
        tree = construct_tree_from_neighbors(1)

        return tree
        
    def compute_kruskal_weight(self) -> float:
        edges = [item for item in self.weights.items()]
        edges.sort(key=lambda item: item[1])  # sort by weight
        component_by_vertex = UnionFind(len(self))
        
        total_weight = 0

        neighbors_in_tree: dict[int, list[int]] = {i:[] for i in range(len(self))}

        for item in edges:
            vertex1, vertex2 = item[0]
            if not component_by_vertex.connected(vertex1, vertex2):
                component_by_vertex.union(vertex1, vertex2)
                
                total_weight += item[1]

                neighbors_in_tree[vertex1].append(vertex2)
                neighbors_in_tree[vertex2].append(vertex1)

        return total_weight
