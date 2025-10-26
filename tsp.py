from functools import lru_cache
from pydantic import BaseModel
import random
import math
import matplotlib.pyplot as plt
import numpy as np


class Tree(BaseModel):
    root: int
    children: list["Tree"]

    def __str__(self) -> str:
      return f"{self.root}: {[str(c) for c in self.children]}"


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
    def __init__(self, vertex_nb: int, weights: np.array, enforced_edges:np.array|None=None, banned_edges:np.array|None=None):
        self.vertex_nb: int = vertex_nb
        self.weights: np.array = weights  # weight matrix (symmetric)

        self.enforced_edges: np.array
        if enforced_edges == None:
            self.enforced_edges = np.zeros((vertex_nb, vertex_nb))
        else:
            self.enforced_edges = enforced_edges

        self.banned_edges: np.array
        if banned_edges == None:  # we don t allow edges to themselves, and symmetric edges (undirected graph)
            self.banned_edges = np.ones((vertex_nb, vertex_nb))
            for k in range(vertex_nb-1):
                self.banned_edges[k][k+1:] = np.zeros(vertex_nb-k-1)
        else:
            self.banned_edges = banned_edges

    def copy(self, graph: "Graph") -> "Graph":
        vertex_nb = graph.vertex_nb
        weights = graph.weights
        enforced_edges = graph.enforced_edges
        banned_edges = graph.banned_edges

        return Graph(vertex_nb=vertex_nb, weights=weights, banned_edges=banned_edges, enforced_edges=enforced_edges)
    
    def random_triangular_equality_abiding_graph(size: int, graph_amplitude: int=10) -> "Graph":
        points = graph_amplitude*np.random.random((size, 2))
        differences = points[:, None, :]-points[None:, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        return Graph(size, distances)

    def get_edges(self) -> list[tuple[int, int, float]]:
        allowed_edges_mat = self.banned_edges==0
        allowed_edges=np.where(allowed_edges_mat)
        return np.stack((allowed_edges[0], allowed_edges[1], self.weights[allowed_edges_mat].flatten()), axis=-1)

    def get_enforced_edges_and_other_edges(self, computing_one_tree:bool=False) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
        """ if computing_one_tree, then we don't care about all the edges linked to node 0 """
        enforced_edges_mat = self.enforced_edges==1
        if computing_one_tree:
            enforced_edges_mat[0] = np.zeros(len(self))
        enforced_edges=np.where(enforced_edges_mat)
        
        other_edges_mat = self.banned_edges==0*self.enforced_edges==0
        if computing_one_tree:
            other_edges_mat[0] = np.zeros(len(self))
        allowed_edges=np.where(other_edges_mat)
        return np.stack((enforced_edges[0], enforced_edges[1], self.weights[enforced_edges_mat].flatten()), axis=-1), np.stack((allowed_edges[0], allowed_edges[1], self.weights[other_edges_mat].flatten()), axis=-1)

    def get_enforced_neighbors(self, vertex: int) -> list[int]:
        return np.where(self.enforced_edges[vertex]==1)
    
    def enforce(self, vertex1: int, vertex2: int) -> None:
        self.enforced_edges[vertex1][vertex2] = 1
        self.enforced_edges[vertex2][vertex1] = 1

        enforced_nb1 = np.sum(self.enforced_edges[vertex1])
        assert 0 <= enforced_nb1 <= 2

        if enforced_nb1 == 2:
            self.banned_edges[vertex1][self.enforced_edges[vertex1]==0] = 1
        
        enforced_nb2 = np.sum(self.enforced_edges[vertex2])
        assert 0 <= enforced_nb2 <= 2

        if enforced_nb2 == 2:
            self.banned_edges[vertex2][self.enforced_edges[vertex2]==0] = 1
    
    def ban(self, vertex1: int, vertex2: int) -> None:
        self.banned_edges[vertex1][vertex2] = 1
        self.banned_edges[vertex2][vertex1] = 1

    def __len__(self) -> int:
        return len(self.weights)
    
    def solve_dynamic_programming(self) -> float:

        vertices_no0 = frozenset(range(1, len(self)))

        @lru_cache(maxsize=None)  # memoization here
        def minimal_chain(T:frozenset[int], target_vertex: int) -> float:
            """
            returns minimal length of a path leading from vertex 0 to vertex target_vertex going once through every vertex of T
            """
            if len(T) == 0:  # we always start from 0 
                return self.weights[0][target_vertex]
            else:
                return min(minimal_chain(frozenset(T-set([k])), k)+self.weights[k][target_vertex] for k in T)

        return min(minimal_chain(frozenset(vertices_no0-set([k])), k)+self.weights[0][k] for k in vertices_no0)
        
    def compute_heuristic(self) -> float:
        """ returns 2 guarantee heuristic """
        best_spanning_tree = self.compute_kruskal_enforced_edges()
        assert isinstance(best_spanning_tree, Tree)
        # now find the real length of this lagrangian cycle by skipping the doubled edges
        return get_tree_hc_length(best_spanning_tree, self.weights)

    def compute_heuristic_for_constrained_graph(self) -> tuple[bool, float, bool]:
        """ returns if a feasible solution exists, if so the heuristic value, and if it is the best value """
        best_spanning_tree = self.compute_kruskal_enforced_edges()
        if isinstance(best_spanning_tree, bool): # a cycle is enforced
            if best_spanning_tree:
                return True, self.weights*self.banned_edges*self.enforced_edges, True
            return False, 0, False

        # now find the real length of this lagrangian cycle by skipping the doubled edges
        return True, get_tree_hc_length(best_spanning_tree, self.weights), False
    
    def compute_kruskal_enforced_edges(self, computing_one_tree:bool=False) -> Tree | bool | tuple[float, dict[int, list[int]]]:
        """ 
        if a cycle is enforced, returns whether its a full lagrangian cycle;
         otherwise return depending on the context:
            the spaning tree
            or the weight of the spanning tree and the neighbors in this tree 
        """
        enforced_edges, edges = self.get_enforced_edges_and_other_edges(computing_one_tree)
        component_by_vertex = UnionFind(len(self))
        
        kept_edges = []
        neighbors_in_tree: dict[int, list[int]] = {i:[] for i in range(len(self))}
        spanning_tree_weight: float = 0

        for item in enforced_edges:
            vertex1, vertex2, w = item
            if component_by_vertex.connected(vertex1, vertex2):  # there exists a subcycle
                return False
            else:
                component_by_vertex.union(vertex1, vertex2)
                kept_edges.append((vertex1, vertex2))
                spanning_tree_weight += self.weights[vertex1][vertex2]

                neighbors_in_tree[vertex1].append(vertex2)
                neighbors_in_tree[vertex2].append(vertex1)
        
        edges.sort(key=lambda item: item[2])  # sort by weight

        for item in edges:
            vertex1, vertex2 = item[:2]
            if not component_by_vertex.connected(vertex1, vertex2):
                component_by_vertex.union(vertex1, vertex2)
                
                kept_edges.append((vertex1, vertex2))
                spanning_tree_weight += self.weights[vertex1][vertex2]

                neighbors_in_tree[vertex1].append(vertex2)
                neighbors_in_tree[vertex2].append(vertex1)

                if len(kept_edges) == len(self):  # the spanning tree is complete
                    break
                elif computing_one_tree and len(kept_edges) == len(self)-1:  # the spanning one tree is complete
                    break
        
        if computing_one_tree:
            return spanning_tree_weight, neighbors_in_tree
        
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

    def compute_best_one_tree(self) -> tuple[float, list[int]]:
        """
        a best one tree is composed of the two smallest edges around some vertex and of the minimum spanning tree on the rest
        """

        # 
        enforced_edges_for_first_vertex = np.where(self.enforced_edges[0] == 1)
        enforced_edges_for_first_vertex_nb = np.sum(enforced_edges_for_first_vertex)
        enforced_edges_for_first_vertex_weight = np.sum(self.weights[0][enforced_edges_for_first_vertex])
        other_edges_weights_for_first_vertex = self.weights[np.where(self.enforced_edges[0] == 0, self.banned_edges == 0)]
        other_edge_needed_nb = 2-enforced_edges_for_first_vertex_nb
        if other_edge_needed_nb == 0:
            first_vertex_weight = enforced_edges_for_first_vertex_weight
        else:
            first_vertex_weight = np.sum(np.partition(other_edges_weights_for_first_vertex, other_edge_needed_nb)[:other_edge_needed_nb])+enforced_edges_for_first_vertex_weight
        
        spanning_tree_weight, neighbors = self.compute_kruskal_enforced_edges(computing_one_tree=True)
        neighbor_nb = [len(neighbors[i]) for i in range(len(self))]
        neighbor_nb[0] = 2  # bcs it's a one tree
        
        return first_vertex_weight+spanning_tree_weight, neighbor_nb



def test_basic_graph_functions():
    def test_graph(graph: Graph):
        feasible_value = graph.compute_heuristic()
        value = graph.solve_dynamic_programming()
        assert abs(feasible_value - value) >= -0.0001, f"{feasible_value} < {value}"
        print(round(feasible_value/value, 3))


    test_graph(Graph(vertex_nb=3, weights=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    for _ in range(10):
      test_graph(Graph.random_triangular_equality_abiding_graph(15, 10))
    print("All tests successful")

test_basic_graph_functions()
    