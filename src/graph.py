import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm


class Graph(object):
    def __init__(self):
        self.edges = dict()
        self.nodes = dict()

    def adjacency_matrix(self, weight=None):
        N = len(self.nodes)
        A = np.zeros((N, N))

        for (i, j), edge_data in self.edges.items():
            A[i - 1, j - 1] = (
                1 if weight == None else edge_data.edge_dict["weight"]
            )
        return A

    def degree_matrix(self):
        return np.diag([degree for node, degree in self.degree.items()])

    def laplacian_matrix(self):
        D = self.degree_matrix()
        W = self.adjacency_matrix()
        L = D - W
        return L

    def normalized_laplacian_matrix(self):
        D = self.degree_matrix()
        D_neg_2 = sqrtm(inv(D))
        L = self.laplacian_matrix()
        L_sym = D_neg_2 @ L @ D_neg_2
        return L_sym

    @property
    def degree(self):
        return {
            node: node_data.degree() for (node, node_data) in self.nodes.items()
        }

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return len(self.edges)

    def add_node(self, node=None):
        if node == None:
            node = max(self.nodes.keys()) + 1
        self.nodes[node] = Node(self, node)

    def add_edge(self, out_node, in_node, *edge_dict):
        edge = (out_node, in_node)
        reverse_edge = (in_node, out_node)

        if not in_node in self.nodes or not out_node in self.nodes:
            missing_nodes = [node for node in edge if not (node in self.nodes)]
            raise ValueError(
                f"Edge {edge} can't be added. The graph does not contain the following nodes: {missing_nodes}"
            )

        self.edges[edge] = (
            Edge(self, edge, edge_dict[0]) if edge_dict else Edge(self, edge)
        )
        self.edges[reverse_edge] = (
            Edge(self, reverse_edge, edge_dict[0])
            if edge_dict
            else Edge(self, reverse_edge)
        )

    def add_nodes_from(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_edges_from(self, edges):
        for edge in edges:
            self.add_edge(*edge)

    def remove_edge_permutations(self):
        for edge in self.edges:
            double_edge = (edge[1], edge[0])
            if double_edge in self.edges:
                self.edges.pop(double_edge)
                return self.remove_edge_permutations()

    def connect_nodes(self, out_nodes=None, in_nodes=None):
        if out_nodes == None:
            out_nodes = range(1, len(self.nodes) + 1)
        if in_nodes == None:
            in_nodes = range(1, len(self.nodes) + 1)
        self.edges.update(
            {
                (i, j): Edge(self, (i, j))
                for i in out_nodes
                for j in in_nodes
                if i != j
            }
        )


class DiGraph(Graph):
    def __init__(self):
        super(DiGraph, self).__init__()

    def add_edge(self, in_node, out_node, *edge_dict):
        edge = (in_node, out_node)
        if (out_node, in_node) in self.edges:
            raise ValueError(
                f"Edge {edge} can't be added. Edge in opposite direction already exists!"
            )
        if not in_node in self.nodes or not out_node in self.nodes:
            missing_nodes = [node for node in edge if not (node in self.nodes)]
            raise ValueError(
                f"Edge {edge} can't be added. The graph does not contain the following nodes: {missing_nodes}"
            )
        self.edges[edge] = (
            Edge(self, edge, edge_dict[0]) if edge_dict else Edge(self, edge)
        )


class Edge(object):
    def __init__(self, graph, edge, *edge_dict):
        self.graph = graph

        (out_node, in_node) = edge

        self.edge_dict = edge_dict[0] if edge_dict else {}
        self.out_node = graph.nodes[out_node]
        self.in_node = graph.nodes[in_node]


class Node(object):
    def __init__(self, graph, node):
        self.graph = graph
        self.id = node

    def degree(self):
        return int(
            len([True for edge in self.graph.edges if self.id in edge]) / 2
        )


def create_clustered_graph(K, N):
    """
    Arguments
    ---------
    K : int
        number of clusters
    N : int 
        number of nodes per cluster
    Returns
    -------
    G : Graph
    """
    G = Graph()
    # create clusters
    for k in range(K):
        C_k = list()
        for i in range(1, N + 1):
            node = k * N + i
            G.add_node(node)
            C_k.append(node)
        G.connect_nodes(C_k, C_k)

    # unite clusters with one node from each
    for k in range(K - 1):
        G.add_edge(k * N + 1, (k + 1) * N + 1)
    return G

