import torch
import torch.nn as nn
from torch import inverse

import torch_geometric as tg
from torch_geometric.utils import add_self_loops


class GCN(tg.nn.MessagePassing):
    def __init__(self, d):
        super(GCN, self).__init__(aggr="add", flow="source_to_target")
        self.W = nn.Linear(d, d)
        self.act = nn.Softmax(-1)

    def normalize_edges(self, edge_index, num_nodes):
        """
        Arguments
        ---------
        edge_index : torch.tensor [2, E]
        num_nodes : int

        Returns
        -------
        norm : torch.tensor [E, 1]
        """
        d_inv_sqrt = in_degree(edge_index, num_nodes=num_nodes).pow(-0.5)
        out_nodes, in_nodes = edge_index
        norm = d_inv_sqrt[out_nodes] * d_inv_sqrt[in_nodes]
        return norm.view(-1, 1)

    def message(self, edge_index, x, norm, x_j):
        """
        Arguments
        ---------
        edge_index : torch.tensor [2, E]
        x : torch.tensor [N, d]
        x_j : torch.tensor [E, d]
        
        Returns
        -------
        m_j : torch.tensor [E, d]
        """
        norm = self.normalize_edges(edge_index, x.shape[-2])
        m_j = x_j * norm
        return m_j

    def update(self, aggr_out):
        """
        Arguments
        ---------
        m : torch.tensor [N, d]
        
        Returns
        -------
        x_k : torch.tensor [N, d]
        """
        m = aggr_out
        x_k = self.act(m)
        return m

    def forward(self, edge_index, x):
        x = self.W(x)
        edge_index = add_self_loops(edge_index, num_nodes=x.shape[-2])[0]
        return self.propagate(edge_index, x=x, norm=norm)

