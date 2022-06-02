import torch
import torch.nn as nn
from torch import inverse

import torch_geometric as tg
from torch_geometric.utils import add_self_loops


class MultiHeadAttention(nn.Module):
    def __init__(self, K, d_x, d_h):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(d_x, d_h) for _ in range(K)]
        )

        self.W = nn.Linear(K * d_h, d_h)

    def forward(self, edge_index, x):
        """
        Arguments
        ---------
        edge_index : torch.tensor [2, E]
        x : torch.tensor [N, d_x]
        
        Returns
        -------
        h : torch.tensor [N, d_h]
        """
        h_k = [head(edge_index, x) for head in self.heads]
        h = self.W(torch.cat(h_k, dim=-1))
        return h


class SingleHeadAttention(tg.nn.MessagePassing):
    def __init__(self, d_x, d_h):
        super(SingleHeadAttention, self).__init__()

        self.W = nn.Linear(d_x, d_h)
        self.a = nn.Linear(2 * d_h, 1)

    def attention(self, x_i, x_j):
        """
        Arguments
        ---------
        x_i : torch.tensor [E, d_x]
        x_j : torch.tensor [E, d_x]

        Returns
        -------
        e_ij : torch.tensor [E, 1]
        """
        e_ij = nn.LeakyReLU()(
            self.a(torch.cat([self.W(x_i), self.W(x_j)], dim=-1))
        )
        return e_ij

    def alpha(self, edge_index, x_i, x_j):
        """
        Arguments
        ---------
        edge_index : torch.tensor [E, 2]
        x_i : torch.tensor [E, d_x]
        x_j : torch.tensor [E, d_x]
        
        Returns
        -------
        a_ij : torch.tensor [E, 1]
        """
        neighbours, node = edge_index
        e_ij = self.attention(x_i, x_j)
        a_ij = torch.zeros_like(e_ij)
        for i in range(x.shape[-2]):
            i_edges = node == i
            a_ij[i_edges] = nn.Softmax(-2)(e_ij[i_edges])
        return a_ij

    def message(self, edge_index, x_i, x_j):
        """
        Arguments
        ---------
        edge_index : torch.tensor [E, 2]
        x_i : torch.tensor [E, d_x]
        x_j : torch.tensor [E, d_x]
        
        Returns
        -------
        m_j : torch.tensor [E, d_h]
        """
        a_ij = self.alpha(edge_index, x_i, x_j)
        m_j = a_ij * self.W(x_j)
        return m_j

    def update(self, aggr_out):
        """
        Arguments
        ---------
        aggr_out : torch.tensor [N, d_h]
        
        Returns
        -------
        h : torch.tensor [N, d_h]
        """
        m = aggr_out
        h = nn.Softmax(-2)(m)
        return h

    def forward(self, edge_index, x):
        """
        Arguments
        ---------
        edge_index : torch.tensor [2, E]
        x : torch.tensor [N, d_x]
        
        Returns
        -------
        h : torch.tensor [N, d_h]
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[-2])
        h = self.propagate(edge_index=edge_index, x=x)
        return h
