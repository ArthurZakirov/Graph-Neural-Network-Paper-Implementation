import torch
import torch.nn as nn
from torch import inverse
import torch_geometric as tg


class GNN(tg.nn.MessagePassing):
    def __init__(self, d):
        super(GNN, self).__init__(
            aggr="mean", flow="source_to_target", node_dim=-2
        )

        self.W_j = nn.Linear(d, d)
        self.W_i = nn.Linear(d, d)
        self.act = nn.ReLU()

    def message(self, x_j):
        """
        Arguments
        ---------
        x_j : torch.Tensor [E, d]
        
        Returns
        -------
        m_j : torch.Tensor [E, d]
        """
        m_j = self.act(self.W_j(x_j))
        return m_j

    def update(self, aggr_out, x):
        """
        Arguments
        ---------
        x : torch.Tensor [N, d]
        aggr_out : torch.Tensor [N, d]
        
        Returns
        -------
        x_k : torch.Tensor [N, d]
        """
        m = aggr_out

        x_k = m + self.W_i(x)
        return x_k

    def forward(self, edge_index, x):
        """
        Arguments
        ---------
        edge_index : torch.tensor [2, E]
        x : torch.tensor [N, d]
        
        Returns
        -------
        x_k : torch.Tensor [N, d]
        """
        x_k = self.propagate(edge_index, x=x)
        return x_k
