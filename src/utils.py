import torch
import torch.nn as nn
from torch import inverse
from torch_geometric.utils import degree


def total_degree(edge_index, num_nodes):
    """
    Arguments
    ---------
    edge_index : torch.Tensor [2, num_edges]
    num_nodes : int
    
    Return
    ------
    total_degree : torch.Tensor [num_nodes]
    """
    return out_degree(edge_index, num_nodes) + in_degree(edge_index, num_nodes)


def out_degree(edge_index, num_nodes):
    """
    Arguments
    ---------
    edge_index : torch.Tensor [2, num_edges]
    num_nodes : int
    
    Returns
    -------
    out_degree : torch.Tensor [num_nodes]
    """
    return degree(edge_index[0], num_nodes)


def in_degree(edge_index, num_nodes):
    """
    Arguments
    ---------
    edge_index : torch.Tensor [2, num_edges]
    num_nodes : int
    
    Returns
    -------
    in_degree : torch.Tensor [num_nodes]
    """
    return degree(edge_index[1], num_nodes)
