import torch
import torch.nn as nn
from torch_geometric.data import Batch
from .model import EncoderProcesserDecoder


class FEMSurrogate(nn.Module):
    """GNN for FEM: velocity → next velocity"""

    def __init__(self, n_nodes=4, message_passing_num=3, hidden_dim=128, node_feat_dim=3, edge_feat_dim=0):
        super().__init__()
        self.n_nodes = n_nodes
        
        self.gnn = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            node_input_size=node_feat_dim, 
            edge_input_size=edge_feat_dim,   
            hidden_size=hidden_dim,
            output_size=3,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """batch: Batch of Data objects → (batch*4, 3)"""
        return self.gnn(batch)


__all__ = ["FEMSurrogate"]
