import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset, LRGBDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset, LRGBDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn.inits import reset
import math
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import warnings 
import itertools, copy
import random 

from fluxnet.components.feature_modulator import FeatureModulator

class CKGConv(MessagePassing):
    """
    Optimized convolution layer that utilizes concatenated node/edge features
    with positional encodings and applies adaptive degree scaling.
    """
    def __init__(self, node_in_dim, edge_in_dim, pe_dim, out_channels,
                 modulator_hidden_dim=64, dropout=0.0, add_self_loops=True,
                 aggr='mean'):
        super(CKGConv, self).__init__(aggr=aggr, node_dim=0)
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.pe_dim = pe_dim
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        # Combined dimensions
        self.node_feature_dim = node_in_dim + pe_dim
        self.edge_feature_dim = edge_in_dim + pe_dim

        # Feature modulator (ψ function)
        self.modulator = FeatureModulator(
            edge_dim=self.edge_feature_dim,
            node_dim=self.node_feature_dim,
            hidden_dim=modulator_hidden_dim,
            dropout=dropout
        )

        # Linear transformation
        self.linear = Linear(self.node_feature_dim, out_channels)

        # Learnable degree scaling parameters
        self.theta1 = nn.Parameter(torch.ones(out_channels))
        self.theta2 = nn.Parameter(torch.zeros(out_channels))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        reset(self.modulator.mlp)
        self.linear.reset_parameters()
        nn.init.ones_(self.theta1)
        nn.init.zeros_(self.theta2)

    def forward(self, x, x_pe, edge_index, edge_attr, edge_pe, batch=None):
        """
        Forward pass of the layer.
        """
        num_nodes = x.size(0)

        # Concat raw features with positional encodings
        x = torch.cat([x, x_pe], dim=-1)

        # Handle 1D edge attributes
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Concat edge features with positional encodings
        edge_attr = torch.cat([edge_attr, edge_pe], dim=-1)

        # Add self-loops if specified
        if self.add_self_loops:
            # Implementation left to PyTorch Geometric's MessagePassing
            pass

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        # Compute node degrees (optionally includes self-loops)
        deg = degree(edge_index[0], num_nodes=num_nodes).to(x.dtype)
        deg = deg.clamp(min=1)

        # Apply adaptive degree scaling with learnable parameters
        deg_sqrt = deg.sqrt().view(-1, 1)
        out = out * self.theta1 + deg_sqrt * (out * self.theta2)

        return out

    def message(self, x_j, edge_attr):
        """
        Message computation: modulate source node features with edge features
        """
        # Apply modulator to get weights for node features
        edge_weights = self.modulator(edge_attr)
        
        # Element-wise multiplication of source features with modulated weights
        return x_j * edge_weights

    def update(self, aggr_out):
        """
        Update function: apply linear transformation to aggregated messages
        """
        return self.linear(aggr_out)

