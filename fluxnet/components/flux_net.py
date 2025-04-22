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

from fluxnet.components.ckg_conv import CKGConv

class FluxNet(nn.Module):
    """
    Optimized CKGConv block with GATv2Conv attention mechanism.
    """
    def __init__(self, node_in_dim, edge_in_dim, pe_dim, out_channels,
                 ffn_hidden_dim=None, modulator_hidden_dim=64,
                 dropout=0.0, norm_type='batch', add_self_loops=True, 
                 aggr='mean', num_heads=4, use_attention=True):
        super(FluxNet, self).__init__()
        
        # Original CKGConv components
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * out_channels

        self.conv = CKGConv(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            pe_dim=pe_dim,
            out_channels=out_channels,
            modulator_hidden_dim=modulator_hidden_dim,
            dropout=dropout,
            add_self_loops=add_self_loops,
            aggr=aggr
        )
        
        # Normalization layers
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
            self.norm3 = nn.BatchNorm1d(out_channels) if use_attention else None
        elif norm_type == 'layer':
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
            self.norm3 = nn.LayerNorm(out_channels) if use_attention else None
        elif norm_type == 'instance':
            # InstanceNorm is often faster than BatchNorm for graph data
            self.norm1 = nn.InstanceNorm1d(out_channels)
            self.norm2 = nn.InstanceNorm1d(out_channels)
            self.norm3 = nn.InstanceNorm1d(out_channels) if use_attention else None
        else:  # 'none'
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity() if use_attention else None
            
        # Self-attention module - replaced with GATv2Conv
        self.use_attention = use_attention
        if use_attention:
            self.attention = GATv2Conv(
                in_channels=out_channels,
                out_channels=out_channels,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_in_dim + pe_dim,
                concat=False  # Average attention heads for consistent dimensions
            )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            Linear(out_channels, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(ffn_hidden_dim, out_channels)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_pe, edge_index, edge_attr, edge_pe, batch=None):
        # Store original input for residual connection
        identity = x if x.size(1) == self.conv.out_channels else None
        
        # Apply graph convolution
        x = self.conv(x, x_pe, edge_index, edge_attr, edge_pe, batch)
        x = self.norm1(x)
        
        # Residual connection (if dimensions match)
        if identity is not None:
            x = x + identity
        
        # Apply GAT attention mechanism
        if self.use_attention:
            identity = x
            # Prepare edge features for GAT
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            
            # Precompute the combined edge features once
            edge_features = torch.cat([edge_attr, edge_pe], dim=-1)
            
            # Apply GATv2Conv - more efficient than custom attention
            attn_out = self.attention(x, edge_index, edge_features)
            x = x + self.dropout(attn_out)  # Residual connection with dropout
            x = self.norm3(x)
        
        # Apply FFN with residual connection
        identity = x
        x = self.ffn(x)
        x = self.dropout(x) + identity  # Apply dropout before adding residual
        x = self.norm2(x)
        
        return x

