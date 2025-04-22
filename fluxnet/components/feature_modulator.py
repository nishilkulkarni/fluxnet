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



class FeatureModulator(nn.Module):
    """
    Neural network that modulates node features based on edge features.
    Continuos Kernel function.
    """
    def __init__(self, edge_dim, node_dim, hidden_dim=64, dropout=0.0):
        super(FeatureModulator, self).__init__()
        self.mlp = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, node_dim)
        )

    def forward(self, edge_features):
        return self.mlp(edge_features)

