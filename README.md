# FluxNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-blue)](https://pytorch-geometric.readthedocs.io/)

FluxNet is a specialized graph neural network architecture that combines continuous kernel graph convolutions with attention mechanisms for enhanced graph representation learning.

[Project Website](https://nishilkulkarni.github.io/fluxnet/)

## Features

- **Continuous Kernel Graph Convolution**: Implements the CKGConv layer for effective message passing
- **Adaptive Degree Scaling**: Handles graphs with varying node degrees using learnable parameters
- **GATv2 Attention**: Integrates dynamic attention mechanisms for improved graph learning
- **Feature Modulation**: Transforms edge features to modulate node representations
- **Flexible Architecture**: Configurable normalization, aggregation, and residual connections

## Installation

```bash
# Clone the repository
git clone https://github.com/nishilkulkarni/fluxnet.git
cd fluxnet

# Create a virtual environment (optional but recommended)
python -m venv fluxnet-env
source fluxnet-env/bin/activate  # On Windows: fluxnet-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```python
import torch
from fluxnet import FluxNet

# Create a FluxNet layer
model = FluxNet(
    node_in_dim=32, 
    edge_in_dim=16, 
    pe_dim=8,
    out_channels=64,
    dropout=0.1,
    norm_type='layer'
)

# Input features
num_nodes = 100
num_edges = 500
x = torch.randn(num_nodes, 32)
x_pe = torch.randn(num_nodes, 8)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_attr = torch.randn(num_edges, 16)
edge_pe = torch.randn(num_edges, 8)

# Forward pass
output = model(x, x_pe, edge_index, edge_attr, edge_pe)
print(output.shape)  # [100, 64]
```

## Components

FluxNet consists of three main components:

1. **FeatureModulator**: Transforms edge features to modulate node features
2. **CKGConv**: Specialized graph convolution with adaptive degree scaling
3. **FluxNet**: Combines CKGConv with GATv2 attention and feed-forward networks

## Examples

See the `examples/` directory for usage examples on various graph learning tasks:

- Node classification
- Graph classification
- Graph regression

## Requirements

- Python 3.8+
- PyTorch 1.8.0+
- PyTorch Geometric 2.0.0+
- Additional requirements in requirements.txt

## Citation

If you use FluxNet in your research, please cite our work:

```bibtex
@article{nishil2025fluxnet,
  title={FluxNet: Continuous Kernel Graph Convolutions with Multi-Head Attention},
  author={Nishil Kulkarni},
  journal={arXiv preprint arXiv:2504.xxxxx},
  year={2025},
  note={\url{https://github.com/nishilkulkarni/fluxnet}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch Geometric team for their excellent graph deep learning framework
- Contributors and researchers who have provided feedback and suggestions
