# Import main components to make them available at the top level
from fluxnet.components.feature_modulator import FeatureModulator
from fluxnet.components.ckg_conv import CKGConv
from fluxnet.components.flux_net import FluxNet

# Define version
__version__ = "0.1.0"

# Expose key components at the module level
__all__ = [
    "FeatureModulator",
    "CKGConv",
    "FluxNet",
]