"""
Model Wrappers Package (v3.1.2)

Provides unified interface wrappers for different ML model types:
- NeuralNetWrapper: PyTorch (wraps existing SurvivorQualityNet)
- XGBoostWrapper: XGBoost with GPU acceleration
- LightGBMWrapper: LightGBM with GPU acceleration
- CatBoostWrapper: CatBoost with multi-GPU support

All wrappers implement the ModelInterface protocol.
"""

from models.wrappers.base import ModelInterface, TorchModelMixin

__all__ = [
    'ModelInterface',
    'TorchModelMixin',
]

# Lazy imports to avoid dependency issues
def get_neural_net_wrapper():
    from models.wrappers.neural_net_wrapper import NeuralNetWrapper
    return NeuralNetWrapper

def get_xgboost_wrapper():
    from models.wrappers.xgboost_wrapper import XGBoostWrapper
    return XGBoostWrapper

def get_lightgbm_wrapper():
    from models.wrappers.lightgbm_wrapper import LightGBMWrapper
    return LightGBMWrapper

def get_catboost_wrapper():
    from models.wrappers.catboost_wrapper import CatBoostWrapper
    return CatBoostWrapper
