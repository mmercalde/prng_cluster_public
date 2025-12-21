"""
Multi-Model ML Architecture v3.1.2

Provides unified interface for multiple ML model types:
- neural_net: PyTorch (wraps existing SurvivorQualityNet)
- xgboost: XGBoost with GPU acceleration
- lightgbm: LightGBM with GPU acceleration
- catboost: CatBoost with multi-GPU support

Usage:
    from models import create_model, ModelSelector, get_feature_schema

    # Create a model
    model = create_model('xgboost', config={'n_estimators': 100})
    
    # Train
    model.fit(X_train, y_train, X_val, y_val)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Compare models
    selector = ModelSelector()
    selector.load_model('neural_net', 'models/reinforcement/best_model.pth')
    selector.load_model('xgboost', 'models/reinforcement/best_model.json')
    results = selector.evaluate_all(X_test, y_test)
"""

from models.model_factory import create_model, load_model, list_available_models
from models.model_selector import ModelSelector
from models.feature_schema import (
    get_feature_schema_from_data,
    get_feature_schema_with_hash,
    validate_feature_schema_hash,
    get_feature_count
)

__version__ = "3.1.2"
__all__ = [
    'create_model',
    'load_model', 
    'list_available_models',
    'ModelSelector',
    'get_feature_schema_from_data',
    'get_feature_schema_with_hash',
    'validate_feature_schema_hash',
    'get_feature_count'
]
