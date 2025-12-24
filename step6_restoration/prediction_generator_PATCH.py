#!/usr/bin/env python3
"""
PREDICTION_GENERATOR.PY PATCH - Step 6 Restoration v2.2
========================================================

This file contains the code changes needed for prediction_generator.py.

KEY TEAM BETA REQUIREMENTS ADDRESSED:
1. Lazy-import enforcement - only load winning backend from sidecar
2. Sidecar schema keys - support both legacy (feature_names) and new (per_seed_feature_names)
3. Handle both int and dict survivor formats
4. Use model.predict() for scoring

"""

# =============================================================================
# SECTION 1: ADD TO IMPORTS (after line ~55)
# =============================================================================

# Add this import block - LAZY LOADING (Team Beta Requirement #1)
# GlobalStateTracker is GPU-neutral, safe to import at module level
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES


# =============================================================================
# SECTION 2: UPDATE PredictionConfig dataclass (add models_dir field)
# =============================================================================

# Add this field to the PredictionConfig dataclass (around line 90):
#     models_dir: str = 'models/reinforcement'

# Also add survivors_forward: str = '' for schema validation


# =============================================================================
# SECTION 3: REPLACE __init__ method (around line 145)
# =============================================================================

def __init__(self, config: 'PredictionConfig', logger: logging.Logger):
    """Initialize PredictionGenerator with model loading."""
    self.config = config
    self.logger = logger
    
    # Initialize survivor scorer (Whitepaper Section 4.1)
    self.scorer = SurvivorScorer(
        prng_type=config.prng_type,
        mod=config.mod
    )
    
    # Model loading with lazy imports (Team Beta Requirement #1)
    # Only load the specific backend needed based on sidecar model_type
    self.model = None
    self.model_meta = None
    
    if MULTI_MODEL_AVAILABLE:
        models_dir = getattr(config, 'models_dir', 'models/reinforcement')
        survivors_file = getattr(config, 'survivors_forward', None)
        
        try:
            # load_model_from_sidecar handles lazy loading internally
            self.model, self.model_meta = load_model_from_sidecar(
                models_dir=models_dir,
                device='cuda' if GPU_AVAILABLE else 'cpu',
                survivors_file=survivors_file,
                strict=True
            )
            self.logger.info(f"  Model loaded: {self.model_meta['model_type']}")
            self.logger.info(f"  Checkpoint: {self.model_meta['checkpoint_path']}")
            
            # Log feature schema info
            fs = self.model_meta.get('feature_schema', {})
            total = fs.get('total_features', fs.get('feature_count', 'unknown'))
            self.logger.info(f"  Total features expected: {total}")
            
        except FileNotFoundError as e:
            self.logger.warning(f"  Model not found: {e}")
            self.logger.warning("  Will use fallback scoring (feature mean)")
        except Exception as e:
            self.logger.error(f"  Model loading failed: {e}")
            self.logger.warning("  Will use fallback scoring (feature mean)")
    else:
        self.logger.warning("  Multi-model architecture not available")
    
    # GlobalStateTracker will be initialized with lottery_history
    self.global_tracker = None
    
    self.logger.info("PredictionGenerator initialized")
    self.logger.info(f"  PRNG: {config.prng_type}, mod={config.mod}")
    self.logger.info(f"  Pool size: {config.pool_size}, Top-K: {config.k}")
    self.logger.info(f"  Dual-sieve: {config.use_dual_sieve}")
    self.logger.info(f"  GPU: {GPU_AVAILABLE}")


# =============================================================================
# SECTION 4: REPLACE generate_predictions method
# =============================================================================

def generate_predictions(
    self,
    survivors_forward: List,  # Can be List[int] or List[Dict]
    lottery_history: List[int],
    survivors_reverse: Optional[List] = None,
    k: Optional[int] = None
) -> Dict:
    """
    Generate Top-K predictions from survivors using trained ML model.
    
    Args:
        survivors_forward: Forward sieve survivors (int seeds or dict with features)
        lottery_history: Historical lottery draws
        survivors_reverse: Reverse sieve survivors (optional)
        k: Number of predictions (default: config.k)
        
    Returns:
        Dictionary with predictions, confidence scores, and metadata
    """
    k = k or self.config.k
    
    self.logger.info(f"Generating {k} predictions from {len(survivors_forward)} survivors")
    
    # Initialize GlobalStateTracker with current lottery history
    self.global_tracker = GlobalStateTracker(
        lottery_history=lottery_history,
        config={'mod': self.config.mod}
    )
    global_values = self.global_tracker.get_feature_values()
    self.logger.info(f"Global features computed: {len(global_values)}")
    
    # Determine working survivors
    method = 'forward_only'
    working_survivors = survivors_forward
    intersection_result = None
    
    if survivors_reverse and self.config.use_dual_sieve:
        self.logger.info(f"Dual-sieve mode: {len(survivors_reverse)} reverse survivors")
        method = 'dual_sieve'
        
        # Extract seeds for intersection (handle both int and dict formats)
        forward_seeds = [s['seed'] if isinstance(s, dict) else s for s in survivors_forward]
        reverse_seeds = [s['seed'] if isinstance(s, dict) else s for s in survivors_reverse]
        
        # Compute intersection (returns Dict per Team Beta requirement)
        intersection_result = self.scorer.compute_dual_sieve_intersection(
            forward_seeds, reverse_seeds
        )
        
        intersection_seeds = set(intersection_result["intersection"])
        jaccard = intersection_result["jaccard"]
        self.logger.info(f"Intersection: {len(intersection_seeds)} survivors (Jaccard: {jaccard:.4f})")
        
        if intersection_seeds:
            # Filter to intersection
            working_survivors = [
                s for s in survivors_forward
                if (s['seed'] if isinstance(s, dict) else s) in intersection_seeds
            ]
        else:
            self.logger.warning("No intersection - using forward survivors only")
    
    # Build prediction pool using trained model
    pool_result = self._build_prediction_pool(
        survivors=working_survivors,
        lottery_history=lottery_history,
        global_values=global_values,
        pool_size=self.config.pool_size,
        skip=self.config.skip
    )
    
    # Build response
    result = {
        'predictions': pool_result.get('predictions', [])[:k],
        'confidence_scores': pool_result.get('confidence_scores', [])[:k],
        'metadata': {
            'method': method,
            'model_type': pool_result.get('model_type', 'unknown'),
            'survivor_count': pool_result.get('survivor_count', 0),
            'pool_size': pool_result.get('pool_size', 0),
            'global_features_used': len(global_values),
            'total_features': pool_result.get('total_features', 0),
        }
    }
    
    if intersection_result:
        result['metadata']['intersection'] = intersection_result
    
    # Save if configured
    if self.config.save_predictions:
        self._save_predictions(result)
    
    return result


# =============================================================================
# SECTION 5: ADD _build_prediction_pool method
# =============================================================================

def _build_prediction_pool(
    self,
    survivors: List,  # List[int] or List[Dict]
    lottery_history: List[int],
    global_values: np.ndarray,
    pool_size: int = 10,
    skip: int = 0
) -> Dict[str, Any]:
    """
    Build prediction pool using trained model.
    
    Per Team Beta Requirements:
    - Uses model.predict() for scoring, NOT features['score']
    - Uses sidecar feature_names ordering (supports legacy + new keys)
    - Handles both int and dict survivor formats
    - Appends global features to match training schema
    - Validates feature count matches model expectation
    
    Args:
        survivors: List of seeds (int) or scored survivors (dict)
        lottery_history: Historical draws
        global_values: Global feature values from GlobalStateTracker
        pool_size: Number of top predictions
        skip: PRNG skip value
        
    Returns:
        Dict with predictions, confidence_scores, and metadata
    """
    if not survivors:
        return self._empty_pool_result()
    
    # Get feature schema from sidecar (Team Beta Requirement #2: support legacy keys)
    if self.model_meta:
        feature_schema = self.model_meta.get('feature_schema', {})
        # Support both new (per_seed_feature_names) and legacy (feature_names) keys
        per_seed_names = feature_schema.get('per_seed_feature_names', 
                                            feature_schema.get('feature_names', []))
        global_feature_count = feature_schema.get('global_feature_count', 0)
        total_features = feature_schema.get('total_features', 
                                            len(per_seed_names) + global_feature_count)
    else:
        per_seed_names = []
        global_feature_count = 0
        total_features = 0
    
    # Build feature matrix
    X_per_seed = []
    survivor_seeds = []
    
    for survivor in survivors:
        # Handle both int and dict formats (Team Beta Blocker #2)
        if isinstance(survivor, dict):
            seed = survivor['seed']
            features = survivor.get('features', {})
        elif isinstance(survivor, (int, np.integer)):
            seed = int(survivor)
            # Compute features if not provided
            features = self.scorer.extract_ml_features(
                seed, lottery_history, skip=skip
            )
        else:
            self.logger.warning(f"Unexpected survivor type: {type(survivor)}")
            continue
        
        # Extract per-seed features in sidecar order
        if per_seed_names:
            row = [float(features.get(name, 0.0)) for name in per_seed_names]
        else:
            # Fallback: use all features except score/confidence in sorted order
            row = [
                float(v) for k, v in sorted(features.items())
                if k not in ('score', 'confidence')
            ]
        
        X_per_seed.append(row)
        survivor_seeds.append(seed)
    
    if not X_per_seed:
        return self._empty_pool_result()
    
    X_per_seed = np.array(X_per_seed, dtype=np.float32)
    
    # Append global features if model expects them
    if global_feature_count > 0 and len(global_values) > 0:
        global_broadcast = np.tile(global_values, (X_per_seed.shape[0], 1))
        X = np.hstack([X_per_seed, global_broadcast])
        self.logger.info(f"Features: {X_per_seed.shape[1]} per-seed + {len(global_values)} global = {X.shape[1]}")
    else:
        X = X_per_seed
        if global_feature_count > 0:
            self.logger.warning(f"Sidecar expects {global_feature_count} global features but none provided")
    
    # Validate feature count (fail-fast per Team Beta)
    if total_features > 0 and X.shape[1] != total_features:
        # Check if this is an old sidecar without global features
        if X.shape[1] == len(per_seed_names) and global_feature_count == 0:
            self.logger.info("Legacy sidecar detected (no global features)")
        else:
            raise ValueError(
                f"FATAL: Feature count mismatch!\n"
                f"  Built: {X.shape[1]} (per_seed={X_per_seed.shape[1]}, global={len(global_values) if global_feature_count > 0 else 0})\n"
                f"  Expected: {total_features} (from sidecar)\n"
                f"  This indicates schema drift between training and inference."
            )
    
    # Score using trained model (Team Beta: model.predict(), not features['score'])
    if self.model is not None:
        predicted_quality = self.model.predict(X)
        model_type = self.model_meta.get('model_type', 'unknown')
        self.logger.info(f"Model predictions: min={predicted_quality.min():.4f}, max={predicted_quality.max():.4f}, mean={predicted_quality.mean():.4f}")
    else:
        # Fallback: use mean of features as proxy
        self.logger.warning("No model loaded - using feature mean as fallback")
        predicted_quality = np.mean(X, axis=1)
        model_type = 'fallback_mean'
    
    # Rank by predicted quality (descending)
    ranked_indices = np.argsort(predicted_quality)[::-1]
    
    # Generate predictions from top survivors
    predictions = []
    confidence_scores = []
    next_draw_index = len(lottery_history)
    
    for idx in ranked_indices[:pool_size]:
        seed = survivor_seeds[idx]
        quality = float(predicted_quality[idx])
        
        # Generate next prediction using PRNG
        seq = self.scorer._generate_sequence(seed, next_draw_index + 1, skip=skip)
        if len(seq) > next_draw_index:
            predictions.append(int(seq[next_draw_index]))
            confidence_scores.append(quality)
    
    # Aggregate predictions with weighted confidence
    unique_preds: Dict[int, Dict] = {}
    for pred, conf in zip(predictions, confidence_scores):
        if pred not in unique_preds:
            unique_preds[pred] = {'confidence': conf, 'count': 1}
        else:
            unique_preds[pred]['confidence'] += conf
            unique_preds[pred]['count'] += 1
    
    # Sort by aggregated confidence
    sorted_preds = sorted(unique_preds.items(), key=lambda x: x[1]['confidence'], reverse=True)
    
    final_predictions = [p[0] for p in sorted_preds[:pool_size]]
    final_confidences = [p[1]['confidence'] for p in sorted_preds[:pool_size]]
    
    return {
        'predictions': final_predictions,
        'confidence_scores': final_confidences,
        'survivor_count': len(survivors),
        'pool_size': len(final_predictions),
        'mean_confidence': float(np.mean(final_confidences)) if final_confidences else 0.0,
        'model_type': model_type,
        'global_features_used': len(global_values) if global_feature_count > 0 else 0,
        'total_features': X.shape[1]
    }


# =============================================================================
# SECTION 6: ADD _empty_pool_result helper
# =============================================================================

def _empty_pool_result(self) -> Dict[str, Any]:
    """Return empty result structure."""
    return {
        'predictions': [],
        'confidence_scores': [],
        'survivor_count': 0,
        'pool_size': 0,
        'mean_confidence': 0.0,
        'model_type': None,
        'global_features_used': 0,
        'total_features': 0
    }


# =============================================================================
# SECTION 7: UPDATE main() CLI to add --models-dir argument
# =============================================================================

# Add this argument to argparse (around line 375):
#     parser.add_argument('--models-dir', type=str, default='models/reinforcement',
#                         help='Directory containing best_model.meta.json')

# Update config loading to include models_dir:
#     if args.models_dir:
#         config.models_dir = args.models_dir
#     if args.survivors_forward:
#         config.survivors_forward = args.survivors_forward


# =============================================================================
# COMPLETE PATCH CHECKLIST
# =============================================================================
"""
To apply this patch to prediction_generator.py:

1. Add import at top (after Multi-Model imports):
   from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_NAMES

2. Add to PredictionConfig dataclass:
   models_dir: str = 'models/reinforcement'
   survivors_forward: str = ''

3. Replace __init__ method with new version

4. Replace generate_predictions method with new version

5. Add _build_prediction_pool method

6. Add _empty_pool_result method

7. Update main() CLI:
   - Already has --models-dir ✓
   - Add config.models_dir = args.models_dir
   - Add config.survivors_forward = args.survivors_forward

8. Test:
   python3 prediction_generator.py \
       --models-dir models/reinforcement \
       --survivors-forward survivors_with_scores.json \
       --lottery-history synthetic_lottery.json \
       --k 10
"""
