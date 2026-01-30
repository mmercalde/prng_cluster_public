#!/bin/bash
# Phase 9A Acceptance Engine Patch Installer
# Applies selfplay candidate validation to chapter_13_acceptance.py
#
# VERSION: 9A.1.0
# DATE: 2026-01-30
#
# USAGE: bash install_phase9a_acceptance.sh
#
# CREATES BACKUP: chapter_13_acceptance.py.backup_pre_phase9a

set -e

SOURCE_FILE="chapter_13_acceptance.py"
BACKUP_FILE="${SOURCE_FILE}.backup_pre_phase9a"
POLICIES_FILE="watcher_policies.json"

echo "=========================================="
echo "Phase 9A Acceptance Engine Patch Installer"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Error: $SOURCE_FILE not found"
    echo "   Run this from ~/distributed_prng_analysis/"
    exit 1
fi

# Create backup
echo "📦 Creating backup: $BACKUP_FILE"
cp "$SOURCE_FILE" "$BACKUP_FILE"

# Check if patch already applied
if grep -q "validate_selfplay_candidate" "$SOURCE_FILE"; then
    echo "⚠️  Patch appears to already be applied (found validate_selfplay_candidate)"
    echo "   Skipping code patch. Checking policies..."
else
    echo "📝 Applying code patch..."
    
    # We'll use Python to safely inject the code
    python3 << 'PATCH_SCRIPT'
import re

SOURCE = "chapter_13_acceptance.py"

with open(SOURCE, 'r') as f:
    content = f.read()

# ============================================================
# PATCH 1: Add imports and constants after existing imports
# ============================================================

imports_addition = '''
# Phase 9A: Selfplay candidate validation
SELFPLAY_CANDIDATE_FILE = "learned_policy_candidate.json"
SELFPLAY_ACTIVE_FILE = "learned_policy_active.json"
TELEMETRY_DIR = "telemetry"

# Selfplay validation thresholds
SELFPLAY_MIN_FITNESS = 0.50
SELFPLAY_MIN_VAL_R2 = 0.80
SELFPLAY_MAX_TRAIN_VAL_GAP = 5.0
SELFPLAY_MIN_SURVIVOR_COUNT = 1000
'''

# Insert after DEFAULT_POLICIES line
if "SELFPLAY_CANDIDATE_FILE" not in content:
    content = content.replace(
        'ACCEPTANCE_LOG_FILE = "acceptance_decisions.jsonl"',
        'ACCEPTANCE_LOG_FILE = "acceptance_decisions.jsonl"' + imports_addition
    )

# ============================================================
# PATCH 2: Add SelfplayCandidate dataclass after AcceptanceDecision
# ============================================================

selfplay_dataclass = '''

@dataclass
class SelfplayCandidate:
    """Selfplay policy candidate from Phase 8."""
    schema_version: str
    policy_id: str
    created_at: str
    source: str
    model_type: str
    fitness: float
    val_r2: float
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    episode_id: str
    survivors_source: str
    status: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfplayCandidate':
        """Create from dictionary."""
        return cls(
            schema_version=data.get("schema_version", "unknown"),
            policy_id=data.get("policy_id", "unknown"),
            created_at=data.get("created_at", ""),
            source=data.get("source", "unknown"),
            model_type=data.get("model_type", "unknown"),
            fitness=data.get("fitness", 0.0),
            val_r2=data.get("val_r2", 0.0),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            episode_id=data.get("episode_id", ""),
            survivors_source=data.get("survivors_source", ""),
            status=data.get("status", "unknown")
        )
    
    @classmethod
    def from_file(cls, filepath: str = SELFPLAY_CANDIDATE_FILE) -> 'SelfplayCandidate':
        """Load from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SelfplayCandidateDecision:
    """Decision on a selfplay candidate."""
    result: ValidationResult
    reason: str
    candidate_id: str
    fitness: float
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "reason": self.reason,
            "candidate_id": self.candidate_id,
            "fitness": self.fitness,
            "violations": self.violations,
            "warnings": self.warnings,
            "timestamp": self.timestamp
        }

'''

# Find position after AcceptanceDecision class ends (look for class Chapter13AcceptanceEngine)
if "class SelfplayCandidate" not in content:
    content = content.replace(
        "class Chapter13AcceptanceEngine:",
        selfplay_dataclass + "\nclass Chapter13AcceptanceEngine:"
    )

# ============================================================
# PATCH 3: Add methods to Chapter13AcceptanceEngine class
# Find the last method and insert before main()
# ============================================================

methods_addition = '''
    # =========================================================================
    # Phase 9A: Selfplay Candidate Validation
    # =========================================================================
    
    def validate_selfplay_candidate(
        self, 
        candidate: SelfplayCandidate,
        policies: Optional[Dict[str, Any]] = None
    ) -> SelfplayCandidateDecision:
        """
        Validate a selfplay policy candidate.
        
        Phase 9A: Chapter 13 decides whether to promote a candidate
        to active policy based on deterministic criteria.
        """
        policies = policies or self.policies
        violations = []
        warnings = []
        
        logger.info(f"Validating selfplay candidate: {candidate.policy_id}")
        
        # --- HARD REJECTIONS ---
        
        # 1. Source must be "selfplay"
        if candidate.source != "selfplay":
            violations.append(f"Invalid source: {candidate.source} (expected 'selfplay')")
        
        # 2. Status must be "candidate"
        if candidate.status != "candidate":
            violations.append(f"Invalid status: {candidate.status} (expected 'candidate')")
        
        # 3. Fitness threshold
        selfplay_policies = policies.get("selfplay", {})
        min_fitness = selfplay_policies.get("min_fitness", SELFPLAY_MIN_FITNESS)
        if candidate.fitness < min_fitness:
            violations.append(f"Fitness {candidate.fitness:.4f} below threshold {min_fitness}")
        
        # 4. Validation R² threshold
        min_r2 = selfplay_policies.get("min_val_r2", SELFPLAY_MIN_VAL_R2)
        if candidate.val_r2 < min_r2:
            violations.append(f"val_r2 {candidate.val_r2:.4f} below threshold {min_r2}")
        
        # 5. Train/val gap (overfit detection)
        train_val_gap = candidate.metrics.get("train_val_gap", 0.0)
        max_gap = selfplay_policies.get("max_train_val_gap", SELFPLAY_MAX_TRAIN_VAL_GAP)
        if train_val_gap > max_gap:
            violations.append(f"train_val_gap {train_val_gap:.2f} exceeds threshold {max_gap}")
        
        # 6. Minimum survivor count
        survivor_count = candidate.metrics.get("survivor_count", 0)
        min_survivors = selfplay_policies.get("min_survivor_count", SELFPLAY_MIN_SURVIVOR_COUNT)
        if survivor_count < min_survivors:
            violations.append(f"survivor_count {survivor_count} below threshold {min_survivors}")
        
        # --- WARNINGS (non-blocking) ---
        
        # High fold_std suggests instability
        fold_std = candidate.metrics.get("fold_std", 0.0)
        if fold_std > 0.01:
            warnings.append(f"High fold_std: {fold_std:.6f} (potential instability)")
        
        # Very high training time
        training_time_ms = candidate.metrics.get("training_time_ms", 0)
        if training_time_ms > 60000:
            warnings.append(f"High training time: {training_time_ms/1000:.1f}s")
        
        # --- DECISION ---
        
        if violations:
            decision = SelfplayCandidateDecision(
                result=ValidationResult.REJECT,
                reason=f"Failed validation: {len(violations)} violation(s)",
                candidate_id=candidate.policy_id,
                fitness=candidate.fitness,
                violations=violations,
                warnings=warnings
            )
            logger.warning(f"REJECT candidate {candidate.policy_id}: {violations}")
        
        elif warnings and len(warnings) >= 2:
            decision = SelfplayCandidateDecision(
                result=ValidationResult.ESCALATE,
                reason=f"Multiple warnings ({len(warnings)}) - requires human review",
                candidate_id=candidate.policy_id,
                fitness=candidate.fitness,
                violations=[],
                warnings=warnings
            )
            logger.info(f"ESCALATE candidate {candidate.policy_id}: {warnings}")
        
        else:
            decision = SelfplayCandidateDecision(
                result=ValidationResult.ACCEPT,
                reason="Passed all validation checks",
                candidate_id=candidate.policy_id,
                fitness=candidate.fitness,
                violations=[],
                warnings=warnings
            )
            logger.info(f"ACCEPT candidate {candidate.policy_id} (fitness={candidate.fitness:.4f})")
        
        # Log decision
        self._log_selfplay_decision(decision, candidate)
        
        return decision
    
    def promote_candidate(
        self, 
        candidate: SelfplayCandidate,
        output_path: str = SELFPLAY_ACTIVE_FILE
    ) -> bool:
        """
        Promote a validated candidate to active policy.
        
        AUTHORITY: Only Chapter 13 can call this. Selfplay cannot.
        """
        logger.info(f"Promoting candidate {candidate.policy_id} to active policy")
        
        active_policy = {
            "schema_version": candidate.schema_version,
            "policy_id": candidate.policy_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "promoted_by": "chapter_13_acceptance",
            "source": candidate.source,
            "model_type": candidate.model_type,
            "fitness": candidate.fitness,
            "val_r2": candidate.val_r2,
            "metrics": candidate.metrics,
            "parameters": candidate.parameters,
            "episode_id": candidate.episode_id,
            "survivors_source": candidate.survivors_source,
            "status": "active",
            "original_created_at": candidate.created_at
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(active_policy, f, indent=2)
            logger.info(f"Wrote active policy to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write active policy: {e}")
            return False
        
        # Record promotion via telemetry
        try:
            self._record_promotion_telemetry(candidate, active_policy)
        except Exception as e:
            logger.warning(f"Telemetry recording failed (non-fatal): {e}")
        
        return True
    
    def _log_selfplay_decision(
        self, 
        decision: SelfplayCandidateDecision, 
        candidate: SelfplayCandidate
    ) -> None:
        """Log selfplay decision to audit trail."""
        log_entry = {
            "type": "selfplay_candidate",
            "timestamp": decision.timestamp,
            "candidate_id": candidate.policy_id,
            "model_type": candidate.model_type,
            "fitness": candidate.fitness,
            "result": decision.result.value,
            "reason": decision.reason,
            "violations": decision.violations,
            "warnings": decision.warnings
        }
        
        log_path = Path(ACCEPTANCE_LOG_FILE)
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\\n")
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")
    
    def _record_promotion_telemetry(
        self, 
        candidate: SelfplayCandidate,
        active_policy: Dict[str, Any]
    ) -> None:
        """Record promotion event via learning telemetry."""
        try:
            from modules.learning_telemetry import LearningTelemetry
            
            telemetry = LearningTelemetry(
                run_id=f"promotion_{candidate.policy_id}",
                telemetry_dir=TELEMETRY_DIR
            )
            
            telemetry.record_promotion(
                policy_id=candidate.policy_id,
                fitness=candidate.fitness,
                model_type=candidate.model_type,
                promoted_by="chapter_13_acceptance"
            )
            
            logger.info(f"Recorded promotion telemetry for {candidate.policy_id}")
            
        except ImportError:
            logger.debug("Telemetry module not available")
        except Exception as e:
            logger.warning(f"Telemetry recording failed: {e}")


'''

# Insert before def main():
if "def validate_selfplay_candidate" not in content:
    content = content.replace(
        "\ndef main():",
        methods_addition + "\ndef main():"
    )

# ============================================================
# PATCH 4: Add CLI arguments
# ============================================================

cli_args = '''    mode_group.add_argument("--validate-selfplay", type=str, metavar="FILE",
                           help="Validate a selfplay candidate JSON file")
    mode_group.add_argument("--promote", type=str, metavar="FILE",
                           help="Promote a selfplay candidate to active policy")
'''

if "--validate-selfplay" not in content:
    content = content.replace(
        'mode_group.add_argument("--test"',
        cli_args + '    mode_group.add_argument("--test"'
    )

# ============================================================
# PATCH 5: Add CLI handlers
# ============================================================

cli_handlers = '''
        if args.validate_selfplay:
            candidate = SelfplayCandidate.from_file(args.validate_selfplay)
            decision = engine.validate_selfplay_candidate(candidate)
            
            if args.json:
                print(json.dumps(decision.to_dict(), indent=2))
            else:
                print(f"\\n{'='*60}")
                print("CHAPTER 13 ACCEPTANCE ENGINE — Selfplay Validation")
                print(f"{'='*60}")
                print(f"\\n📋 Candidate: {decision.candidate_id}")
                print(f"   Model: {candidate.model_type}")
                print(f"   Fitness: {candidate.fitness:.4f}")
                print(f"   Val R²: {candidate.val_r2:.4f}")
                print(f"\\n📊 Result: {decision.result.value.upper()}")
                print(f"   Reason: {decision.reason}")
                if decision.violations:
                    print(f"\\n❌ Violations:")
                    for v in decision.violations:
                        print(f"      - {v}")
                if decision.warnings:
                    print(f"\\n⚠️  Warnings:")
                    for w in decision.warnings:
                        print(f"      - {w}")
                if decision.result == ValidationResult.ACCEPT:
                    print(f"\\n✅ Candidate approved for promotion")
                    print(f"   Run: --promote {args.validate_selfplay}")
                print()
            return 0
        
        if args.promote:
            candidate = SelfplayCandidate.from_file(args.promote)
            decision = engine.validate_selfplay_candidate(candidate)
            
            if decision.result != ValidationResult.ACCEPT:
                print(f"\\n❌ Cannot promote: candidate failed validation")
                print(f"   Result: {decision.result.value}")
                print(f"   Reason: {decision.reason}")
                return 1
            
            success = engine.promote_candidate(candidate)
            
            if success:
                print(f"\\n{'='*60}")
                print("CHAPTER 13 ACCEPTANCE ENGINE — Promotion Complete")
                print(f"{'='*60}")
                print(f"\\n✅ Promoted: {candidate.policy_id}")
                print(f"   Model: {candidate.model_type}")
                print(f"   Fitness: {candidate.fitness:.4f}")
                print(f"   Output: {SELFPLAY_ACTIVE_FILE}")
                print()
                return 0
            else:
                print(f"\\n❌ Promotion failed")
                return 1

'''

if "args.validate_selfplay" not in content:
    # Insert before "if args.status:"
    content = content.replace(
        "        if args.status:",
        cli_handlers + "        if args.status:"
    )

# Write patched file
with open(SOURCE, 'w') as f:
    f.write(content)

print("✅ Code patch applied successfully")
PATCH_SCRIPT
fi

# ============================================================
# Update watcher_policies.json
# ============================================================

echo "📝 Updating watcher_policies.json..."

python3 << 'POLICIES_SCRIPT'
import json

POLICIES = "watcher_policies.json"

with open(POLICIES, 'r') as f:
    policies = json.load(f)

# Add selfplay section if not present
if "selfplay" not in policies:
    policies["selfplay"] = {
        "min_fitness": 0.50,
        "min_val_r2": 0.80,
        "max_train_val_gap": 5.0,
        "min_survivor_count": 1000,
        "auto_promote": False,
        "require_human_approval": True
    }
    
    with open(POLICIES, 'w') as f:
        json.dump(policies, f, indent=2)
    
    print("✅ Added selfplay section to watcher_policies.json")
else:
    print("⚠️  selfplay section already exists in watcher_policies.json")
POLICIES_SCRIPT

# ============================================================
# Verify
# ============================================================

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

echo ""
echo "📋 Checking for new functions..."
if grep -q "def validate_selfplay_candidate" "$SOURCE_FILE"; then
    echo "   ✅ validate_selfplay_candidate() present"
else
    echo "   ❌ validate_selfplay_candidate() NOT FOUND"
fi

if grep -q "def promote_candidate" "$SOURCE_FILE"; then
    echo "   ✅ promote_candidate() present"
else
    echo "   ❌ promote_candidate() NOT FOUND"
fi

if grep -q "validate-selfplay" "$SOURCE_FILE"; then
    echo "   ✅ --validate-selfplay CLI flag present"
else
    echo "   ❌ --validate-selfplay CLI flag NOT FOUND"
fi

echo ""
echo "📋 Testing syntax..."
python3 -m py_compile "$SOURCE_FILE" && echo "   ✅ Syntax OK" || echo "   ❌ Syntax error"

echo ""
echo "📋 Testing import..."
python3 -c "import chapter_13_acceptance; print('   ✅ Import OK')" 2>/dev/null || echo "   ❌ Import error"

echo ""
echo "=========================================="
echo "Installation Complete"
echo "=========================================="
echo ""
echo "Test with:"
echo "  python3 chapter_13_acceptance.py --validate-selfplay learned_policy_candidate.json"
echo ""
echo "To promote (after validation passes):"
echo "  python3 chapter_13_acceptance.py --promote learned_policy_candidate.json"
echo ""
