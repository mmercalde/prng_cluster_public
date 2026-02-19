#!/usr/bin/env python3
"""
Category B Phase 1B — Patch neural_net_wrapper.py
==================================================

Adds use_leaky_relu parameter to SurvivorQualityNet constructor.
When True, replaces ReLU activations with LeakyReLU(0.01).

Verified against live code:
  github.com/mmercalde/prng_cluster_public/main/models/wrappers/neural_net_wrapper.py

Key facts:
  - SurvivorQualityNet has nn.BatchNorm1d (always-on) — NOT TOUCHED
  - Dynamic layer building via loop over hidden_layers
  - nn.ReLU() appears once inside the loop
  - Also patches NeuralNetWrapper.load() to pass use_leaky_relu from checkpoint

Author: Team Alpha (S92)
Date: 2026-02-15
"""

import sys
import shutil
from pathlib import Path

TARGET = Path("models/wrappers/neural_net_wrapper.py")
BACKUP = TARGET.with_suffix(".pre_category_b_phase1")


def verify_preconditions():
    """Verify the file is in expected state."""
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        return False

    content = TARGET.read_text()

    if 'use_leaky_relu' in content:
        print("ERROR: use_leaky_relu already present. Already patched?")
        return False

    # Verify exact anchor strings from live code
    anchors = [
        'class SurvivorQualityNet(nn.Module):',
        'def __init__(self, input_size: int, hidden_layers: List[int], dropout: float = 0.3):',
        "layers.append(nn.ReLU())",
        "self.network = nn.Sequential(*layers)",
        "class NeuralNetWrapper(TorchModelMixin, GPUMemoryMixin):",
    ]
    for anchor in anchors:
        if anchor not in content:
            print(f"ERROR: Missing anchor: {anchor}")
            return False

    print("All preconditions PASSED")
    return True


def apply_patch():
    content = TARGET.read_text()

    # ================================================================
    # PATCH 1: Update SurvivorQualityNet __init__ signature
    # ================================================================
    old_init = '    def __init__(self, input_size: int, hidden_layers: List[int], dropout: float = 0.3):'
    new_init = '    def __init__(self, input_size: int, hidden_layers: List[int], dropout: float = 0.3, use_leaky_relu: bool = False):'

    if content.count(old_init) != 1:
        print(f"ERROR: Expected exactly 1 occurrence of __init__ signature, found {content.count(old_init)}")
        return False

    content = content.replace(old_init, new_init)
    print("[1/5] Added use_leaky_relu=False to SurvivorQualityNet.__init__() signature")

    # ================================================================
    # PATCH 2: Update docstring to document new param
    # ================================================================
    old_docstring = '''        """
        Initialize network.
        
        Args:
            input_size: Number of input features (dynamic, NOT hardcoded)
            hidden_layers: List of hidden layer sizes e.g. [256, 128, 64]
            dropout: Dropout probability
        """'''

    new_docstring = '''        """
        Initialize network.
        
        Args:
            input_size: Number of input features (dynamic, NOT hardcoded)
            hidden_layers: List of hidden layer sizes e.g. [256, 128, 64]
            dropout: Dropout probability
            use_leaky_relu: If True, use LeakyReLU(0.01) instead of ReLU (Category B)
        """'''

    if old_docstring not in content:
        print("WARNING: Could not find exact docstring, skipping docstring update")
    else:
        content = content.replace(old_docstring, new_docstring)
    print("[2/5] Updated docstring")

    # ================================================================
    # PATCH 3: Store use_leaky_relu as instance attribute
    # ================================================================
    old_attrs = """        self.input_size = input_size
        self.hidden_layers = hidden_layers"""

    new_attrs = """        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.use_leaky_relu = use_leaky_relu"""

    if old_attrs not in content:
        print("ERROR: Cannot find self.input_size / self.hidden_layers block")
        return False

    content = content.replace(old_attrs, new_attrs)
    print("[3/5] Added self.use_leaky_relu instance attribute")

    # ================================================================
    # PATCH 4: Replace nn.ReLU() with conditional activation
    # ================================================================
    old_relu = "            layers.append(nn.ReLU())"
    new_relu = "            layers.append(nn.LeakyReLU(0.01) if self.use_leaky_relu else nn.ReLU())"

    if content.count(old_relu) != 1:
        print(f"ERROR: Expected exactly 1 nn.ReLU() in layer loop, found {content.count(old_relu)}")
        return False

    content = content.replace(old_relu, new_relu)
    print("[4/5] Replaced nn.ReLU() with conditional LeakyReLU/ReLU in layer loop")

    # ================================================================
    # PATCH 5: Update NeuralNetWrapper.load() to pass use_leaky_relu
    # ================================================================
    old_load_build = """        # Build model with correct architecture
        wrapper.model = SurvivorQualityNet(
            input_size=feature_count,
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(wrapper.device)"""

    new_load_build = """        # Build model with correct architecture
        # Category B: restore activation mode from checkpoint
        use_leaky_relu = checkpoint.get('use_leaky_relu', False)
        wrapper.model = SurvivorQualityNet(
            input_size=feature_count,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_leaky_relu=use_leaky_relu,
        ).to(wrapper.device)"""

    if old_load_build not in content:
        print("ERROR: Cannot find NeuralNetWrapper.load() model build block")
        return False

    content = content.replace(old_load_build, new_load_build)
    print("[5/5] Updated NeuralNetWrapper.load() to restore use_leaky_relu from checkpoint")

    # ================================================================
    # Write patched file
    # ================================================================
    TARGET.write_text(content)
    print(f"\nAll 5 patches applied to {TARGET}")
    return True


def main():
    print("=" * 60)
    print("Category B Phase 1B: neural_net_wrapper.py")
    print("=" * 60)

    if not verify_preconditions():
        sys.exit(1)

    # Create backup
    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    if not apply_patch():
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to patch failure")
        sys.exit(1)

    # Syntax check
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print(f"Syntax check PASSED")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to syntax error")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Phase 1B COMPLETE")
    print("=" * 60)
    print(f"  File: {TARGET}")
    print(f"  Backup: {BACKUP}")
    print(f"  SurvivorQualityNet: +use_leaky_relu=False (default)")
    print(f"  When True: nn.ReLU() -> nn.LeakyReLU(0.01) in layer loop")
    print(f"  NeuralNetWrapper.load(): restores use_leaky_relu from checkpoint")
    print(f"  BatchNorm1d: UNCHANGED (always-on)")


if __name__ == "__main__":
    main()
