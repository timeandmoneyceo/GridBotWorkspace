#!/usr/bin/env python3
"""Test script to check imports"""

try:
    from automated_debugging_strategy.automated_file_editor import SafeFileEditor
    print("SafeFileEditor import: SUCCESS")
except ImportError as e:
    print(f"SafeFileEditor import: FAILED - {e}")

try:
    from automated_debugging_strategy.enhanced_optimization_system import EnhancedOptimizationSystem
    print("EnhancedOptimizationSystem import: SUCCESS")
except ImportError as e:
    print(f"EnhancedOptimizationSystem import: FAILED - {e}")

try:
    eos = EnhancedOptimizationSystem()
    priority = eos._calculate_priority_from_target({"priority": "high"})
    print(f"_calculate_priority_from_target test: SUCCESS - returned {priority}")
except Exception as e:
    print(f"_calculate_priority_from_target test: FAILED - {e}")