#!/usr/bin/env python3
"""Test script to check SafeFileEditor import"""

try:
    from automated_file_editor import SafeFileEditor
    print("✓ SafeFileEditor import successful")
except Exception as e:
    print(f"✗ SafeFileEditor import failed: {e}")
    import traceback
    traceback.print_exc()