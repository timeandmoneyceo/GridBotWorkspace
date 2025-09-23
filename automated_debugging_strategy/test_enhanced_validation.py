#!/usr/bin/env python3
"""
Test script for enhanced AI code validation and correction
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_enhanced_validation():
    """Test the enhanced validation functions"""
    print("Testing Enhanced AI Code Validation and Correction...")

    # Import the enhanced optimization system
    try:
        from enhanced_optimization_system import EnhancedOptimizationSystem
        print("✓ Enhanced optimization system imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced optimization system: {e}")
        return False

    # Create a mock instance to test validation methods
    class MockLLM:
        pass

    class MockFileEditor:
        pass

    try:
        # Create instance with mocks
        system = EnhancedOptimizationSystem.__new__(EnhancedOptimizationSystem)
        system.logger = type('MockLogger', (), {
            'info': lambda *args, **kwargs: print(f"INFO: {args[0] if args else ''}"),
            'warning': lambda *args, **kwargs: print(f"WARN: {args[0] if args else ''}"),
            'error': lambda *args, **kwargs: print(f"ERROR: {args[0] if args else ''}")
        })()

        print("✓ Enhanced optimization system instance created")
    except Exception as e:
        print(f"✗ Failed to create system instance: {e}")
        return False

    # Test validation functions
    test_cases = [
        # Valid code
        ("def test(): return 1", "Valid Python code"),

        # Code with unterminated triple quotes
        ('"""This is unterminated', "Unterminated triple double quotes"),

        # Code with unbalanced braces in f-string
        ('f"Value: {variable"', "Unbalanced braces in f-string"),

        # Code with unclosed parentheses
        ("def test(arg", "Unclosed parentheses"),

        # Code with syntax error
        ("def test(: return 1", "Invalid syntax"),
    ]

    print("\nTesting validation functions:")
    for code, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Code: {code}")

        # Test structure integrity check
        issues = system._check_code_structure_integrity(code)
        if issues:
            print(f"✓ Detected issues: {issues}")
        else:
            print("✗ No issues detected (might be expected for valid code)")

        # Test validation
        result = system._validate_optimized_code(code, "test.py")
        if result['valid']:
            print("✓ Code passed validation")
        else:
            print(f"✓ Code failed validation: {result['error']}")

    # Test correction functions
    print("\nTesting correction functions:")

    problematic_code = 'f"Value: {variable"'
    print(f"Original problematic code: {problematic_code}")

    corrected = system._correct_ai_generated_code(problematic_code)
    print(f"Corrected code: {corrected}")

    if corrected != problematic_code:
        print("✓ Code was corrected")
    else:
        print("✗ Code was not corrected")

    # Test aggressive correction
    print("\nTesting aggressive correction:")
    aggressive_corrected = system._aggressive_code_correction(problematic_code)
    print(f"Aggressively corrected: {aggressive_corrected}")

    print("\nEnhanced validation and correction test completed!")
    return True

if __name__ == "__main__":
    test_enhanced_validation()