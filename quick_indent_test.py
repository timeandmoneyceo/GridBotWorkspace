#!/usr/bin/env python3
"""
Quick test for the robust indentation fix
"""

import sys
import os

# Since we're already in the automated_debugging_strategy directory, use direct import
from automated_file_editor import SafeFileEditor

def test_indentation_fix():
    editor = SafeFileEditor()
    
    # Test case that was failing
    broken_code = """def test():
\t    if True:
        print("test")
\treturn False"""
    
    print("Testing robust indentation fix:")
    print("Original broken code:")
    print(repr(broken_code))
    
    # Try our new fix
    fixed = editor._mixed_indent_fix(broken_code, "indentation error")
    
    print("\nFixed code:")
    print(repr(fixed))
    print("\nFixed code (readable):")
    print(fixed)
    
    # Test if it compiles
    try:
        compile(fixed, '<string>', 'exec')
        print("\n✅ SUCCESS: Fixed code compiles!")
        return True
    except SyntaxError as e:
        print(f"\n❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    test_indentation_fix()