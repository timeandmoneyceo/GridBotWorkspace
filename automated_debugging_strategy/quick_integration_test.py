#!/usr/bin/env python3
"""
Quick integration test for the robust indentation fix
"""

import sys
import os

# Since we're already in the automated_debugging_strategy directory, use direct import
from automated_file_editor import SafeFileEditor

def test_integration():
    editor = SafeFileEditor()
    
    # Test case that was failing in Iteration 5
    broken_code = """def test():
\t    if True:
        print("ok")
\treturn"""
    
    print("Testing integrated robust indentation fix:")
    print("Original broken code:")
    print(repr(broken_code))
    
    # Use the comprehensive fix engine (this should now call our robust fix)
    result = editor.comprehensive_syntax_fix(
        content=broken_code,
        file_path="<test>",
        syntax_error="mixed_indentation error",
        enable_all_strategies=True
    )
    
    print(f"\nResult success: {result.success}")
    if result.success and result.fixed_content:
        print("Fixed code:")
        print(repr(result.fixed_content))
        print("\nFixed code (readable):")
        print(result.fixed_content)
        
        # Test if it compiles
        try:
            compile(result.fixed_content, '<string>', 'exec')
            print("\n✅ SUCCESS: Fixed code compiles!")
            return True
        except SyntaxError as e:
            print(f"\n❌ COMPILATION FAILED: {e}")
            return False
    else:
        print(f"\n❌ FIX FAILED: {result.error}")
        return False

if __name__ == "__main__":
    test_integration()