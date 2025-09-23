#!/usr/bin/env python3
"""
Quick Test for Iteration 2 Improvements

Test the multi-pass indentation fix and caching optimizations.
"""

import sys
import time
from pathlib import Path

# Since we're already in the automated_debugging_strategy directory, use relative import
try:
    from automated_file_editor import SafeFileEditor
except ImportError:
    print("Error: Could not import SafeFileEditor")
    sys.exit(1)


def test_iteration_2_improvements():
    """Test multi-pass and caching improvements"""
    print("[ITERATION 2] Testing Multi-Pass & Caching Optimizations")
    print("=" * 60)

    editor = SafeFileEditor(use_serena=False)

    # Focus on the complex indentation case that failed in Iteration 1
    complex_case = {
        "name": "Complex Mixed Indentation (Multi-Pass)",
        "code": """def complex_function():
\t\t  if condition:
    \t    \t  for item in items:
\t  \t        if item.valid:
      \t\t          process(item)
\t    \t\t    return True""",
        "error": "unindent does not match any outer indentation level"
    }

    print(f"[TEST] {complex_case['name']}")

    try:
        start_time = time.time()

        # This should now use the multi-pass approach
        result = editor._mixed_indent_fix(complex_case['code'], complex_case['error'])

        end_time = time.time()
        execution_time = end_time - start_time

        # Validate the result
        compile(result, '<string>', 'exec')

        print(f"  [SUCCESS] Fixed in {execution_time:.3f}s")
        print("  [RESULT]")
        for i, line in enumerate(result.split('\n'), 1):
            print(f"    {i:2d}: '{line}'")

        # Test caching by running the same case again
        print(f"\n[CACHE TEST] Running same case again...")
        start_time_cached = time.time()
        result_cached = editor._mixed_indent_fix(complex_case['code'], complex_case['error'])
        end_time_cached = time.time()
        cached_time = end_time_cached - start_time_cached

        print(f"  [CACHED] Fixed in {cached_time:.3f}s")
        speedup = execution_time / cached_time if cached_time > 0 else float('inf')
        print(f"  [SPEEDUP] {speedup:.1f}x faster with cache")

        return {
            "success": True,
            "first_run_time": execution_time,
            "cached_run_time": cached_time,
            "speedup": speedup
        }

    except Exception as e:
        print(f"  [FAILED] {e}")
        return {
            "success": False,
            "error": str(e)
        }


def test_simple_cases_speed():
    """Test execution speed on simpler cases"""
    print(f"\n[SPEED TEST] Testing simpler cases for baseline speed")
    
    editor = SafeFileEditor(use_serena=False)
    
    simple_cases = [
        ("Simple tabs", """def test():
\tprint("hello")
\treturn True"""),
        ("Simple mixed", """def test():
\t  if True:
  \t    print("test")""")
    ]
    
    total_time = 0
    success_count = 0
    
    for name, code in simple_cases:
        try:
            start_time = time.time()
            result = editor._mixed_indent_fix(code, "mixed indentation")
            end_time = time.time()
            
            compile(result, '<string>', 'exec')
            
            execution_time = end_time - start_time
            total_time += execution_time
            success_count += 1
            
            print(f"  {name}: {execution_time:.3f}s")
            
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"  [AVERAGE] {avg_time:.3f}s per simple case")
        return avg_time
    else:
        return 0


if __name__ == "__main__":
    iteration_2_result = test_iteration_2_improvements()
    simple_speed = test_simple_cases_speed()

    print(f"\n[ITERATION 2 SUMMARY]")
    if iteration_2_result["success"]:
        print("Complex case: PASSED")
        print(f"Speed improvement from caching: {iteration_2_result['speedup']:.1f}x")
    else:
        print("Complex case: FAILED")

    print(f"Simple cases average speed: {simple_speed:.3f}s")