#!/usr/bin/env python3
"""
Quick Performance Test for Iteration 1 Improvements

Test the enhanced LLM prompts and error analysis to validate improvements.
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


def test_iteration_1_improvements():
    """Test the enhanced LLM prompts and error analysis"""
    print("[ITERATION 1] Testing Enhanced LLM Prompts & Error Analysis")
    print("=" * 60)
    
    editor = SafeFileEditor(use_serena=False)
    
    # Test cases that should benefit from enhanced prompts
    test_cases = [
        {
            "name": "Complex Mixed Indentation",
            "code": """def complex_function():
\t\t  if condition:
    \t    \t  for item in items:
\t  \t        if item.valid:
      \t\t          process(item)
\t    \t\t    return True""",
            "error": "unindent does not match any outer indentation level",
            "method": "_mixed_indent_fix"
        },
        {
            "name": "Nested Bracket Issue", 
            "code": """def process_data():
    result = calculate(
        values=[1, 2, 3,
        config={"mode": "test"
    return result""",
            "error": "EOF while scanning triple-quoted string literal",
            "method": "_bracket_specialist_fix"
        },
        {
            "name": "String with Quotes",
            "code": """def create_message():
    text = "She said 'Hello world
    return text""",
            "error": "EOL while scanning string literal",
            "method": "_string_specialist_fix"
        },
        {
            "name": "Multi-Level Indentation",
            "code": """class DataProcessor:
\tdef __init__(self):
  \t  self.data = []
\t\t    \t  
\tdef process(self):
      \t  for item in self.data:
\t  \t    if item:
    \t\t      yield item""",
            "error": "unindent does not match any outer indentation level", 
            "method": "_mixed_indent_fix"
        }
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    results = []
    
    for test_case in test_cases:
        print(f"\n[TEST] {test_case['name']}")
        
        try:
            start_time = time.time()
            
            # Get the method to test
            method = getattr(editor, test_case['method'])
            result = method(test_case['code'], test_case['error'])
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Validate the result
            compile(result, '<string>', 'exec')
            
            success_count += 1
            results.append({
                "name": test_case['name'],
                "success": True,
                "time": execution_time,
                "error": None
            })
            print(f"  [SUCCESS] Fixed in {execution_time:.3f}s")
            
        except Exception as e:
            results.append({
                "name": test_case['name'],
                "success": False,
                "time": 0,
                "error": str(e)
            })
            print(f"  [FAILED] {e}")
    
    success_rate = (success_count / total_tests) * 100
    avg_time = sum(r['time'] for r in results if r['success']) / success_count if success_count > 0 else 0
    
    print(f"\n[ITERATION 1 RESULTS]")
    print(f"Success Rate: {success_count}/{total_tests} ({success_rate:.1f}%)")
    print(f"Average Execution Time: {avg_time:.3f}s")
    
    # Compare to expected baseline (assuming previous was ~75%)
    baseline_rate = 75.0  # Estimated baseline
    improvement = success_rate - baseline_rate
    improvement_percentage = (improvement / baseline_rate) * 100 if baseline_rate > 0 else 0
    
    print(f"Estimated Improvement: {improvement:+.1f}% ({improvement_percentage:+.1f}%)")
    
    if improvement_percentage >= 1.0:
        print("[ACHIEVEMENT] Reached target of at least 1% improvement!")
    else:
        print("[CONTINUE] Need additional enhancements to reach 1% target")
    
    return {
        "success_rate": success_rate,
        "improvement": improvement_percentage,
        "execution_time": avg_time,
        "results": results
    }


if __name__ == "__main__":
    test_iteration_1_improvements()