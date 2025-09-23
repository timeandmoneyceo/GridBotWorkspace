#!/usr/bin/env python3
"""
Iteration 3: Comprehensive Stress Test

This stress test covers all error types to establish our new baseline performance
after the successful Iteration 2 improvements.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Since we're already in the automated_debugging_strategy directory, use relative import
from automated_file_editor import SafeFileEditor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveStressTest:
    """Comprehensive stress test for all syntax recovery capabilities"""
    
    def __init__(self):
        self.editor = SafeFileEditor(use_serena=False)
        self.results = {}
        self.total_tests = 0
        self.total_success = 0
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive stress test across all error categories"""
        print("[ITERATION 3] COMPREHENSIVE STRESS TEST")
        print("=" * 60)
        
        # Test categories with increasingly difficult cases
        test_categories = [
            ("Mixed Indentation", self._test_mixed_indentation_stress),
            ("Bracket Errors", self._test_bracket_errors_stress),
            ("String Errors", self._test_string_errors_stress),
            ("Colon Errors", self._test_colon_errors_stress),
            ("Multi-Error Cases", self._test_multi_error_stress),
            ("Extreme Edge Cases", self._test_extreme_edge_cases)
        ]
        
        category_results = {}
        
        for category_name, test_function in test_categories:
            print(f"\\n[CATEGORY] {category_name}")
            print("-" * 40)
            
            start_time = time.time()
            category_result = test_function()
            end_time = time.time()
            
            category_result["execution_time"] = end_time - start_time
            category_results[category_name] = category_result
            
            success_rate = (category_result["success_count"] / category_result["total_tests"]) * 100
            print(f"[RESULT] {category_result['success_count']}/{category_result['total_tests']} ({success_rate:.1f}%) in {category_result['execution_time']:.2f}s")
            
            self.total_tests += category_result["total_tests"]
            self.total_success += category_result["success_count"]
        
        # Calculate overall performance
        overall_success_rate = (self.total_success / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        results = {
            "overall_success_rate": overall_success_rate,
            "total_success": self.total_success,
            "total_tests": self.total_tests,
            "category_results": category_results,
            "timestamp": time.time()
        }
        
        self._print_summary(results)
        return results
    
    def _test_mixed_indentation_stress(self) -> Dict[str, Any]:
        """Stress test mixed indentation with various complexity levels"""
        test_cases = [
            ("Simple tab/space mix", """def test():
\\t    if True:
        print("test")
\\treturn False"""),
            
            ("Complex nested mix", """class DataProcessor:
\\tdef __init__(self):
  \\t  self.data = []
\\t\\t    \\t  
\\tdef process(self):
      \\t  for item in self.data:
\\t  \\t    if item:
    \\t\\t      yield item"""),
            
            ("Extreme complexity", """def complex_function():
\\t\\t  if condition:
    \\t    \\t  for item in items:
\\t  \\t        if item.valid:
      \\t\\t          process(item)
\\t    \\t\\t    return True"""),
            
            ("Mixed with comments", """def process():  # Comment
\\t    data = []  # Tab start
    \\tfor item in items:  # Mixed  
\\t        data.append(item)  # Tab again
    return data  # Spaces only"""),
            
            ("Deep nesting mix", """class Complex:
\\tdef method1(self):
  \\t  if True:
\\t\\t    \\t  for i in range(10):
    \\t\\t      if i % 2:
\\t  \\t\\t        for j in range(5):
      \\t\\t\\t      print(f"{i},{j}")""")
        ]
        
        return self._run_test_category(test_cases, self.editor._mixed_indent_fix, "unindent does not match any outer indentation level")
    
    def _test_bracket_errors_stress(self) -> Dict[str, Any]:
        """Stress test bracket error handling"""
        test_cases = [
            ("Missing paren", """def test():
    print("hello"
    return True"""),
            
            ("Missing bracket", """def test():
    items = [1, 2, 3
    return items"""),
            
            ("Missing brace", """def test():
    data = {"key": "value"
    return data"""),
            
            ("Multiple missing", """def complex():
    result = func(arg1, [1, 2, 3, {"key": "value"
    return result"""),
            
            ("Nested complex", """def process():
    data = {
        "items": [1, 2, {"nested": [4, 5
        "config": {"mode": "test"
    }
    return data"""),
            
            ("Function call chain", """def chain():
    result = obj.method1().method2([
        item for item in data if validate(item
    ]).process()
    return result""")
        ]
        
        return self._run_test_category(test_cases, self.editor._bracket_specialist_fix, "EOF while scanning triple-quoted string literal")
    
    def _test_string_errors_stress(self) -> Dict[str, Any]:
        """Stress test string error handling"""
        test_cases = [
            ("Unterminated double", """def test():
    message = "Hello world
    return message"""),
            
            ("Unterminated single", """def test():
    name = 'Alice
    return name"""),
            
            ("Mixed quotes", """def test():
    text = "She said 'Hello
    return text"""),
            
            ("Complex string", """def test():
    query = "SELECT * FROM table WHERE name = 'John
    return query"""),
            
            ("Multi-line attempt", """def test():
    text = "This is a very long
    string that spans multiple lines
    return text"""),
            
            ("Escaped quotes issue", """def test():
    path = "C:\\Users\\Name\\file.txt
    return path""")
        ]
        
        return self._run_test_category(test_cases, self.editor._string_specialist_fix, "EOL while scanning string literal")
    
    def _test_colon_errors_stress(self) -> Dict[str, Any]:
        """Stress test missing colon handling"""
        test_cases = [
            ("Missing if colon", """def test():
    if condition
        return True"""),
            
            ("Missing def colon", """def test()
    return True"""),
            
            ("Missing class colon", """class Test
    def method(self):
        pass"""),
            
            ("Missing for colon", """def test():
    for item in items
        print(item)"""),
            
            ("Missing while colon", """def test():
    while True
        break"""),
            
            ("Missing try colon", """def test():
    try
        risky_operation()
    except Exception:
        pass""")
        ]
        
        return self._run_test_category(test_cases, self.editor._colon_specialist_fix, "invalid syntax")
    
    def _test_multi_error_stress(self) -> Dict[str, Any]:
        """Stress test multiple simultaneous errors"""
        test_cases = [
            ("Indent + bracket", """def test():
\\t    if True:
        items = [1, 2, 3
\\treturn items"""),
            
            ("String + colon", """def test()
    message = "Hello world
    return message"""),
            
            ("All three errors", """def broken()
\\t    if True  # Missing colon
        print("test"  # Missing closing paren
\\treturn False"""),
            
            ("Nested chaos", """class Test:
\\tdef method(self):
  \\t  items = [1, 2, 3
\\t\\t      for item in items
      \\t      print(f"Item: {item"
\\t  return items"""),
            
            ("Maximum complexity", """def chaos():
\\t\\t  message = "Hello world
    \\tif x > 0  # Missing colon and closing quote
\\t  \\t    data = [1, 2, {"key": "value"
      \\t\\t      print(f"Data: {data"
\\treturn message""")
        ]
        
        # Use comprehensive syntax fix for multi-error cases
        def comprehensive_fix(content, error):
            result = self.editor._comprehensive_syntax_fix(content, error, "test.py")
            return result[0] if result[1] else content
        
        return self._run_test_category(test_cases, comprehensive_fix, "Multiple syntax errors")
    
    def _test_extreme_edge_cases(self) -> Dict[str, Any]:
        """Stress test with extreme edge cases"""
        test_cases = [
            ("Empty function body", """def test():
\\t\\t  if True:
    \\t    \\t  
\\treturn"""),
            
            ("Only whitespace errors", """def test():
\\t  \\t  \\t
    \\t\\t  \\t  
\\t    \\t"""),
            
            ("Unicode in mixed indent", """def test():
\\t    # Comment with unicode: café
        print("unicode: ñoño")
\\treturn True"""),
            
            ("Very deep nesting", """def deep():
\\tif a:
  \\t  if b:
\\t\\t    if c:
    \\t\\t  if d:
\\t  \\t\\t    if e:
      \\t\\t\\t  return "deep\""""),
            
            ("Everything wrong", """class Broken
\\tdef method(self)
  \\t  data = {"incomplete": [1, 2
\\t\\t      message = "unclosed string
    \\t\\t  if True
      \\t\\t      print(f"chaos: {data"
\\t  return data""")
        ]
        
        # Use comprehensive syntax fix for extreme cases
        def comprehensive_fix(content, error):
            result = self.editor._comprehensive_syntax_fix(content, error, "test.py")
            return result[0] if result[1] else content
        
        return self._run_test_category(test_cases, comprehensive_fix, "Extreme syntax errors")
    
    def _run_test_category(self, test_cases: List[Tuple[str, str]], fix_function, default_error: str) -> Dict[str, Any]:
        """Run a category of tests with timing and validation"""
        success_count = 0
        results = []
        
        for test_name, code in test_cases:
            try:
                start_time = time.time()
                result = fix_function(code, default_error)
                end_time = time.time()
                
                # Validate result
                compile(result, '<string>', 'exec')
                
                success_count += 1
                results.append({
                    "name": test_name,
                    "success": True,
                    "execution_time": end_time - start_time,
                    "error": None
                })
                print(f"  [SUCCESS] {test_name}")
                
            except Exception as e:
                results.append({
                    "name": test_name,
                    "success": False,
                    "execution_time": 0,
                    "error": str(e)
                })
                print(f"  [FAILED] {test_name}: {e}")
        
        return {
            "success_count": success_count,
            "total_tests": len(test_cases),
            "results": results
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        print(f"\\n[ITERATION 3 COMPREHENSIVE RESULTS]")
        print("=" * 60)
        print(f"Overall Success Rate: {results['total_success']}/{results['total_tests']} ({results['overall_success_rate']:.1f}%)")

        print(f"\\nCategory Breakdown:")
        for category, data in results["category_results"].items():
            success_rate = (data["success_count"] / data["total_tests"]) * 100
            print(f"  {category}: {data['success_count']}/{data['total_tests']} ({success_rate:.1f}%) - {data['execution_time']:.2f}s")

        # Compare to previous iterations
        print(f"\\nIteration Comparison:")
        print("  Iteration 1: 75% (3/4 basic tests)")
        print("  Iteration 2: Fixed complex case + 3.3x speed")
        print(f"  Iteration 3: {results['overall_success_rate']:.1f}% ({results['total_success']}/{results['total_tests']} comprehensive tests)")

        improvement_from_baseline = results['overall_success_rate'] - 75.0
        print(f"\\nImprovement from Iteration 1: {improvement_from_baseline:+.1f}%")

        if results['overall_success_rate'] >= 80.0:
            print("[ACHIEVEMENT] Exceeded 80% success rate on comprehensive tests!")
        if results['overall_success_rate'] >= 85.0:
            print("[EXCEPTIONAL] Achieved 85%+ success rate - excellent performance!")


def main():
    """Run the comprehensive stress test"""
    stress_test = ComprehensiveStressTest()
    results = stress_test.run_all_tests()
    
    # Save results for tracking
    import json
    with open("iteration_3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n[SAVED] Results saved to iteration_3_results.json")
    return results


if __name__ == "__main__":
    main()