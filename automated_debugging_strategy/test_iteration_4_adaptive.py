#!/usr/bin/env python3
"""
ITERATION 4: Adaptive Strategy Selection Test
==========================================
Testing intelligent strategy selection based on error patterns and performance tracking
Target: 80%+ success rate with improved efficiency
"""

import time
import json
from automated_debugging_strategy.automated_file_editor import SafeFileEditor

class AdaptiveStrategyTest:
    def __init__(self):
        self.editor = SafeFileEditor()
        self.results = {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'iteration': 4,
            'description': 'Adaptive Strategy Selection',
            'categories': {},
            'total_tests': 0,
            'total_passed': 0,
            'total_time': 0,
            'adaptive_decisions': []
        }
    
    def log_adaptive_decision(self, test_name, error_type, strategy_chosen, reason):
        """Log adaptive strategy selection decisions"""
        decision = {
            'test': test_name,
            'error_type': error_type,
            'strategy': strategy_chosen,
            'reason': reason,
            'timestamp': time.time()
        }
        self.results['adaptive_decisions'].append(decision)
        print(f"  [ADAPTIVE] {strategy_chosen} chosen for {error_type}: {reason}")
    
    def test_category(self, category_name, test_cases):
        """Test a category with adaptive strategy monitoring"""
        print(f"\n[CATEGORY] {category_name}")
        print("-" * 40)
        
        category_results = {
            'passed': 0,
            'total': len(test_cases),
            'time': 0,
            'details': []
        }
        
        start_time = time.time()
        
        for test_name, broken_code, expected_keywords in test_cases:
            test_start = time.time()
            
            try:
                # Create a temporary file to test
                temp_content = broken_code
                
                # Determine error type for adaptive selection
                try:
                    compile(broken_code, '<string>', 'exec')
                    error_type = "no_error"
                except SyntaxError as e:
                    error_msg = str(e).lower()
                    if 'indent' in error_msg or 'unindent' in error_msg:
                        error_type = "indentation"
                    elif 'colon' in error_msg or "expected ':'" in error_msg:
                        error_type = "colon"
                    elif 'paren' in error_msg or 'bracket' in error_msg or 'brace' in error_msg:
                        error_type = "bracket"
                    elif 'string' in error_msg or 'quote' in error_msg:
                        error_type = "string"
                    elif 'multiple' in error_msg:
                        error_type = "multi_error"
                    else:
                        error_type = "unknown"
                
                # Log which strategy would be chosen (for monitoring)
                if error_type == "indentation":
                    self.log_adaptive_decision(test_name, error_type, "enhanced_mixed_indent_fix", "Specialized indentation handling")
                elif error_type == "colon":
                    self.log_adaptive_decision(test_name, error_type, "fixed_colon_specialist", "Targeted colon placement")
                elif error_type == "bracket":
                    self.log_adaptive_decision(test_name, error_type, "llm_bracket_specialist", "Context-aware bracket matching")
                elif error_type == "string":
                    self.log_adaptive_decision(test_name, error_type, "llm_string_specialist", "Intelligent string termination")
                elif error_type == "multi_error":
                    self.log_adaptive_decision(test_name, error_type, "fast_timeout_deepseek", "Timeout-controlled multi-fix")
                else:
                    self.log_adaptive_decision(test_name, error_type, "comprehensive_multi_strategy", "Unknown error type")
                
                # Use the comprehensive fix engine
                result = self.editor.comprehensive_syntax_fix(
                    content=broken_code,
                    file_path="<test>",
                    syntax_error=f"{error_type} error",
                    enable_all_strategies=True
                )
                
                test_time = time.time() - test_start
                
                if result.success:
                    # Verify it compiles
                    try:
                        compile(result.fixed_content, '<string>', 'exec')
                        
                        # Check if expected functionality is preserved
                        has_expected = all(keyword in result.fixed_content for keyword in expected_keywords)
                        
                        if has_expected:
                            print(f"  [SUCCESS] {test_name}")
                            category_results['passed'] += 1
                            category_results['details'].append({
                                'test': test_name,
                                'status': 'success',
                                'time': test_time,
                                'error_type': error_type
                            })
                        else:
                            print(f"  [FAILED] {test_name}: Missing expected keywords")
                            category_results['details'].append({
                                'test': test_name,
                                'status': 'failed',
                                'reason': 'missing_keywords',
                                'time': test_time,
                                'error_type': error_type
                            })
                    except SyntaxError as se:
                        print(f"  [FAILED] {test_name}: {se}")
                        category_results['details'].append({
                            'test': test_name,
                            'status': 'failed',
                            'reason': str(se),
                            'time': test_time,
                            'error_type': error_type
                        })
                else:
                    print(f"  [FAILED] {test_name}: Fix failed")
                    category_results['details'].append({
                        'test': test_name,
                        'status': 'failed',
                        'reason': 'fix_failed',
                        'time': test_time,
                        'error_type': error_type
                    })
                    
            except Exception as e:
                test_time = time.time() - test_start
                print(f"  [ERROR] {test_name}: {e}")
                category_results['details'].append({
                    'test': test_name,
                    'status': 'error',
                    'reason': str(e),
                    'time': test_time,
                    'error_type': 'exception'
                })
        
        category_results['time'] = time.time() - start_time
        success_rate = (category_results['passed'] / category_results['total']) * 100
        print(f"[RESULT] {category_results['passed']}/{category_results['total']} ({success_rate:.1f}%) in {category_results['time']:.2f}s")
        
        self.results['categories'][category_name] = category_results
        self.results['total_tests'] += category_results['total']
        self.results['total_passed'] += category_results['passed']
        self.results['total_time'] += category_results['time']
        
        return category_results

def main():
    print("[ITERATION 4] ADAPTIVE STRATEGY SELECTION TEST")
    print("=" * 60)

    tester = AdaptiveStrategyTest()

    # Test cases focusing on previous failure patterns

    # 1. Enhanced Colon Error Tests (previously 0% success)
    colon_tests = [
        ("Simple if colon", "def test():\n    if condition\n        return True", ["def", "if", "return"]),
        ("Function def colon", "def test()\n    return True", ["def", "return"]),
        ("Class colon", "class Test\n    def method(self):\n        pass", ["class", "def", "pass"]),
        ("For loop colon", "def test():\n    for item in items\n        print(item)", ["def", "for", "print"]),
        ("Try block colon", "def test():\n    try\n        risky_operation()\n    except Exception:\n        pass", ["def", "try", "except"]),
    ]

    # 2. Complex Mixed Indentation (previously 60% success - target: 80%+)
    indentation_tests = [
        ("Tab-space mix simple", "def test():\n\t    if True:\n        print(\"test\")\n\treturn False", ["def", "if", "print", "return"]),
        ("Deep nesting mix", "def deep():\n\tif a:\n  \t  if b:\n\t\t    if c:\n    \t\t  return \"deep\"", ["def", "if", "return"]),
        ("Mixed with comments", "def test():\n\t    # Comment\n    \tdata = []\n\t        return data", ["def", "data", "return"]),
    ]

    # 3. Bracket Errors (previously 100% - maintain excellence)
    bracket_tests = [
        ("Function call chain", "def test():\n    result = obj.method([item for item in data if validate(item]).process()\n    return result", ["def", "result", "return"]),
        ("Nested complex", "def test():\n    data = {\"items\": [1, 2, {\"nested\": [4, 5}, \"config\": {\"mode\": \"test\"}\n    return data", ["def", "data", "return"]),
    ]

    # 4. Multi-Error with Timeout Control (previously 80% but very slow)
    multi_error_tests = [
        ("Fast timeout test", "def test():\n\t    message = \"Hello\"\n    if True\n        data = [1, 2\n        return message", ["def", "message", "if", "data", "return"]),
        ("Complex but manageable", "class Test:\n\tdef method(self):\n  \t  items = [1, 2, 3\n        return items", ["class", "def", "items", "return"]),
    ]

    # Run all test categories
    tester.test_category("Enhanced Colon Errors", colon_tests)
    tester.test_category("Complex Mixed Indentation", indentation_tests)
    tester.test_category("Bracket Errors", bracket_tests)
    tester.test_category("Multi-Error Timeout Control", multi_error_tests)

    # Calculate overall results
    overall_success_rate = (tester.results['total_passed'] / tester.results['total_tests']) * 100

    print(f"\n[ITERATION 4 ADAPTIVE RESULTS]")
    print("=" * 60)
    print(f"Overall Success Rate: {tester.results['total_passed']}/{tester.results['total_tests']} ({overall_success_rate:.1f}%)")
    print(f"\nCategory Breakdown:")

    for category, results in tester.results['categories'].items():
        success_rate = (results['passed'] / results['total']) * 100
        print(f"  {category}: {results['passed']}/{results['total']} ({success_rate:.1f}%) - {results['time']:.2f}s")

    print(f"\nAdaptive Strategy Decisions: {len(tester.results['adaptive_decisions'])}")
    print(f"Total Time: {tester.results['total_time']:.2f}s")

    # Compare with previous iterations
    print(f"\nIteration Comparison:")
    print("  Iteration 1: 75% (3/4 basic tests)")
    print("  Iteration 2: Fixed complex case + 3.3x speed")
    print("  Iteration 3: 72.7% (24/33 comprehensive tests)")
    print(f"  Iteration 4: {overall_success_rate:.1f}% ({tester.results['total_passed']}/{tester.results['total_tests']} adaptive tests)")

    improvement = overall_success_rate - 72.7
    print(f"\nImprovement from Iteration 3: {improvement:+.1f}%")

    if overall_success_rate >= 80.0:
        print("\n🎯 TARGET ACHIEVED: 80%+ success rate reached!")
    else:
        print(f"\n⚡ PROGRESS: {80.0 - overall_success_rate:.1f}% to reach 80% target")

    # Save detailed results
    filename = "iteration_4_adaptive_results.json"
    with open(filename, 'w') as f:
        json.dump(tester.results, f, indent=2)
    print(f"\n[SAVED] Results saved to {filename}")

if __name__ == "__main__":
    main()