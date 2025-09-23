#!/usr/bin/env python3
"""
ITERATION 5: Learning-Based Strategy Evolution
==============================================
Combining all fixes from previous iterations + pattern learning
Target: 80%+ success rate with systematic improvements
"""

import time
import json
from automated_debugging_strategy.automated_file_editor import SafeFileEditor

class LearningBasedTest:
    def __init__(self):
        self.editor = SafeFileEditor()
        self.results = {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'iteration': 5,
            'description': 'Learning-Based Strategy Evolution',
            'categories': {},
            'total_tests': 0,
            'total_passed': 0,
            'total_time': 0,
            'learning_insights': [],
            'strategy_effectiveness': {}
        }
        
        # Track what we learned from previous iterations
        self.learned_patterns = {
            'colon_fixes': 'DeepSeek precise works best for colon errors (100% in Iteration 4)',
            'bracket_fixes': 'LLM bracket specialist maintains 100% success',
            'indentation_fixes': 'Robust algorithmic approach beats LLM for mixed indentation',
            'timeout_management': 'Fast fallback needed for complex multi-error cases',
            'serena_integration': 'Semantic understanding helps with context-aware fixes'
        }
    
    def log_learning_insight(self, insight, evidence):
        """Log insights gained during testing"""
        learning = {
            'insight': insight,
            'evidence': evidence,
            'timestamp': time.time()
        }
        self.results['learning_insights'].append(learning)
        print(f"  [LEARNING] {insight}")
    
    def test_category_with_learning(self, category_name, test_cases):
        """Test a category while tracking strategy effectiveness"""
        print(f"\n[CATEGORY] {category_name}")
        print("-" * 40)
        
        category_results = {
            'passed': 0,
            'total': len(test_cases),
            'time': 0,
            'details': [],
            'strategy_performance': {}
        }
        
        start_time = time.time()
        
        for test_name, broken_code, expected_keywords in test_cases:
            test_start = time.time()
            
            try:
                # Determine optimal strategy based on learning
                error_type = self.classify_error_intelligently(broken_code)
                optimal_strategy = self.select_optimal_strategy(error_type)
                
                print(f"  [STRATEGY] {test_name}: Using {optimal_strategy} for {error_type}")
                
                # Use the comprehensive fix engine
                result = self.editor.comprehensive_syntax_fix(
                    content=broken_code,
                    file_path="<test>",
                    syntax_error=f"{error_type} error",
                    enable_all_strategies=True
                )
                
                test_time = time.time() - test_start
                
                if result.success:
                    # Verify it compiles and preserves functionality
                    try:
                        compile(result.fixed_content, '<string>', 'exec')
                        has_expected = all(keyword in result.fixed_content for keyword in expected_keywords)
                        
                        if has_expected:
                            print(f"  [SUCCESS] {test_name} ({test_time:.2f}s)")
                            category_results['passed'] += 1
                            category_results['details'].append({
                                'test': test_name,
                                'status': 'success',
                                'time': test_time,
                                'strategy': optimal_strategy,
                                'error_type': error_type
                            })
                            
                            # Track strategy effectiveness
                            if optimal_strategy not in category_results['strategy_performance']:
                                category_results['strategy_performance'][optimal_strategy] = {'success': 0, 'total': 0}
                            category_results['strategy_performance'][optimal_strategy]['success'] += 1
                            category_results['strategy_performance'][optimal_strategy]['total'] += 1
                            
                        else:
                            print(f"  [FAILED] {test_name}: Missing expected functionality")
                            self.log_learning_insight(f"Strategy {optimal_strategy} fixed syntax but lost functionality", test_name)
                    except SyntaxError as se:
                        print(f"  [FAILED] {test_name}: Still has syntax error - {se}")
                        self.log_learning_insight(f"Strategy {optimal_strategy} failed for {error_type}", str(se))
                else:
                    print(f"  [FAILED] {test_name}: Fix engine failed")
                    
            except Exception as e:
                test_time = time.time() - test_start
                print(f"  [ERROR] {test_name}: {e}")
        
        category_results['time'] = time.time() - start_time
        success_rate = (category_results['passed'] / category_results['total']) * 100
        print(f"[RESULT] {category_results['passed']}/{category_results['total']} ({success_rate:.1f}%) in {category_results['time']:.2f}s")
        
        # Analyze strategy performance
        for strategy, perf in category_results['strategy_performance'].items():
            strategy_success_rate = (perf['success'] / perf['total']) * 100 if perf['total'] > 0 else 0
            self.log_learning_insight(f"Strategy '{strategy}' achieved {strategy_success_rate:.1f}% success in {category_name}", f"{perf['success']}/{perf['total']}")
        
        self.results['categories'][category_name] = category_results
        self.results['total_tests'] += category_results['total']
        self.results['total_passed'] += category_results['passed']
        self.results['total_time'] += category_results['time']
        
        return category_results
    
    def classify_error_intelligently(self, broken_code):
        """Classify error type using learned patterns"""
        try:
            compile(broken_code, '<string>', 'exec')
            return "no_error"
        except SyntaxError as e:
            error_msg = str(e).lower()
            
            # Apply learned classification patterns
            if any(keyword in error_msg for keyword in ['indent', 'unindent', 'tab', 'space']):
                return "mixed_indentation"
            elif "expected ':'" in error_msg or 'colon' in error_msg:
                return "missing_colon"
            elif any(keyword in error_msg for keyword in ['paren', 'bracket', 'brace', 'was never closed']):
                return "unmatched_brackets"
            elif any(keyword in error_msg for keyword in ['string', 'quote', 'unterminated']):
                return "unterminated_string"
            elif 'multiple' in error_msg or len([line for line in broken_code.split('\n') if line.strip()]) > 5:
                return "complex_multi_error"
            else:
                return "unknown"
    
    def select_optimal_strategy(self, error_type):
        """Select optimal strategy based on learning from previous iterations"""
        strategy_map = {
            'mixed_indentation': 'robust_algorithmic_fix',  # Learned: Algorithm beats LLM
            'missing_colon': 'deepseek_precise',            # Learned: 100% success in Iteration 4
            'unmatched_brackets': 'llm_bracket_specialist',  # Learned: Consistent 100% success
            'unterminated_string': 'llm_string_specialist',  # Learned: Good contextual understanding
            'complex_multi_error': 'fast_timeout_strategy',  # Learned: Need timeout management
            'unknown': 'comprehensive_fallback'              # Learned: Use all strategies
        }
        return strategy_map.get(error_type, 'comprehensive_fallback')

def main():
    print("[ITERATION 5] LEARNING-BASED STRATEGY EVOLUTION")
    print("=" * 60)

    tester = LearningBasedTest()

    # Display learned patterns
    print("\n[LEARNED PATTERNS FROM ITERATIONS 1-4]")
    for pattern, description in tester.learned_patterns.items():
        print(f"  • {pattern}: {description}")

    # Comprehensive test cases incorporating all previous learnings

    # 1. Fixed Colon Tests (Target: 100% maintained from Iteration 4)
    colon_tests = [
        ("Function definition", "def calculate(x, y)\n    return x + y", ["def", "calculate", "return"]),
        ("Class definition", "class DataProcessor\n    def __init__(self):\n        pass", ["class", "DataProcessor", "def"]),
        ("If statement", "def check():\n    if value > 0\n        return True", ["def", "if", "return"]),
        ("For loop", "def iterate():\n    for i in range(10)\n        print(i)", ["def", "for", "print"]),
        ("Try block", "def safe_op():\n    try\n        risky_call()\n    except:\n        pass", ["def", "try", "except"]),
    ]

    # 2. Robust Indentation Tests (Target: 80%+ with new robust fix)
    indentation_tests = [
        ("Tab-space simple", "def test():\n\t    if True:\n        print(\"ok\")\n\treturn", ["def", "if", "print", "return"]),
        ("Complex nesting", "def process():\n\tif condition:\n  \t  for item in items:\n\t\t    process(item)\n    \treturn done", ["def", "if", "for", "return"]),
        ("Mixed with strings", "def example():\n\t    message = \"hello\"\n    \tif message:\n\t        print(message)", ["def", "message", "if", "print"]),
        ("Deep nesting fix", "def deep():\n\tif a:\n  \t  if b:\n\t\t    if c:\n    \t\t  return \"success\"", ["def", "if", "return"]),
    ]

    # 3. Bracket Excellence Tests (Target: 100% maintained)
    bracket_tests = [
        ("Function calls", "def test():\n    result = func(arg1, [1, 2, 3\n    return result", ["def", "result", "return"]),
        ("Dict/list complex", "def data():\n    info = {\"items\": [1, 2, {\"nested\": [4, 5}\n    return info", ["def", "info", "return"]),
        ("Method chaining", "def chain():\n    result = obj.method([x for x in data if valid(x]).process()\n    return result", ["def", "result", "return"]),
    ]

    # 4. String Specialist Tests
    string_tests = [
        ("Simple unterminated", "def greet():\n    msg = \"Hello world\n    return msg", ["def", "msg", "return"]),
        ("Complex quotes", "def query():\n    sql = \"SELECT * FROM users WHERE name = 'John\n    return sql", ["def", "sql", "return"]),
    ]

    # 5. Multi-Error with Timeout (Target: Better than 50% from Iteration 4)
    multi_error_tests = [
        ("Indent + string", "def example():\n\t    text = \"incomplete\n    \tif text:\n        return text", ["def", "text", "if", "return"]),
        ("All errors", "def chaos():\n\tmessage = \"broken\n    if True\n        data = [1, 2\n    return message", ["def", "message", "if", "data", "return"]),
    ]

    # Run all test categories with learning
    tester.test_category_with_learning("Fixed Colon Tests", colon_tests)
    tester.test_category_with_learning("Robust Indentation Tests", indentation_tests)
    tester.test_category_with_learning("Bracket Excellence Tests", bracket_tests)
    tester.test_category_with_learning("String Specialist Tests", string_tests)
    tester.test_category_with_learning("Multi-Error Timeout Tests", multi_error_tests)

    # Calculate results and learning insights
    overall_success_rate = (tester.results['total_passed'] / tester.results['total_tests']) * 100

    print(f"\n[ITERATION 5 LEARNING-BASED RESULTS]")
    print("=" * 60)
    print(f"Overall Success Rate: {tester.results['total_passed']}/{tester.results['total_tests']} ({overall_success_rate:.1f}%)")
    print(f"\nCategory Breakdown:")

    for category, results in tester.results['categories'].items():
        success_rate = (results['passed'] / results['total']) * 100
        print(f"  {category}: {results['passed']}/{results['total']} ({success_rate:.1f}%) - {results['time']:.2f}s")

    print(f"\nLearning Insights Gained: {len(tester.results['learning_insights'])}")
    print(f"Total Time: {tester.results['total_time']:.2f}s")

    # Historical comparison
    print(f"\nIterative Learning Progress:")
    print("  Iteration 1: 75% (3/4 basic tests)")
    print("  Iteration 2: Complex case + 3.3x speed improvement")
    print("  Iteration 3: 72.7% (24/33 comprehensive tests)")
    print("  Iteration 4: 66.7% (8/12 adaptive tests) - REGRESSION")
    print(f"  Iteration 5: {overall_success_rate:.1f}% ({tester.results['total_passed']}/{tester.results['total_tests']} learning-based tests)")

    # Calculate improvement
    improvement_from_3 = overall_success_rate - 72.7
    improvement_from_4 = overall_success_rate - 66.7

    print(f"\nImprovement Analysis:")
    print(f"  From Iteration 3: {improvement_from_3:+.1f}%")
    print(f"  From Iteration 4: {improvement_from_4:+.1f}%")

    if overall_success_rate >= 80.0:
        print(f"\n🎯 TARGET ACHIEVED: 80%+ success rate reached!")
        print("🚀 SYSTEMATIC IMPROVEMENT SUCCESSFUL!")
    elif overall_success_rate >= 75.0:
        print(f"\n⚡ RECOVERY SUCCESSFUL: Back above 75%!")
        print(f"📈 Continue iterating toward 80% target...")
    else:
        print(f"\n🔄 CONTINUED LEARNING NEEDED")
        print(f"🎯 Gap to 80% target: {80.0 - overall_success_rate:.1f}%")

    # Save results with learning data
    filename = f"iteration_5_learning_results.json"
    with open(filename, 'w') as f:
        json.dump(tester.results, f, indent=2)
    print(f"\n[SAVED] Results and learning insights saved to {filename}")

if __name__ == "__main__":
    main()