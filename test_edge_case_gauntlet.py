#!/usr/bin/env python3
"""
Edge Case Syntax Recovery Test Suite

This test pushes our system beyond 100% by testing increasingly complex
edge cases and compound errors that would challenge any syntax recovery system.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Since we're already in the automated_debugging_strategy directory, use relative import
try:
    from automated_file_editor import SafeFileEditor, EditResult
    print("[SUCCESS] Successfully imported enhanced SafeFileEditor")
except ImportError as e:
    print(f" Failed to import SafeFileEditor: {e}")
    sys.exit(1)

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EdgeCaseTest:
    """Test suite for increasingly complex edge cases"""
    
    def __init__(self):
        self.editor = SafeFileEditor(
            validate_syntax=True,
            create_backups=True,
            use_serena=False  # Focus on our core strategies
        )
        self.test_cases = []
        self.results = []
        self.difficulty_levels = ["Basic", "Intermediate", "Advanced", "Expert", "Nightmare"]
        
    def create_edge_cases(self):
        """Create progressively harder edge cases"""
        
        # LEVEL 1: BASIC EDGE CASES
        self.test_cases.extend([
            {
                "name": "Nested Quotes Nightmare",
                "difficulty": "Basic",
                "broken_code": '''def quote_hell():
    msg = "She said \\"Hello, I can't believe it's \\"working\\"!\\"
    return msg
''',
                "expected_fix_count": 1
            },
            
            {
                "name": "Mixed Indentation Hell",
                "difficulty": "Basic", 
                "broken_code": """def mixed_indent():
\\tif True:
    \\tprint("tab then space")
\\t    print("space then tab")
\\treturn True
""",
                "expected_fix_count": 1
            }
        ])
        
        # LEVEL 2: INTERMEDIATE EDGE CASES
        self.test_cases.extend([
            {
                "name": "Compound Missing Elements",
                "difficulty": "Intermediate",
                "broken_code": '''def compound_error()
    data = [1, 2, 3
    if len(data) > 0
        for item in data
            print(f"Item: {item"
    return data
''',
                "expected_fix_count": 5  # Missing colon, bracket, colon, colon, bracket
            },
            
            {
                "name": "Deep Nesting Bracket Hell",
                "difficulty": "Intermediate",
                "broken_code": '''def deep_nesting():
    result = {
        "level1": [
            {"level2": {
                "level3": [
                    {"level4": "value"
                ]
            }
        ]
    return result
''',
                "expected_fix_count": 3  # Multiple missing brackets at different levels
            }
        ])
        
        # LEVEL 3: ADVANCED EDGE CASES
        self.test_cases.extend([
            {
                "name": "Unicode and Special Characters",
                "difficulty": "Advanced",
                "broken_code": '''def unicode_hell():
    msg = "Testing unicode:  \\u2705 \\U0001F4A5"
    special = f"Special chars: {msg
    return special
''',
                "expected_fix_count": 2
            },
            
            {
                "name": "Class Inheritance Compound Error",
                "difficulty": "Advanced",
                "broken_code": '''class Parent
    def __init__(self)
        self.value = 42
        
class Child(Parent
    def __init__(self, extra)
        super().__init__()
        self.extra = extra
        
    def method(self
        return self.value + self.extra
''',
                "expected_fix_count": 6  # Multiple missing colons and parentheses
            }
        ])
        
        # LEVEL 4: EXPERT EDGE CASES
        self.test_cases.extend([
            {
                "name": "Generator Expression Hell",
                "difficulty": "Expert",
                "broken_code": '''def generator_nightmare():
    data = [x**2 for x in range(10) if x % 2 == 0
    nested = [y for sublist in [z for z in data if z > 5 for y in [sublist * 2, sublist * 3]
    return nested
''',
                "expected_fix_count": 3  # Complex bracket matching in comprehensions
            },
            
            {
                "name": "Decorator and Context Manager Hell",
                "difficulty": "Expert",
                "broken_code": '''@property
def complex_decorator(func
    def wrapper(*args, **kwargs
        with open("test.txt", "w"
            return func(*args, **kwargs
    return wrapper

@complex_decorator
def test_function()
    return "test"
''',
                "expected_fix_count": 6  # Multiple missing parentheses and colons
            }
        ])
        
        # LEVEL 5: NIGHTMARE EDGE CASES
        self.test_cases.extend([
            {
                "name": "Everything Wrong At Once",
                "difficulty": "Nightmare",
                "broken_code": '''# The ultimate nightmare scenario
def nightmare_function(param1, param2, param3
data = {
    "list": [1, 2, 3
    "nested": {
        "deep": [
            {"ultra_deep": "value"
        ]
    },
    "string": "unclosed string
}

class NightmareClass
def __init__(self, data
self.data = data
    
def method(self
try
result = self.process_data(
except Exception as e
print(f"Error: {e"
return None

def process_data(self, item
if isinstance(item, dict
for key, value in item.items(
if key == "list"
for i in value
print(f"Item {i}: {i**2"
elif key == "nested"
return self.process_data(value
return item
''',
                "expected_fix_count": 20  # Massive number of errors
            }
        ])
    
    def test_edge_case_recovery(self, test_case: dict) -> dict:
        """Test edge case recovery with detailed analysis"""
        difficulty = test_case['difficulty']
        logger.info(f"\\n{'='*80}")
        logger.info(f" EDGE CASE TEST: {test_case['name']}")
        logger.info(f" DIFFICULTY: {difficulty}")
        logger.info(f" EXPECTED FIXES: {test_case.get('expected_fix_count', 'Unknown')}")
        logger.info(f"{'='*80}")

        try:
            broken_code = test_case['broken_code']

            # Show the broken code
            logger.info(f" BROKEN CODE ({len(broken_code)} chars):\\n{broken_code}")

            # Validate it's actually broken and count issues
            is_valid_before, syntax_error = self.editor.validate_python_syntax(broken_code)

            if is_valid_before:
                logger.warning(" Code is unexpectedly valid - skipping")
                return {
                    "test_name": test_case['name'],
                    "difficulty": difficulty,
                    "success": False,
                    "error_message": "Code was already valid"
                }

            logger.info(f" CONFIRMED SYNTAX ERROR: {syntax_error}")

            # Start timing
            import time
            start_time = time.time()

            # Apply comprehensive syntax fix
            logger.info(" LAUNCHING COMPREHENSIVE RECOVERY...")

            fixed_content, fix_success, fix_method = self.editor._comprehensive_syntax_fix(
                broken_code, syntax_error, f"temp_edge_case_{test_case['name'].replace(' ', '_').lower()}.py"
            )

            recovery_time = time.time() - start_time

            # Analyze results
            test_result = {
                "test_name": test_case['name'],
                "difficulty": difficulty,
                "success": fix_success,
                "recovery_time": recovery_time,
                "original_error": syntax_error,
                "fix_method": fix_method,
                "original_size": len(broken_code),
                "fixed_size": len(fixed_content) if fixed_content else 0
            }

            if fix_success:
                logger.info(f" EDGE CASE CONQUERED! ({recovery_time:.2f}s)")
                logger.info(f" FIX METHOD: {fix_method}")
                logger.info(f" SIZE: {len(broken_code)}  {len(fixed_content)} chars")

                # Show the fixed code (truncated if too long)
                if len(fixed_content) < 500:
                    logger.info(f" FIXED CODE:\\n{fixed_content}")
                else:
                    logger.info(f" FIXED CODE (truncated):\\n{fixed_content[:300]}\\n... [truncated] ...")

                # Validate the fix is actually correct
                is_valid_after, final_error = self.editor.validate_python_syntax(fixed_content)

                if is_valid_after:
                    logger.info(" SYNTAX VALIDATION: PASSED")
                    test_result["final_validation"] = "PASSED"

                    # Calculate complexity metrics
                    complexity_score = self.calculate_complexity_score(test_case, recovery_time)
                    test_result["complexity_score"] = complexity_score
                    logger.info(f" COMPLEXITY SCORE: {complexity_score:.2f}")

                else:
                    logger.warning(f" SYNTAX VALIDATION: FAILED - {final_error}")
                    test_result["final_validation"] = f"FAILED: {final_error}"
                    test_result["success"] = False
            else:
                logger.error(f" EDGE CASE DEFEATED US! ({recovery_time:.2f}s)")
                logger.error(f" RECOVERY FAILED: {fix_method}")
                test_result["error_message"] = fix_method

            return test_result

        except Exception as e:
            logger.error(f" CRITICAL EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            return {
                "test_name": test_case['name'],
                "difficulty": difficulty,
                "success": False,
                "error_message": str(e)
            }
    
    def calculate_complexity_score(self, test_case: dict, recovery_time: float) -> float:
        """Calculate complexity score based on multiple factors"""
        base_score = 0

        # Difficulty multiplier
        difficulty_multipliers = {
            "Basic": 1.0,
            "Intermediate": 2.0, 
            "Advanced": 3.0,
            "Expert": 4.0,
            "Nightmare": 5.0
        }

        difficulty_mult = difficulty_multipliers.get(test_case['difficulty'], 1.0)

        # Expected fix count factor
        expected_fixes = test_case.get('expected_fix_count', 1)
        fix_factor = min(expected_fixes / 5.0, 2.0)  # Cap at 2x

        # Recovery time factor (faster = higher score)
        time_factor = max(0.1, 10.0 / (recovery_time + 1))  # Inverse relationship

        # Code size factor
        code_size = len(test_case['broken_code'])
        size_factor = min(code_size / 500.0, 2.0)  # Larger code = harder

        return difficulty_mult * fix_factor * time_factor * size_factor
    
    def run_all_edge_cases(self):
        """Run all edge case tests"""
        logger.info(" STARTING EDGE CASE SYNTAX RECOVERY GAUNTLET")
        logger.info(f"Testing {len(self.test_cases)} progressively harder scenarios...")

        for test_case in self.test_cases:
            result = self.test_edge_case_recovery(test_case)
            self.results.append(result)

        return self.print_edge_case_report()
    
    def print_edge_case_report(self):
        """Print comprehensive edge case results"""
        logger.info(f"\\n{'='*100}")
        logger.info(" EDGE CASE SYNTAX RECOVERY GAUNTLET RESULTS")
        logger.info(f"{'='*100}")

        total_tests = len(self.results)
        successful_tests = sum(bool(r['success'])
                           for r in self.results)
        failed_tests = total_tests - successful_tests

        # Calculate stats by difficulty
        difficulty_stats = {}
        for result in self.results:
            diff = result['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'passed': 0}
            difficulty_stats[diff]['total'] += 1
            if result['success']:
                difficulty_stats[diff]['passed'] += 1

        logger.info(" OVERALL GAUNTLET STATISTICS:")
        logger.info(f"   Total Edge Cases: {total_tests}")
        logger.info(f"    Conquered: {successful_tests}")
        logger.info(f"    Defeated By: {failed_tests}")
        logger.info(f"    Gauntlet Success Rate: {(successful_tests/total_tests)*100:.1f}%")

        if successful_results := [r for r in self.results if r['success']]:
            avg_time = sum(r.get('recovery_time', 0) for r in successful_results) / len(successful_results)
            avg_complexity = sum(r.get('complexity_score', 0) for r in successful_results) / len(successful_results)
            logger.info(f"    Average Recovery Time: {avg_time:.2f}s")
            logger.info(f"    Average Complexity Score: {avg_complexity:.2f}")

        logger.info(f"\\n DIFFICULTY BREAKDOWN:")
        for difficulty in self.difficulty_levels:
            if difficulty in difficulty_stats:
                stats = difficulty_stats[difficulty]
                success_rate = (stats['passed'] / stats['total']) * 100
                logger.info(f"   {difficulty}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")

        logger.info(f"\\n DETAILED EDGE CASE RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = " CONQUERED" if result['success'] else " DEFEATED"
            difficulty = result['difficulty']
            time_info = f" ({result.get('recovery_time', 0):.2f}s)" if result.get('recovery_time') else ""
            complexity_info = f" [Score: {result.get('complexity_score', 0):.1f}]" if result.get('complexity_score') else ""

            logger.info(f"   {i}. {result['test_name']} ({difficulty}): {status}{time_info}{complexity_info}")

            if result['success'] and 'fix_method' in result:
                logger.info(f"      Victory Method: {result['fix_method']}")
            elif not result['success'] and 'error_message' in result:
                logger.info(f"      Defeat Reason: {result['error_message']}")

        # Final assessment based on edge case performance
        success_rate = (successful_tests/total_tests)*100

        if success_rate >= 95:
            logger.info(f"\\n LEGENDARY! {success_rate:.1f}% gauntlet success!")
            logger.info(" Your system has achieved LEGENDARY status in syntax recovery!")
        elif success_rate >= 85:
            logger.info(f"\\n HEROIC! {success_rate:.1f}% gauntlet success!")
            logger.info(" Your system shows HEROIC-level recovery capabilities!")
        elif success_rate >= 70:
            logger.info(f"\\n VALIANT! {success_rate:.1f}% gauntlet success!")
            logger.info(f" Your system demonstrates VALIANT effort against edge cases!")
        elif success_rate >= 50:
            logger.info(f"\\n BRAVE! {success_rate:.1f}% gauntlet success!")
            logger.info(f" Your system shows BRAVE resistance to complex errors!")
        else:
            logger.info(f"\\n CRUSHED! {success_rate:.1f}% gauntlet success")
            logger.info(f" The edge cases have claimed victory... for now!")

        return success_rate

def main():
    """Run the edge case gauntlet"""
    print(" EDGE CASE SYNTAX RECOVERY GAUNTLET")
    print("=" * 80)
    print("Preparing to test increasingly nightmarish syntax errors...")
    
    try:
        test_suite = EdgeCaseTest()
        test_suite.create_edge_cases()
        success_rate = test_suite.run_all_edge_cases()
        
        print("\\n" + "=" * 80)
        print(f" GAUNTLET COMPLETED! Success Rate: {success_rate:.1f}%")
        
        # Return appropriate exit code for gauntlet
        return 0 if success_rate >= 70 else 1
        
    except Exception as e:
        print(f" Critical gauntlet failure: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())