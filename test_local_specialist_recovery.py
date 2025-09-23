#!/usr/bin/env python3
"""
Simplified Enhanced Syntax Recovery Test (No External LLM Dependencies)

This test focuses on our specialist strategies and context analysis without
requiring external LLM connections, allowing us to test our core improvements.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add the automated_debugging_strategy directory to the path
# Since we're already in the automated_debugging_strategy directory, use relative import
try:
    from automated_file_editor import SafeFileEditor, EditResult
    print("✅ Successfully imported enhanced SafeFileEditor")
except ImportError as e:
    print(f"❌ Failed to import SafeFileEditor: {e}")
    sys.exit(1)

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalSpecialistTest:
    """Test suite focusing on local specialist strategies (no external LLMs)"""
    
    def __init__(self):
        # Create SafeFileEditor with no Serena to avoid any external dependencies
        self.editor = SafeFileEditor(
            validate_syntax=True,
            create_backups=True,
            use_serena=False  # Disable to avoid external dependencies
        )
        self.test_cases = []
        self.results = []
        
    def create_test_cases(self):
        """Create comprehensive test cases targeting our specialist fixes"""
        
        # Test Case 1: Simple Missing Colon
        self.test_cases.append({
            "name": "Simple Missing Colon",
            "broken_code": """def simple_function()
    return "missing colon"
""",
            "error_type": "missing_colon",
            "expected_strategies": ["colon_specialist", "context_analysis"]
        })
        
        # Test Case 2: Indentation After Colon Fix
        self.test_cases.append({
            "name": "Indentation After Function",
            "broken_code": """def function_with_indentation():
print("needs indentation")
return "fixed"
""",
            "error_type": "indentation_error",
            "expected_strategies": ["indentation_specialist", "auto_indentation"]
        })
        
        # Test Case 3: Unmatched Parentheses
        self.test_cases.append({
            "name": "Unmatched Parentheses",
            "broken_code": """def function_parens():
    result = calculate(x, y, z
    return result
""",
            "error_type": "unmatched_brackets",
            "expected_strategies": ["bracket_specialist", "context_analysis"]
        })
        
        # Test Case 4: Unterminated String
        self.test_cases.append({
            "name": "Unterminated String", 
            "broken_code": """def string_test():
    message = "this string is not closed
    return message
""",
            "error_type": "unterminated_string",
            "expected_strategies": ["string_specialist"]
        })
        
        # Test Case 5: Multiple Bracket Types
        self.test_cases.append({
            "name": "Multiple Bracket Types",
            "broken_code": """def complex_brackets():
    data = [1, 2, {"key": "value"
    return data
""",
            "error_type": "unmatched_brackets",
            "expected_strategies": ["bracket_specialist", "context_analysis"]
        })
        
        # Test Case 6: Class Definition Missing Colon
        self.test_cases.append({
            "name": "Class Missing Colon",
            "broken_code": """class MyClass
    def __init__(self):
        self.value = 42
""",
            "error_type": "missing_colon",
            "expected_strategies": ["colon_specialist", "context_analysis"]
        })
        
        # Test Case 7: For Loop Missing Colon and Parentheses
        self.test_cases.append({
            "name": "For Loop Complex Error",
            "broken_code": """def loop_function():
    for i in range(10
        print(i)
""",
            "error_type": "unmatched_brackets",  # First error will be the unclosed parentheses
            "expected_strategies": ["bracket_specialist", "context_analysis"]
        })
        
        # Test Case 8: If Statement Missing Colon
        self.test_cases.append({
            "name": "If Statement Missing Colon",
            "broken_code": """def conditional():
    if True
        return "condition met"
    return "fallback"
""",
            "error_type": "missing_colon",
            "expected_strategies": ["colon_specialist", "context_analysis"]
        })
    
    def test_specialist_recovery(self, test_case: dict) -> dict:
        """Test specialist recovery focusing on local strategies"""
        logger.info(f"\\n{'='*60}")
        logger.info(f"🧪 TESTING: {test_case['name']}")
        logger.info(f"Expected Error Type: {test_case['error_type']}")
        logger.info(f"{'='*60}")

        try:
            broken_code = test_case['broken_code']

            # Show the broken code
            logger.info(f"📝 BROKEN CODE:\\n{broken_code}")

            # Validate it's actually broken
            is_valid_before, syntax_error = self.editor.validate_python_syntax(broken_code)

            if is_valid_before:
                logger.warning("⚠️ Code is already valid - not a good test case")
                return {
                    "test_name": test_case['name'],
                    "success": False,
                    "error_message": "Code was already valid"
                }

            logger.info(f"❌ CONFIRMED SYNTAX ERROR: {syntax_error}")

            # Classify the error to verify our classification logic
            classified_type = self.editor._classify_syntax_error(syntax_error)
            logger.info(f"🔍 CLASSIFIED AS: {classified_type}")

            # Create temporary file path for testing
            temp_path = f"temp_test_{test_case['name'].replace(' ', '_').lower()}.py"

            # Apply comprehensive syntax fix directly
            logger.info("🔧 ATTEMPTING LOCAL SPECIALIST RECOVERY...")

            fixed_content, fix_success, fix_method = self.editor._comprehensive_syntax_fix(
                broken_code, syntax_error, temp_path
            )

            # Analyze results
            test_result = {
                "test_name": test_case['name'],
                "success": fix_success,
                "error_type": classified_type,
                "expected_type": test_case['error_type'],
                "original_error": syntax_error,
                "fix_method": fix_method,
                "classification_correct": classified_type == test_case['error_type']
            }

            if fix_success:
                logger.info(f"✅ SUCCESS: {test_case['name']} was successfully recovered!")
                logger.info(f"🔧 FIX METHOD: {fix_method}")
                logger.info(f"🔧 FIXED CODE:\\n{fixed_content}")

                # Validate the fix is actually correct
                is_valid_after, final_error = self.editor.validate_python_syntax(fixed_content)

                if is_valid_after:
                    logger.info("✅ SYNTAX VALIDATION: PASSED")
                    test_result["final_validation"] = "PASSED"
                else:
                    logger.warning(f"⚠️ SYNTAX VALIDATION: FAILED - {final_error}")
                    test_result["final_validation"] = f"FAILED: {final_error}"
                    test_result["success"] = False
            else:
                logger.error(f"❌ FAILED: {test_case['name']} could not be recovered")
                logger.error(f"Fix method attempted: {fix_method}")
                test_result["error_message"] = fix_method

            return test_result

        except Exception as e:
            logger.error(f"💥 EXCEPTION during test: {e}")
            import traceback
            traceback.print_exc()
            return {
                "test_name": test_case['name'],
                "success": False,
                "error_message": str(e)
            }
    
    def run_all_tests(self):
        """Run all specialist recovery tests"""
        logger.info("🚀 STARTING LOCAL SPECIALIST SYNTAX RECOVERY TEST")
        logger.info(f"Testing {len(self.test_cases)} different error scenarios...")

        for test_case in self.test_cases:
            result = self.test_specialist_recovery(test_case)
            self.results.append(result)

        return self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive test results"""
        logger.info(f"\\n{'='*80}")
        logger.info("📊 LOCAL SPECIALIST SYNTAX RECOVERY TEST RESULTS")
        logger.info(f"{'='*80}")

        total_tests = len(self.results)
        successful_tests = sum(bool(r['success'])
                           for r in self.results)
        failed_tests = total_tests - successful_tests
        correct_classifications = sum(bool(r.get('classification_correct', False))
                                  for r in self.results)

        logger.info("📈 OVERALL STATISTICS:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   ✅ Successful: {successful_tests}")
        logger.info(f"   ❌ Failed: {failed_tests}")
        logger.info(f"   🎯 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        logger.info(f"   🔍 Classification Accuracy: {(correct_classifications/total_tests)*100:.1f}%")

        logger.info(f"\\n📋 DETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            classification = "✅" if result.get('classification_correct', False) else "❌"
            logger.info(f"   {i}. {result['test_name']}: {status} (Classification: {classification})")

            if result['success'] and 'fix_method' in result:
                logger.info(f"      Fix Method: {result['fix_method']}")
                if 'final_validation' in result:
                    logger.info(f"      Final Validation: {result['final_validation']}")
            elif not result['success'] and 'error_message' in result:
                logger.info(f"      Error: {result['error_message']}")

        # Strategy effectiveness analysis
        logger.info(f"\\n🔬 STRATEGY ANALYSIS:")

        # Count successful fix methods
        fix_methods = [r.get('fix_method', 'unknown') for r in self.results if r['success']]
        logger.info("   Successful Fix Methods:")
        for method in set(fix_methods):
            count = fix_methods.count(method)
            logger.info(f"     - {method}: {count}")

        # Classification accuracy by type
        logger.info(f"\\n🔍 CLASSIFICATION ANALYSIS:")
        error_types = {}
        for result in self.results:
            expected = result.get('expected_type', 'unknown')
            classified = result.get('error_type', 'unknown')
            if expected not in error_types:
                error_types[expected] = {'correct': 0, 'total': 0}
            error_types[expected]['total'] += 1
            if expected == classified:
                error_types[expected]['correct'] += 1

        for error_type, stats in error_types.items():
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"   {error_type}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

        # Final assessment
        success_rate = (successful_tests/total_tests)*100

        if success_rate >= 95:
            logger.info(f"\\n🏆 OUTSTANDING! {success_rate:.1f}% success rate!")
            logger.info("🎉 The specialist recovery system is working excellently!")
        elif success_rate >= 85:
            logger.info(f"\\n🥇 EXCELLENT! {success_rate:.1f}% success rate!")
            logger.info("💪 The specialist system is highly effective!")
        elif success_rate >= 70:
            logger.info(f"\\n🥈 GOOD! {success_rate:.1f}% success rate!")
            logger.info(f"🔧 The specialist system shows strong capabilities!")
        else:
            logger.info(f"\\n⚠️  NEEDS IMPROVEMENT: {success_rate:.1f}% success rate")
            logger.info(f"🛠️  Consider enhancing the specialist strategies!")

        return success_rate

def main():
    """Run the local specialist syntax recovery test"""
    print("🚀 LOCAL SPECIALIST SYNTAX RECOVERY TEST")
    print("=" * 70)
    
    try:
        test_suite = LocalSpecialistTest()
        test_suite.create_test_cases()
        success_rate = test_suite.run_all_tests()
        
        print("\\n" + "=" * 70)
        print(f"🏁 LOCAL SPECIALIST TEST COMPLETED! Success Rate: {success_rate:.1f}%")
        
        # Return appropriate exit code
        return 0 if success_rate >= 85 else 1
        
    except Exception as e:
        print(f"💥 Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())