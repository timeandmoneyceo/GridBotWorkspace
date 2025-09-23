#!/usr/bin/env python3
"""
Enhanced Comprehensive Syntax Recovery Test

This test directly triggers our comprehensive syntax recovery system
by creating broken files and using the correct SafeFileEditor interface.
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

class EnhancedSyntaxRecoveryTest:
    """Enhanced test suite for comprehensive syntax error recovery"""
    
    def __init__(self):
        self.editor = SafeFileEditor(
            validate_syntax=True,
            create_backups=True,
            use_serena=True
        )
        self.test_cases = []
        self.results = []
        
    def create_test_cases(self):
        """Create various types of broken Python code to test recovery"""
        
        # Test Case 1: Simple Indentation Error
        self.test_cases.append({
            "name": "Simple Indentation Error",
            "broken_code": """def simple_function():
print("This needs indentation")
return "fixed"
""",
            "expected_fix": """def simple_function():
    print("This needs indentation")
    return "fixed"
""",
            "error_type": "IndentationError"
        })
        
        # Test Case 2: Missing Colon
        self.test_cases.append({
            "name": "Missing Colon",
            "broken_code": """def function_missing_colon()
    return "needs colon"
""",
            "expected_fix": """def function_missing_colon():
    return "needs colon"
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 3: Unmatched Parentheses
        self.test_cases.append({
            "name": "Unmatched Parentheses", 
            "broken_code": """def function_with_parens():
    result = some_function(arg1, arg2
    return result
""",
            "expected_fix": """def function_with_parens():
    result = some_function(arg1, arg2)
    return result
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 4: Missing Quote
        self.test_cases.append({
            "name": "Missing Quote",
            "broken_code": """def string_function():
    message = "unclosed string
    return message
""",
            "expected_fix": """def string_function():
    message = "unclosed string"
    return message
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 5: Complex Mixed Errors
        self.test_cases.append({
            "name": "Complex Mixed Errors",
            "broken_code": """def complex_function()
data = []
    for i in range(5
        data.append(i * 2
    return data
""",
            "expected_fix": """def complex_function():
    data = []
    for i in range(5):
        data.append(i * 2)
    return data
""",
            "error_type": "SyntaxError"
        })
    
    def test_comprehensive_recovery_directly(self, test_case: dict) -> dict:
        """Test comprehensive recovery by directly calling the comprehensive fix"""
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
                    "error_type": test_case['error_type'],
                    "error_message": "Code was already valid"
                }

            logger.info(f"❌ CONFIRMED SYNTAX ERROR: {syntax_error}")

            # Create temporary file path for testing
            temp_path = f"temp_test_{test_case['name'].replace(' ', '_').lower()}.py"

            # Apply comprehensive syntax fix directly
            logger.info("🔧 ATTEMPTING COMPREHENSIVE RECOVERY...")

            fixed_content, fix_success, fix_method = self.editor._comprehensive_syntax_fix(
                broken_code, syntax_error, temp_path
            )

            # Analyze results
            test_result = {
                "test_name": test_case['name'],
                "success": fix_success,
                "error_type": test_case['error_type'],
                "original_error": syntax_error,
                "fix_method": fix_method,
                "recovery_attempted": True
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
                "error_type": test_case['error_type'],
                "error_message": str(e),
                "recovery_attempted": False
            }
    
    def run_all_tests(self):
        """Run all comprehensive recovery tests"""
        logger.info("🚀 STARTING ENHANCED COMPREHENSIVE SYNTAX RECOVERY TEST")
        logger.info(f"Testing {len(self.test_cases)} different error scenarios...")

        for test_case in self.test_cases:
            result = self.test_comprehensive_recovery_directly(test_case)
            self.results.append(result)

        return self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive test results"""
        logger.info(f"\\n{'='*80}")
        logger.info("📊 ENHANCED COMPREHENSIVE SYNTAX RECOVERY TEST RESULTS")
        logger.info(f"{'='*80}")

        total_tests = len(self.results)
        successful_tests = sum(bool(r['success'])
                           for r in self.results)
        failed_tests = total_tests - successful_tests

        logger.info("📈 OVERALL STATISTICS:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   ✅ Successful: {successful_tests}")
        logger.info(f"   ❌ Failed: {failed_tests}")
        logger.info(f"   🎯 Success Rate: {(successful_tests/total_tests)*100:.1f}%")

        logger.info(f"\\n📋 DETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"   {i}. {result['test_name']}: {status}")

            if result['success'] and 'fix_method' in result:
                logger.info(f"      Fix Method: {result['fix_method']}")
                if 'final_validation' in result:
                    logger.info(f"      Final Validation: {result['final_validation']}")
            elif not result['success'] and 'error_message' in result:
                logger.info(f"      Error: {result['error_message']}")

        # Strategy effectiveness analysis
        logger.info(f"\\n🔬 STRATEGY ANALYSIS:")
        logger.info(f"   Recovery Attempted: {sum(bool(r.get('recovery_attempted', False))
                                              for r in self.results)}/{total_tests}")

        # Count successful fix methods
        fix_methods = [r.get('fix_method', 'unknown') for r in self.results if r['success']]
        logger.info("   Successful Fix Methods:")
        for method in set(fix_methods):
            count = fix_methods.count(method)
            logger.info(f"     - {method}: {count}")

        # Final assessment
        success_rate = (successful_tests/total_tests)*100

        if success_rate == 100:
            logger.info(f"\\n🏆 PERFECT SCORE! All syntax errors successfully recovered!")
            logger.info("🎉 The comprehensive recovery system is working flawlessly!")
        elif success_rate >= 80:
            logger.info(f"\\n🥈 EXCELLENT! {success_rate:.1f}% success rate!")
            logger.info("💪 The recovery system is highly effective!")
        elif success_rate >= 60:
            logger.info(f"\\n🥉 GOOD! {success_rate:.1f}% success rate!")
            logger.info(f"🔧 The recovery system shows strong capabilities!")
        else:
            logger.info(f"\\n⚠️  NEEDS IMPROVEMENT: {success_rate:.1f}% success rate")
            logger.info(f"🛠️  Consider enhancing the recovery strategies!")

        return success_rate

def main():
    """Run the enhanced comprehensive syntax recovery test"""
    print("🚀 ENHANCED COMPREHENSIVE SYNTAX RECOVERY TEST")
    print("=" * 70)
    
    try:
        test_suite = EnhancedSyntaxRecoveryTest()
        test_suite.create_test_cases()
        success_rate = test_suite.run_all_tests()
        
        print("\\n" + "=" * 70)
        print(f"🏁 ENHANCED TEST COMPLETED! Success Rate: {success_rate:.1f}%")
        
        # Return appropriate exit code
        return 0 if success_rate >= 80 else 1
        
    except Exception as e:
        print(f"💥 Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())