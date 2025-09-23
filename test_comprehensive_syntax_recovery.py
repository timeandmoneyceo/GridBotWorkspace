#!/usr/bin/env python3
"""
Comprehensive Syntax Recovery Stress Test

This test creates intentionally broken Python code with various types of errors
and challenges our enhanced SafeFileEditor to fix ALL of them using the
comprehensive multi-strategy approach.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

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

class ComprehensiveSyntaxRecoveryTest:
    """Comprehensive test suite for syntax error recovery"""
    
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
        
        # Test Case 1: Indentation Errors
        self.test_cases.append({
            "name": "Indentation Error",
            "broken_code": """def function_with_bad_indentation():
print("This should be indented")
    return "mixed indentation"

class BadClass:
def method(self):
return "not indented"
""",
            "error_type": "IndentationError"
        })
        
        # Test Case 2: Missing Colons
        self.test_cases.append({
            "name": "Missing Colons",
            "broken_code": """def function_missing_colon()
    return "needs colon"

if True
    print("missing colon in if")

for i in range(10)
    print(i)

class MissingColonClass
    def __init__(self)
        self.value = 42
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 3: Unmatched Parentheses/Brackets
        self.test_cases.append({
            "name": "Unmatched Parentheses",
            "broken_code": """def function_with_unmatched_parens():
    result = some_function(arg1, arg2
    return result
    
    other_result = another_function()))))
    
    list_with_issues = [1, 2, 3, 4
    dict_with_issues = {"key": "value", "other": missing_bracket}
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 4: Mixed Tabs and Spaces (Python's nemesis)
        self.test_cases.append({
            "name": "Mixed Tabs and Spaces",
            "broken_code": """def mixed_indentation():
    if True:
\t    print("tab indented")
        print("space indented")
\treturn "tab return"
""",
            "error_type": "IndentationError"
        })
        
        # Test Case 5: Complex Nested Structure Errors
        self.test_cases.append({
            "name": "Complex Nested Errors",
            "broken_code": """class ComplexClass:
def __init__(self)
    self.data = {
        "nested": [
            {"inner": "value"
        ]
    }
    
def method_with_issues(self, param1, param2
    if param1
        for item in param2
        print(item
        return item * 2
    else
        print("else clause"
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 6: Import and String Errors
        self.test_cases.append({
            "name": "Import and String Errors",
            "broken_code": """import os, sys
from pathlib import Path, 

def function_with_string_issues():
    message = "unclosed string
    other_message = 'another unclosed string
    return f"formatted {message
    
class StringClass:
    def __init__(self):
        self.value = "This is a very long string that might cause issues if not properly closed
""",
            "error_type": "SyntaxError"
        })
        
        # Test Case 7: Multiple Error Types Combined
        self.test_cases.append({
            "name": "Multiple Combined Errors",
            "broken_code": """import json
from typing import List Dict

def complex_function_with_multiple_errors(params: List[str])  # Missing colon
data = []  # Wrong indentation
    for param in params  # Missing colon
        try
            result = json.loads(param
            data.append(result
        except Exception as e  # Missing colon
        print(f"Error: {e}")  # Wrong indentation
    return data

class MultiErrorClass
    def __init__(self, value)  # Missing colon
    self.value = value  # Wrong indentation
    
def another_broken_function(
    return "missing parameter and colon"
""",
            "error_type": "SyntaxError"
        })
    
    def run_single_test(self, test_case: dict) -> dict:
        """Run a single test case through comprehensive recovery"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 TESTING: {test_case['name']}")
        logger.info(f"Expected Error Type: {test_case['error_type']}")
        logger.info(f"{'='*60}")

        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(test_case['broken_code'])
            temp_path = temp_file.name

        try:
            # Show the broken code
            logger.info(f"📝 BROKEN CODE:\n{test_case['broken_code']}")

            # Attempt comprehensive fix using line replacement (triggers all our recovery logic)
            logger.info("🔧 ATTEMPTING COMPREHENSIVE RECOVERY...")

            result = self.editor.apply_line_replacement(
                file_path=temp_path,
                old_line="# This line doesn't exist, triggering our recovery logic",
                new_line="# Fixed by comprehensive recovery system",
                line_number=1
            )

            # Analyze results
            test_result = {
                "test_name": test_case['name'],
                "success": result.success,
                "error_type": test_case['error_type'],
                "backup_created": result.backup_path is not None,
                "changes_made": result.changes_made,
                "error_message": result.error,
                "recovery_attempted": True
            }

            if result.success:
                logger.info(f"✅ SUCCESS: {test_case['name']} was successfully recovered!")
                # Read the fixed content
                with open(temp_path, 'r') as f:
                    fixed_content = f.read()
                logger.info(f"🔧 FIXED CODE:\n{fixed_content}")
            else:
                logger.error(f"❌ FAILED: {test_case['name']} could not be recovered")
                if result.error:
                    logger.error(f"Error details: {result.error}")

            return test_result

        except Exception as e:
            logger.error(f"💥 EXCEPTION during test: {e}")
            return {
                "test_name": test_case['name'],
                "success": False,
                "error_type": test_case['error_type'],
                "backup_created": False,
                "changes_made": False,
                "error_message": str(e),
                "recovery_attempted": False
            }

        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def run_all_tests(self):
        """Run all comprehensive recovery tests"""
        logger.info("🚀 STARTING COMPREHENSIVE SYNTAX RECOVERY STRESS TEST")
        logger.info(f"Testing {len(self.test_cases)} different error scenarios...")

        for test_case in self.test_cases:
            result = self.run_single_test(test_case)
            self.results.append(result)

        self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive test results"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 COMPREHENSIVE SYNTAX RECOVERY TEST RESULTS")
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

        logger.info(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"   {i}. {result['test_name']}: {status}")
            if not result['success'] and result['error_message']:
                logger.info(f"      Error: {result['error_message']}")

        # Strategy effectiveness analysis
        logger.info(f"\n🔬 STRATEGY ANALYSIS:")
        logger.info(f"   Recovery Attempted: {sum(bool(r['recovery_attempted'])
                                              for r in self.results)}/{total_tests}")
        logger.info(f"   Backups Created: {sum(bool(r['backup_created'])
                                           for r in self.results)}/{total_tests}")
        logger.info(f"   Changes Applied: {sum(bool(r['changes_made'])
                                           for r in self.results)}/{total_tests}")

        # Final assessment
        if successful_tests == total_tests:
            logger.info(f"\n🏆 PERFECT SCORE! All syntax errors successfully recovered!")
            logger.info("🎉 The comprehensive recovery system is working flawlessly!")
        elif successful_tests >= total_tests * 0.8:
            logger.info(f"\n🥈 EXCELLENT! {(successful_tests/total_tests)*100:.1f}% success rate!")
            logger.info("💪 The recovery system is highly effective!")
        elif successful_tests >= total_tests * 0.6:
            logger.info(f"\n🥉 GOOD! {(successful_tests/total_tests)*100:.1f}% success rate!")
            logger.info("🔧 The recovery system shows strong capabilities!")
        else:
            logger.info(f"\n⚠️  NEEDS IMPROVEMENT: {(successful_tests/total_tests)*100:.1f}% success rate")
            logger.info(f"🛠️  Consider enhancing the recovery strategies!")

def main():
    """Run the comprehensive syntax recovery stress test"""
    print("🚀 LAUNCHING COMPREHENSIVE SYNTAX RECOVERY STRESS TEST")
    print("=" * 70)
    
    try:
        test_suite = ComprehensiveSyntaxRecoveryTest()
        test_suite.create_test_cases()
        test_suite.run_all_tests()
        
        print("\n" + "=" * 70)
        print("🏁 COMPREHENSIVE STRESS TEST COMPLETED!")
        
    except Exception as e:
        print(f"💥 Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())