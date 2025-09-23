"""
AI-Driven Testing and Debugging Module

This module provides automated test generation and intelligent debugging assistance
for the GridBot automation pipeline using AI-powered analysis.
"""

import ast
import inspect
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
try:
    import pytest  # type: ignore
except ImportError:
    pytest = None
from datetime import datetime
import json
import os

class AITestGenerator:
    """
    AI-powered test case generator for GridBot codebase
    """
    
    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)
        
        # Test generation patterns
        self.test_patterns = {
            'basic_function': self.generate_basic_function_tests,
            'class_method': self.generate_method_tests,
            'error_handling': self.generate_error_tests,
            'integration': self.generate_integration_tests,
            'performance': self.generate_performance_tests
        }
        
        # Code analysis cache
        self.analysis_cache = {}
        
        # Redundant code for testing Sourcery auto-fix capabilities
        x = 1
        if x == 1:
            y = 2
        else:
            y = 3
        z = y
        
    def analyze_function(self, function_code: str, function_name: str) -> Dict:
        """
        Analyze function to determine appropriate test types
        
        Args:
            function_code: Source code of the function
            function_name: Name of the function
            
        Returns:
            Dict with function analysis results
        """
        analysis = {
            'function_name': function_name,
            'complexity': 'low',
            'has_parameters': False,
            'has_return': False,
            'raises_exceptions': False,
            'uses_external_deps': False,
            'is_async': False,
            'test_types_needed': []
        }

        try:
            # Parse the function code
            tree = ast.parse(function_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check parameters
                    if node.args.args:
                        analysis['has_parameters'] = True

                    # Check for async
                    if hasattr(node, 'decorator_list'):
                        for decorator in node.decorator_list:
                            if hasattr(decorator, 'id') and decorator.id == 'asyncio':  # type: ignore
                                analysis['is_async'] = True

                elif isinstance(node, ast.Return):
                    analysis['has_return'] = True

                elif isinstance(node, ast.Raise):
                    analysis['raises_exceptions'] = True

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis['uses_external_deps'] = True

            # Determine complexity based on code structure
            if function_code.count('\n') > 20 or function_code.count('if ') > 3:
                analysis['complexity'] = 'high'
            elif function_code.count('\n') > 10 or function_code.count('if ') > 1:
                analysis['complexity'] = 'medium'

            # Determine needed test types
            test_types = ['basic_function']

            if analysis['has_parameters']:
                test_types.append('parameter_validation')

            if analysis['has_return']:
                test_types.append('return_value_validation')

            if analysis['raises_exceptions']:
                test_types.append('error_handling')

            if analysis['uses_external_deps']:
                test_types.append('integration')

            if analysis['complexity'] == 'high':
                test_types.append('performance')

            analysis['test_types_needed'] = test_types

        except Exception as e:
            self.logger.warning(f"Error analyzing function {function_name}: {e}")

        return analysis
        
    def generate_basic_function_tests(self, function_name: str, analysis: Dict) -> List[str]:
        """Generate basic function tests"""
        tests = [
            f"""
def test_{function_name}_executes_without_error():
    \"\"\"Test that {function_name} executes without raising exceptions\"\"\"
    try:
        result = {function_name}()
        assert True  # Function executed successfully
    except Exception as e:
        pytest.fail(f"{function_name} raised an exception: {{e}}")
"""
        ]

        # Return type test if function returns values
        if analysis['has_return']:
            tests.append(f"""
def test_{function_name}_returns_expected_type():
    \"\"\"Test that {function_name} returns expected type\"\"\"
    result = {function_name}()
    # Add specific type assertions based on function analysis
    assert result is not None, "Function should return a value"
""")

        return tests
        
    def generate_method_tests(self, function_name: str, analysis: Dict) -> List[str]:
        """Generate tests for class methods"""
        return [
            f"""
def test_{function_name}_method():
    \"\"\"Test {function_name} method functionality\"\"\"
    # Create test instance
    instance = TestClass()  # Replace with actual class
    
    # Test method execution
    result = instance.{function_name}()
    
    # Add specific assertions based on method purpose
    assert hasattr(instance, '{function_name}'), "Method should exist"
"""
        ]
        
    def generate_error_tests(self, function_name: str, analysis: Dict) -> List[str]:
        """Generate error handling tests"""
        tests = []
        
        if analysis['raises_exceptions']:
            tests.append(f"""
def test_{function_name}_error_handling():
    \"\"\"Test error handling in {function_name}\"\"\"
    # Test with invalid inputs
    with pytest.raises(Exception):
        {function_name}(invalid_input=True)
        
    # Test specific exception types if known
    # with pytest.raises(ValueError):
    #     {function_name}(invalid_value)
""")
            
        # Parameter validation tests
        if analysis['has_parameters']:
            tests.append(f"""
def test_{function_name}_parameter_validation():
    \"\"\"Test parameter validation in {function_name}\"\"\"
    # Test with None parameters
    with pytest.raises((TypeError, ValueError)):
        {function_name}(None)
        
    # Test with wrong parameter types
    # Add specific parameter tests based on function signature
""")
            
        return tests
        
    def generate_integration_tests(self, function_name: str, analysis: Dict) -> List[str]:
        """Generate integration tests"""
        return [
            f"""
def test_{function_name}_integration():
    \"\"\"Integration test for {function_name}\"\"\"
    # Setup test environment
    # Mock external dependencies if needed
    
    # Test function with realistic data
    result = {function_name}()
    
    # Verify integration points
    assert result is not None, "Integration should produce results"
    
    # Test with various scenarios
    # Add scenario-specific assertions
"""
        ]
        
    def generate_performance_tests(self, function_name: str, analysis: Dict) -> List[str]:
        """Generate performance tests"""
        return [
            f"""
def test_{function_name}_performance():
    \"\"\"Performance test for {function_name}\"\"\"
    import time
    
    # Measure execution time
    start_time = time.time()
    result = {function_name}()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Assert reasonable execution time (adjust threshold as needed)
    assert execution_time < 5.0, f"Function took too long: {{execution_time}}s"
    
    # Test with larger datasets if applicable
    # Add performance benchmarks
"""
        ]
        
    def generate_test_file(self, file_path: str, functions_to_test: Optional[List[str]] = None) -> str:
        """
        Generate comprehensive test file for a Python module
        
        Args:
            file_path: Path to the Python file to test
            functions_to_test: List of specific functions to test (optional)
            
        Returns:
            Generated test file content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse the source code to extract functions
            tree = ast.parse(source_code)
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not functions_to_test or node.name in functions_to_test:
                        functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
            # Generate test file content
            module_name = os.path.basename(file_path).replace('.py', '')
            test_content = f'''"""
AI-Generated Test Suite for {module_name}

This test file was automatically generated by the GridBot AI Testing System.
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Module under test: {file_path}
Functions tested: {", ".join(functions)}
Classes tested: {", ".join(classes)}
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the module directory to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module under test
try:
    from {module_name} import *
except ImportError as e:
    pytest.skip(f"Could not import module {{module_name}}: {{e}}")

class TestAIGenerated:
    """AI-Generated test class for {module_name}"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Initialize test environment
        self.test_data = {{}}
        
    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up test environment
        pass

'''
            
            # Generate tests for each function
            for func_name in functions:
                try:
                    # Extract function code for analysis
                    func_code = self.extract_function_code(source_code, func_name)
                    analysis = self.analyze_function(func_code, func_name)
                    
                    # Generate tests based on analysis
                    for test_type in analysis['test_types_needed']:
                        if test_type in self.test_patterns:
                            generated_tests = self.test_patterns[test_type](func_name, analysis)
                            for test in generated_tests:
                                test_content += test + '\n'
                                
                except Exception as e:
                    self.logger.warning(f"Could not generate tests for {func_name}: {e}")
                    # Add a basic test as fallback
                    test_content += f'''
    def test_{func_name}_basic(self):
        """Basic test for {func_name} (AI-generated fallback)"""
        # This is a fallback test - please customize based on function requirements
        assert hasattr(sys.modules['{module_name}'], '{func_name}'), "Function {func_name} should exist"
'''
                    
            # Add integration tests
            test_content += f'''

class TestIntegration:
    """Integration tests for {module_name}"""
    
    def test_module_imports(self):
        """Test that module imports correctly"""
        try:
            import {module_name}
            assert True
        except ImportError:
            pytest.fail("Module {module_name} should import without errors")
            
    def test_module_functions_exist(self):
        """Test that expected functions exist in module"""
        import {module_name}
        
        expected_functions = {functions}
        for func_name in expected_functions:
            assert hasattr({module_name}, func_name), f"Function {{func_name}} should exist in module"
            assert callable(getattr({module_name}, func_name)), f"{{func_name}} should be callable"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
'''
            
            return test_content
            
        except Exception as e:
            self.logger.error(f"Error generating test file for {file_path}: {e}")
            return f"# Error generating tests: {e}"
            
    def extract_function_code(self, source_code: str, function_name: str) -> str:
        """Extract the code for a specific function"""
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get the function code by line numbers
                    lines = source_code.split('\n')
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                    
                    return '\n'.join(lines[start_line:end_line])
                    
        except Exception as e:
            self.logger.warning(f"Could not extract function code for {function_name}: {e}")
            
        return ""

class AIDebuggingAssistant:
    """
    AI-powered debugging assistant for intelligent error analysis
    """
    
    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface
        self.logger = logging.getLogger(__name__)
        
        # Error pattern database
        self.error_patterns = {
            'SyntaxError': self.analyze_syntax_error,
            'NameError': self.analyze_name_error,
            'TypeError': self.analyze_type_error,
            'AttributeError': self.analyze_attribute_error,
            'ImportError': self.analyze_import_error,
            'ValueError': self.analyze_value_error,
            'KeyError': self.analyze_key_error,
            'IndexError': self.analyze_index_error
        }
        
        # Fix suggestion database
        self.fix_suggestions = {}
        
    def analyze_error(self, error_info: Dict) -> Dict:
        """
        Comprehensive error analysis using AI
        
        Args:
            error_info: Dictionary containing error details
            
        Returns:
            Dict with analysis results and fix suggestions
        """
        analysis = {
            'error_type': error_info.get('type', 'Unknown'),
            'error_message': error_info.get('message', ''),
            'file_path': error_info.get('file', ''),
            'line_number': error_info.get('line', 0),
            'code_context': error_info.get('context', ''),
            'analysis_results': {},
            'fix_suggestions': [],
            'confidence_score': 0.0,
            'estimated_fix_time': 'Unknown'
        }
        
        error_type = analysis['error_type']
        
        # Apply specific error analysis
        if error_type in self.error_patterns:
            specific_analysis = self.error_patterns[error_type](error_info)
            analysis['analysis_results'].update(specific_analysis)
            
        # Generate fix suggestions
        fix_suggestions = self.generate_fix_suggestions(analysis)
        analysis['fix_suggestions'] = fix_suggestions
        
        # Calculate confidence score
        analysis['confidence_score'] = self.calculate_confidence_score(analysis)
        
        # Estimate fix time
        analysis['estimated_fix_time'] = self.estimate_fix_time(analysis)
        
        return analysis
        
    def analyze_syntax_error(self, error_info: Dict) -> Dict:
        """Analyze syntax errors"""
        analysis = {
            'likely_causes': [],
            'specific_suggestions': [],
            'code_patterns': []
        }
        
        message = error_info.get('message', '').lower()
        context = error_info.get('context', '')
        
        if 'invalid syntax' in message:
            analysis['likely_causes'].append('Missing or incorrect punctuation')
            analysis['specific_suggestions'].append('Check for missing colons, parentheses, or quotes')
            
        if 'eol while scanning' in message:
            analysis['likely_causes'].append('Unterminated string literal')
            analysis['specific_suggestions'].append('Add closing quote to string')
            
        if 'unexpected eof' in message:
            analysis['likely_causes'].append('Unmatched brackets or parentheses')
            analysis['specific_suggestions'].append('Check for missing closing brackets')
            
        # Analyze code context
        if context:
            if context.count('(') != context.count(')'):
                analysis['code_patterns'].append('Unmatched parentheses detected')
            if context.count('[') != context.count(']'):
                analysis['code_patterns'].append('Unmatched square brackets detected')
            if context.count('{') != context.count('}'):
                analysis['code_patterns'].append('Unmatched curly braces detected')
                
        return analysis
        
    def analyze_name_error(self, error_info: Dict) -> Dict:
        """Analyze name errors"""
        analysis = {
            'likely_causes': ['Variable not defined', 'Typo in variable name', 'Missing import'],
            'specific_suggestions': [],
            'scope_analysis': {}
        }
        
        message = error_info.get('message', '')
        
        # Extract variable name from error message
        if "name '" in message and "' is not defined" in message:
            var_name = message.split("name '")[1].split("' is not defined")[0]
            analysis['undefined_variable'] = var_name
            
            # Generate specific suggestions
            analysis['specific_suggestions'].extend([
                f"Check if '{var_name}' is spelled correctly",
                f"Ensure '{var_name}' is defined before use",
                f"Check if '{var_name}' requires an import statement"
            ])
            
        return analysis
        
    def analyze_type_error(self, error_info: Dict) -> Dict:
        """Analyze type errors"""
        analysis = {
            'likely_causes': ['Incorrect parameter types', 'Wrong number of arguments', 'Type mismatch'],
            'specific_suggestions': [],
            'type_hints': []
        }
        
        message = error_info.get('message', '')
        
        if 'takes' in message and 'positional argument' in message:
            analysis['specific_suggestions'].append('Check function call arguments')
            analysis['type_hints'].append('Add type hints to function parameters')
            
        if 'unsupported operand type' in message:
            analysis['specific_suggestions'].append('Check operand types for compatibility')
            
        return analysis
        
    def analyze_attribute_error(self, error_info: Dict) -> Dict:
        """Analyze attribute errors"""
        analysis = {
            'likely_causes': ['Object does not have attribute', 'Typo in attribute name', 'None object'],
            'specific_suggestions': [],
            'object_analysis': {}
        }
        
        message = error_info.get('message', '')
        
        if 'has no attribute' in message:
            parts = message.split("'")
            if len(parts) >= 4:
                obj_type = parts[1]
                attr_name = parts[3]
                
                analysis['object_analysis'] = {
                    'object_type': obj_type,
                    'missing_attribute': attr_name
                }
                
                analysis['specific_suggestions'].extend([
                    f"Check if '{attr_name}' is spelled correctly",
                    f"Verify that {obj_type} objects have '{attr_name}' attribute",
                    f"Check if object is None before accessing '{attr_name}'"
                ])
                
        return analysis
        
    def analyze_import_error(self, error_info: Dict) -> Dict:
        """Analyze import errors"""
        analysis = {
            'likely_causes': ['Module not installed', 'Module not found', 'Circular import'],
            'specific_suggestions': [],
            'installation_commands': []
        }

        message = error_info.get('message', '')

        if 'No module named' in message:
            module_name = message.split("'")[1] if "'" in message else ""

            if module_name:
                analysis['missing_module'] = module_name
                analysis['specific_suggestions'].extend(
                    [
                        f"Install module: pip install {module_name}",
                        f"Check if '{module_name}' is in PYTHONPATH",
                        "Verify module name spelling",
                    ]
                )
                analysis['installation_commands'].append(f"pip install {module_name}")

        return analysis
        
    def analyze_value_error(self, error_info: Dict) -> Dict:
        """Analyze value errors"""
        return {
            'likely_causes': ['Invalid input value', 'Format mismatch', 'Range error'],
            'specific_suggestions': ['Validate input values', 'Check data format', 'Add input validation']
        }
        
    def analyze_key_error(self, error_info: Dict) -> Dict:
        """Analyze key errors"""
        return {
            'likely_causes': ['Key does not exist in dictionary', 'Typo in key name'],
            'specific_suggestions': ['Use dict.get() with default value', 'Check key existence with "in" operator']
        }
        
    def analyze_index_error(self, error_info: Dict) -> Dict:
        """Analyze index errors"""
        return {
            'likely_causes': ['Index out of range', 'Empty list/string', 'Negative index issue'],
            'specific_suggestions': ['Check list/string length before indexing', 'Use try-except for index access']
        }
        
    def generate_fix_suggestions(self, analysis: Dict) -> List[Dict]:
        """Generate specific fix suggestions based on analysis"""
        suggestions = []

        error_type = analysis['error_type']
        analysis_results = analysis['analysis_results']

        # Generate type-specific suggestions
        if error_type == 'SyntaxError':
            for suggestion in analysis_results.get('specific_suggestions', []):
                suggestions.append({
                    'type': 'syntax_fix',
                    'description': suggestion,
                    'priority': 'high',
                    'auto_fixable': True
                })

        elif error_type == 'NameError':
            if var_name := analysis_results.get('undefined_variable', ''):
                suggestions.append({
                    'type': 'variable_definition',
                    'description': f"Define variable '{var_name}' before use",
                    'priority': 'high',
                    'auto_fixable': False,
                    'suggested_code': f"# Define {var_name} here\n{var_name} = None  # Replace with appropriate value"
                })

        elif error_type == 'ImportError':
            if module_name := analysis_results.get('missing_module', ''):
                suggestions.append({
                    'type': 'install_module',
                    'description': f"Install missing module: {module_name}",
                    'priority': 'high',
                    'auto_fixable': True,
                    'command': f"pip install {module_name}"
                })

        # Add general suggestions
        suggestions.append({
            'type': 'debug_logging',
            'description': 'Add debug logging to understand the issue better',
            'priority': 'medium',
            'auto_fixable': False,
            'suggested_code': 'import logging\nlogging.debug(f"Debug info: {locals()}")'
        })

        return suggestions
        
    def calculate_confidence_score(self, analysis: Dict) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.5  # Base score
        
        # Increase confidence based on specific analysis
        if analysis['analysis_results']:
            score += 0.2
            
        if analysis['fix_suggestions']:
            score += 0.2
            
        # Increase confidence for well-known error types
        if analysis['error_type'] in self.error_patterns:
            score += 0.1
            
        return min(score, 1.0)
        
    def estimate_fix_time(self, analysis: Dict) -> str:
        """Estimate time needed to fix the error"""
        error_type = analysis['error_type']
        
        time_estimates = {
            'SyntaxError': '1-5 minutes',
            'NameError': '2-10 minutes',
            'TypeError': '5-15 minutes',
            'AttributeError': '5-15 minutes',
            'ImportError': '1-5 minutes',
            'ValueError': '10-30 minutes',
            'KeyError': '5-15 minutes',
            'IndexError': '5-15 minutes'
        }
        
        return time_estimates.get(error_type, '10-30 minutes')

def create_test_for_function(function_code: str, function_name: str, llm_interface=None) -> str:
    """
    Helper function to create tests for a specific function
    
    Args:
        function_code: Source code of the function
        function_name: Name of the function
        llm_interface: LLM interface for enhanced generation
        
    Returns:
        Generated test code
    """
    generator = AITestGenerator(llm_interface)
    analysis = generator.analyze_function(function_code, function_name)
    
    test_code = f'''"""
AI-Generated Tests for {function_name}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pytest
from unittest.mock import Mock, patch

def test_{function_name}_comprehensive():
    """Comprehensive test for {function_name}"""
'''
    
    # Generate tests based on analysis
    for test_type in analysis['test_types_needed']:
        if test_type in generator.test_patterns:
            generated_tests = generator.test_patterns[test_type](function_name, analysis)
            for test in generated_tests:
                test_code += test + '\n'
                
    return test_code

def analyze_error_with_ai(error_traceback: str, llm_interface=None) -> Dict:
    """
    Helper function to analyze errors using AI
    
    Args:
        error_traceback: Full error traceback
        llm_interface: LLM interface for enhanced analysis
        
    Returns:
        Dict with error analysis and suggestions
    """
    assistant = AIDebuggingAssistant(llm_interface)

    # Parse error traceback
    error_info = parse_error_traceback(error_traceback)

    return assistant.analyze_error(error_info)

def parse_error_traceback(traceback_str: str) -> Dict:
    """Parse error traceback to extract structured information"""
    lines = traceback_str.strip().split('\n')

    error_info = {
        'type': 'Unknown',
        'message': '',
        'file': '',
        'line': 0,
        'context': ''
    }

    # Find the last line with error type and message
    for line in reversed(lines):
        if ':' in line and any(error_type in line for error_type in ['Error', 'Exception']):
            parts = line.split(':', 1)
            error_info['type'] = parts[0].strip()
            error_info['message'] = parts[1].strip() if len(parts) > 1 else ''
            break

    # Find file and line information
    for line in lines:
        if 'File "' in line and 'line' in line:
            try:
                file_part = line.split('File "')[1].split('"')[0]
                line_part = line.split('line ')[1].split(',')[0]
                error_info['file'] = file_part
                error_info['line'] = int(line_part)
            except (IndexError, ValueError):
                pass
            break

    # Extract code context
    for line in lines:
        if (
            line.strip()
            and not line.startswith('Traceback')
            and not line.startswith('File')
            and (
                ':' not in line
                or all(
                    error_type not in line
                    for error_type in ['Error', 'Exception']
                )
            )
        ):
            error_info['context'] = line.strip()
            break

    return error_info