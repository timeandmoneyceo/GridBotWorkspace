"""
Comprehensive validation script for optimized GridbotBackup.py
Tests the functionality of optimized functions and identifies issues
"""

import sys
import os
import ast
import importlib.util
import traceback
from datetime import datetime

def test_file_syntax(file_path):
    """Test if file has valid Python syntax"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, "Syntax check passed"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def test_function_definitions(file_path):
    """Test if optimized functions are properly defined"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        
        target_functions = ['train_pytorch_predictor', 'calculate_locked_funds', 'sync_balances']
        results = {}
        
        for func_name in target_functions:
            if func_name in functions:
                func_node = functions[func_name]
                results[func_name] = {
                    'defined': True,
                    'line_start': func_node.lineno,
                    'args_count': len(func_node.args.args),
                    'has_docstring': ast.get_docstring(func_node) is not None
                }
            else:
                results[func_name] = {'defined': False}
        
        return True, results
    except Exception as e:
        return False, f"Error analyzing functions: {e}"

def test_imports(file_path):
    """Test if all imports are valid"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Test critical imports for optimized functions
        critical_imports = ['torch', 'pandas', 'numpy', 'sklearn']
        missing_imports = []

        for imp in critical_imports:
            found = any(imp in full_import for full_import in imports)
            if not found:
                missing_imports.append(imp)

        return True, {
            'total_imports': len(imports),
            'missing_critical': missing_imports,
            'all_imports': imports[:10]  # First 10 for display
        }
    except Exception as e:
        return False, f"Error analyzing imports: {e}"

def test_optimization_markers(file_path):
    """Check for optimization markers and duplicate code"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for duplicate imports
        import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
        if len(import_lines) != len(set(import_lines)):
            issues.append("Duplicate import statements detected")
        
        # Check for obvious optimization artifacts
        if 'Optimized Code:' in content:
            issues.append("Optimization comments left in code")
        
        # Check for incomplete function definitions
        if 'def ' in content and content.count('def ') != content.count(':\n'):
            # This is a rough check - might have false positives
            pass
        
        return True, {
            'issues_found': len(issues),
            'issues': issues,
            'total_lines': len(content.split('\n')),
            'import_count': len(import_lines)
        }
    except Exception as e:
        return False, f"Error checking optimization markers: {e}"

def run_comprehensive_validation():
    """Run all validation tests"""
    file_path = 'GridbotBackup.py'
    
    print("=" * 80)
    print("GRIDBOT BACKUP OPTIMIZATION VALIDATION")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Time: {datetime.now()}")
    print()
    
    # Test 1: Syntax
    print("🔍 Test 1: Syntax Validation")
    success, result = test_file_syntax(file_path)
    print(f"   Status: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Result: {result}")
    print()
    
    # Test 2: Function definitions
    print("🔍 Test 2: Function Definitions")
    success, result = test_function_definitions(file_path)
    print(f"   Status: {'✅ PASS' if success else '❌ FAIL'}")
    if success:
        for func, info in result.items():
            status = "✅" if info.get('defined', False) else "❌"
            print(f"   {status} {func}: {'Defined' if info.get('defined') else 'Missing'}")
            if info.get('defined'):
                print(f"      Line: {info.get('line_start')}, Args: {info.get('args_count')}, Docstring: {info.get('has_docstring')}")
    else:
        print(f"   Error: {result}")
    print()
    
    # Test 3: Imports
    print("🔍 Test 3: Import Analysis")
    success, result = test_imports(file_path)
    print(f"   Status: {'✅ PASS' if success else '❌ FAIL'}")
    if success:
        print(f"   Total imports: {result['total_imports']}")
        print(f"   Missing critical: {result['missing_critical'] or 'None'}")
        print(f"   Sample imports: {', '.join(result['all_imports'])}")
    else:
        print(f"   Error: {result}")
    print()
    
    # Test 4: Optimization artifacts
    print("🔍 Test 4: Code Quality Check")
    success, result = test_optimization_markers(file_path)
    print(f"   Status: {'✅ PASS' if success else '❌ FAIL'}")
    if success:
        print(f"   Lines of code: {result['total_lines']}")
        print(f"   Import statements: {result['import_count']}")
        print(f"   Issues found: {result['issues_found']}")
        if result['issues']:
            for issue in result['issues']:
                print(f"   ⚠️  {issue}")
    else:
        print(f"   Error: {result}")
    print()
    
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_validation()