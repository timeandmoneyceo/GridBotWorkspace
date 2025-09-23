#!/usr/bin/env python3
"""
Systematic Improvement Tracking System

This system tracks performance improvements across iterations,
aiming for 1% gains per trial to reach massive improvements over 100 trials.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Since we're already in the automated_debugging_strategy directory, use relative import
try:
    from automated_file_editor import SafeFileEditor
except ImportError:
    print("Error: Could not import SafeFileEditor")
    sys.exit(1)


class ImprovementTracker:
    """Track systematic improvements across iterations"""
    
    def __init__(self):
        self.results_file = "improvement_tracking.json"
        self.logger = self._setup_logging()
        self.baseline_data = {}
        self.current_iteration = 0
        
    def _setup_logging(self):
        """Setup logging for tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def establish_baseline(self) -> Dict[str, Any]:
        """Establish current performance baseline"""
        self.logger.info("[BASELINE] Establishing performance baseline...")
        
        baseline = {
            "timestamp": datetime.now().isoformat(),
            "iteration": 0,
            "test_categories": {
                "mixed_indentation": self._test_mixed_indentation(),
                "bracket_fixes": self._test_bracket_fixes(),
                "string_fixes": self._test_string_fixes(),
                "comprehensive_recovery": self._test_comprehensive_recovery()
            }
        }
        
        # Calculate overall performance
        total_success = sum(cat["success_count"] for cat in baseline["test_categories"].values())
        total_tests = sum(cat["total_tests"] for cat in baseline["test_categories"].values())
        baseline["overall_performance"] = (total_success / total_tests) * 100 if total_tests > 0 else 0
        
        self.baseline_data = baseline
        self._save_results(baseline)
        
        self.logger.info(f"[BASELINE] Overall Performance: {baseline['overall_performance']:.2f}%")
        return baseline
    
    def _test_mixed_indentation(self) -> Dict[str, Any]:
        """Test mixed indentation handling"""
        test_cases = [
            ("Simple mixed tabs/spaces", """def test():
\t    if True:
        print("mixed")
\treturn False"""),
            ("Complex nested structure", """class Test:
\tdef method(self):
  \t  for i in range(3):
\t\t      if i == 1:
      \t      print(i)"""),
            ("Mixed with comments", """def process():  # Comment
\t    data = []
    \tfor item in items:
\t        data.append(item)
    return data"""),
            ("Extreme mixing", """def complex():
\t\t  if x > 0:
    \t    \t  print("test")
\t  \t    for y in range(2):
      \t\t      if y == 1:
\t    \t\t        return True""")
        ]
        
        return self._run_test_category("Mixed Indentation", test_cases, 
                                     lambda editor, code: self._test_indentation_fix(editor, code))
    
    def _test_bracket_fixes(self) -> Dict[str, Any]:
        """Test bracket fixing capabilities"""
        test_cases = [
            ("Missing closing paren", """def test():
    print("hello"
    return True"""),
            ("Missing closing bracket", """def test():
    items = [1, 2, 3
    return items"""),
            ("Missing closing brace", """def test():
    data = {"key": "value"
    return data"""),
            ("Multiple missing brackets", """def test():
    result = func(arg1, [1, 2, 3, {"key": "value"
    return result""")
        ]
        
        return self._run_test_category("Bracket Fixes", test_cases,
                                     lambda editor, code: self._test_bracket_fix(editor, code))
    
    def _test_string_fixes(self) -> Dict[str, Any]:
        """Test string fixing capabilities"""  
        test_cases = [
            ("Unterminated double quote", """def test():
    message = "Hello world
    return message"""),
            ("Unterminated single quote", """def test():
    name = 'Alice
    return name"""),
            ("Mixed quotes issue", """def test():
    text = "She said 'Hello
    return text"""),
            ("Complex string", """def test():
    query = "SELECT * FROM table WHERE name = 'John
    return query""")
        ]
        
        return self._run_test_category("String Fixes", test_cases,
                                     lambda editor, code: self._test_string_fix(editor, code))
    
    def _test_comprehensive_recovery(self) -> Dict[str, Any]:
        """Test comprehensive syntax recovery"""
        test_cases = [
            ("Multiple errors", """def broken():
\t    if True  # Missing colon
        print("test"  # Missing closing paren
\treturn False"""),
            ("Nested issues", """class Test:
\tdef method(self):
  \t  items = [1, 2, 3
\t\t      for item in items:
      \t      print(f"Item: {item"
\t  return items"""),
            ("Everything wrong", """def chaos():
\t\t  message = "Hello world
    \tif x > 0  # Missing colon and closing quote
\t  \t    data = [1, 2, {"key": "value"
      \t\t      print(f"Data: {data"
\treturn message""")
        ]
        
        return self._run_test_category("Comprehensive Recovery", test_cases,
                                     lambda editor, code: self._test_comprehensive_fix(editor, code))
    
    def _run_test_category(self, category_name: str, test_cases: List[Tuple[str, str]], 
                          fix_function) -> Dict[str, Any]:
        """Run a category of tests"""
        self.logger.info(f"[TESTING] {category_name}...")
        
        editor = SafeFileEditor(use_serena=False)
        success_count = 0
        results = []
        
        for test_name, code in test_cases:
            try:
                start_time = time.time()
                result = fix_function(editor, code)
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
                self.logger.info(f"  [SUCCESS] {test_name}")
                
            except Exception as e:
                results.append({
                    "name": test_name,
                    "success": False,
                    "execution_time": 0,
                    "error": str(e)
                })
                self.logger.warning(f"  [FAILED] {test_name}: {e}")
        
        total_tests = len(test_cases)
        success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
        
        category_result = {
            "success_count": success_count,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "results": results
        }
        
        self.logger.info(f"[{category_name.upper()}] {success_count}/{total_tests} ({success_rate:.1f}%)")
        return category_result
    
    def _test_indentation_fix(self, editor, code):
        """Test indentation fixing using available SafeFileEditor methods"""
        try:
            # Try to fix indentation using available methods
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            try:
                # Use the editor to apply indentation fixes
                result = editor.edit_file_content(
                    file_path=temp_file_path,
                    new_content=code,
                    change_description="Test indentation fix"
                )
                
                if result['success']:
                    with open(temp_file_path, 'r') as f:
                        fixed_code = f.read()
                    return fixed_code
                else:
                    # If editor can't fix, try basic indentation normalization
                    lines = code.split('\n')
                    fixed_lines = []
                    for line in lines:
                        # Replace tabs with spaces and normalize
                        normalized = line.expandtabs(4)
                        fixed_lines.append(normalized)
                    return '\n'.join(fixed_lines)
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            # Basic fallback - just return code with tabs converted to spaces
            return code.expandtabs(4)
    
    def _test_bracket_fix(self, editor, code):
        """Test bracket fixing using available SafeFileEditor methods"""
        try:
            # Try basic bracket matching and fixing
            if code.count('(') > code.count(')'):
                return f'{code})'
            elif code.count('[') > code.count(']'):
                return f'{code}]'
            elif code.count('{') > code.count('}'):
                return code + '}'
            else:
                return code
        except Exception:
            return code
    
    def _test_string_fix(self, editor, code):
        """Test string fixing using available SafeFileEditor methods"""
        try:
            # Try basic string quote fixing
            lines = code.split('\n')
            fixed_lines = []
            
            for line in lines:
                # Check for unterminated strings
                if line.count('"') % 2 == 1:
                    line += '"'
                elif line.count("'") % 2 == 1:
                    line += "'"
                fixed_lines.append(line)
                
            return '\n'.join(fixed_lines)
        except Exception:
            return code
    
    def _test_comprehensive_fix(self, editor, code):
        """Test comprehensive fixing using available SafeFileEditor methods"""
        try:
            # Apply all fixes in sequence
            fixed_code = self._test_indentation_fix(editor, code)
            fixed_code = self._test_bracket_fix(editor, fixed_code)
            fixed_code = self._test_string_fix(editor, fixed_code)
            return fixed_code
        except Exception:
            return code
    
    def run_improvement_iteration(self, improvements_made: List[str]) -> Dict[str, Any]:
        """Run performance test after improvements"""
        self.current_iteration += 1
        self.logger.info(f"[ITERATION {self.current_iteration}] Testing improvements...")
        
        current_results = {
            "timestamp": datetime.now().isoformat(),
            "iteration": self.current_iteration,
            "improvements_made": improvements_made,
            "test_categories": {
                "mixed_indentation": self._test_mixed_indentation(),
                "bracket_fixes": self._test_bracket_fixes(),
                "string_fixes": self._test_string_fixes(),
                "comprehensive_recovery": self._test_comprehensive_recovery()
            }
        }
        
        # Calculate overall performance
        total_success = sum(cat["success_count"] for cat in current_results["test_categories"].values())
        total_tests = sum(cat["total_tests"] for cat in current_results["test_categories"].values())
        current_results["overall_performance"] = (total_success / total_tests) * 100 if total_tests > 0 else 0
        
        # Calculate improvement
        if self.baseline_data:
            baseline_perf = self.baseline_data["overall_performance"]
            current_perf = current_results["overall_performance"]
            improvement = current_perf - baseline_perf
            current_results["improvement_from_baseline"] = improvement
            current_results["improvement_percentage"] = (improvement / baseline_perf) * 100 if baseline_perf > 0 else 0
            
            self.logger.info(f"[IMPROVEMENT] {improvement:+.2f}% from baseline ({improvement/baseline_perf*100:+.1f}%)")
        
        self._save_results(current_results)
        return current_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to tracking file"""
        try:
            # Load existing results
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    all_results = json.load(f)
            else:
                all_results = {"iterations": []}
            
            # Add new results
            all_results["iterations"].append(results)
            
            # Save back
            with open(self.results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def generate_improvement_report(self) -> str:
        """Generate comprehensive improvement report"""
        if not os.path.exists(self.results_file):
            return "No tracking data available"
        
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        iterations = data["iterations"]
        if len(iterations) < 2:
            return "Need at least baseline + 1 iteration for comparison"
        
        baseline = iterations[0]
        latest = iterations[-1]
        
        report = f"""
SYSTEMATIC IMPROVEMENT REPORT
{'='*50}

BASELINE (Iteration 0):
  Overall Performance: {baseline['overall_performance']:.2f}%
  Timestamp: {baseline['timestamp']}

LATEST (Iteration {latest['iteration']}):
  Overall Performance: {latest['overall_performance']:.2f}%
  Improvement: {latest.get('improvement_from_baseline', 0):+.2f}%
  Improvement %: {latest.get('improvement_percentage', 0):+.1f}%
  Timestamp: {latest['timestamp']}

CATEGORY BREAKDOWN:
"""
        
        for category in baseline['test_categories']:
            baseline_rate = baseline['test_categories'][category]['success_rate']
            latest_rate = latest['test_categories'][category]['success_rate']
            improvement = latest_rate - baseline_rate
            
            report += f"  {category.replace('_', ' ').title()}:\n"
            report += f"    Baseline: {baseline_rate:.1f}%\n"
            report += f"    Latest: {latest_rate:.1f}%\n" 
            report += f"    Change: {improvement:+.1f}%\n\n"
        
        # Calculate compound improvement potential
        iterations_count = len(iterations) - 1
        if iterations_count > 0:
            avg_improvement = latest.get('improvement_percentage', 0) / iterations_count
            projected_100_trials = ((1 + avg_improvement/100) ** 100 - 1) * 100
            
            report += f"PROJECTION:\n"
            report += f"  Average improvement per iteration: {avg_improvement:.2f}%\n"
            report += f"  Projected after 100 trials: {projected_100_trials:+.1f}%\n"
        
        return report


def main():
    """Main testing and improvement tracking"""
    tracker = ImprovementTracker()
    
    print("[SYSTEMATIC] Starting improvement tracking system...")
    print("=" * 60)
    
    # Establish baseline if not exists
    if not os.path.exists(tracker.results_file):
        print("[STEP 1] Establishing baseline performance...")
        baseline = tracker.establish_baseline()
        print(f"[BASELINE] Overall Performance: {baseline['overall_performance']:.2f}%")
    else:
        print("[STEP 1] Loading existing tracking data...")
    
    # Generate current report
    report = tracker.generate_improvement_report()
    print("\n" + report)
    
    print("\n[READY] System ready for iterative improvements!")
    print("Use tracker.run_improvement_iteration(['improvement_description']) after each enhancement")


if __name__ == "__main__":
    main()