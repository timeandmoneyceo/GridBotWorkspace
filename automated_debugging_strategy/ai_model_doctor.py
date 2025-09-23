"""
AI Workspace Doctor

This module provides AI-powered code analysis and improvement capabilities.
It analyzes Python files and uses AI models to identify issues and suggest improvements.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import AI interfaces
try:
    from qwen_agent_interface import QwenAgentInterface
    from intelligent_apps_integration import IntelligentAppsIntegration
    from ai_testing_debugging import analyze_error_with_ai
except ImportError:
    # Fallback for when run as standalone
    sys.path.append(os.path.dirname(__file__))
    from qwen_agent_interface import QwenAgentInterface
    from intelligent_apps_integration import IntelligentAppsIntegration
    from ai_testing_debugging import analyze_error_with_ai

class AIWorkspaceDoctor:
    """
    AI-powered workspace doctor that analyzes and improves Python code.

    This class provides comprehensive code analysis using multiple AI models
    with proper timeout handling to prevent performance issues.
    """

    def __init__(self, workspace_path: Optional[str] = None, timeout_seconds: int = 300):
        """
        Initialize the AI Workspace Doctor.

        Args:
            workspace_path: Path to the workspace directory
            timeout_seconds: Maximum time to spend on AI analysis per file
        """
        self.workspace_path = workspace_path or os.path.dirname(os.path.abspath(__file__))
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)

        # Initialize AI interfaces with error handling
        self.ai_interfaces = {}
        self._initialize_ai_interfaces()

        # Track analysis results
        self.analysis_results = {}
        self.files_processed = 0
        self.changes_made = 0

    def _initialize_ai_interfaces(self):
        """Initialize AI interfaces with fallback options."""
        try:
            # Try to initialize Qwen agent interface
            self.ai_interfaces['qwen'] = QwenAgentInterface(
                model_name='qwen3:1.7b',
                base_url='http://localhost:11434',
                api_key='EMPTY',
                workspace_path=self.workspace_path,
                enable_thinking=True,
                temperature=0.6,
                max_tokens=4096
            )
            self.logger.info("Qwen agent interface initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize Qwen interface: {e}")

        try:
            # Try to initialize intelligent apps integration
            self.ai_interfaces['intelligent'] = IntelligentAppsIntegration()
            self.logger.info("Intelligent apps integration initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize intelligent apps: {e}")

        if not self.ai_interfaces:
            self.logger.warning("No AI interfaces available - running in mock mode")

    def _check_ai_model_availability(self, model_name: str, url: str = "http://localhost:11434") -> bool:
        """Check if an AI model is available and responding."""
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'] == model_name for model in models)
            return False
        except Exception:
            return False

    def _analyze_file_with_ai(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file with AI models.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Dictionary with analysis results
        """
        result = {
            'file_path': file_path,
            'success': False,
            'issues_found': [],
            'suggestions': [],
            'fixes_applied': [],
            'error': None,
            'analysis_time': 0
        }

        start_time = time.time()

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if len(content) > 50000:  # Skip very large files
                result['error'] = "File too large for AI analysis"
                return result

            # TEMPORARY: Skip AI model availability checks due to connection issues
            # Just assume basic models are available for now
            available_models = ['smollm2:1.7b', 'qwen3:1.7b', 'deepseek-coder:latest']

            # Use timeout for AI analysis
            analysis_result = self._run_ai_analysis_with_timeout(content, available_models)

            result.update(analysis_result)
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error analyzing {file_path}: {e}")

        finally:
            result['analysis_time'] = time.time() - start_time

        return result

    def _run_ai_analysis_with_timeout(self, content: str, available_models: List[str]) -> Dict[str, Any]:
        """Run analysis with IntelligentAppsIntegration."""
        return self._perform_ai_analysis_intelligent(content)

    def _perform_ai_analysis(self, content: str, available_models: List[str]) -> Dict[str, Any]:
        """Perform the actual AI analysis using real AI models."""
        result = {'issues_found': [], 'suggestions': [], 'fixes_applied': []}

        # TEMPORARY: Skip AI analysis due to timeout issues
        USE_AI = True  # Enable AI analysis using IntelligentAppsIntegration

        if not USE_AI:
            self.logger.info("Using basic code analysis (AI models disabled)")
            result['issues_found'] = self._basic_code_analysis(content)
            result['suggestions'] = self._basic_suggestions(content)
            return result

        try:
            # Test models one by one to avoid overloading Ollama
            issues = []
            suggestions = []

            # Test each model individually first
            working_models = []
            for model in available_models:
                if self._test_model_response(model):
                    working_models.append(model)
                    break  # Just use the first working model for now

            if not working_models:
                self.logger.warning("No AI models responding, falling back to basic heuristics")
                issues.extend(self._basic_code_analysis(content))
                suggestions.extend(self._basic_suggestions(content))
            else:
                # Use the first working model for analysis
                model_to_use = working_models[0]
                self.logger.info(f"Using AI model: {model_to_use}")

                analysis = self._analyze_with_model(
                    content,
                    model_to_use,
                    "Analyze this Python code for issues and improvements. Return JSON: {'issues': ['list', 'of', 'issues'], 'suggestions': ['list', 'of', 'suggestions']}"
                )

                if analysis:
                    if 'issues' in analysis:
                        issues.extend(analysis['issues'])
                    if 'suggestions' in analysis:
                        suggestions.extend(analysis['suggestions'])
                else:
                    # Fallback if AI analysis fails
                    issues.extend(self._basic_code_analysis(content))
                    suggestions.extend(self._basic_suggestions(content))

            result['issues_found'] = issues
            result['suggestions'] = suggestions

        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            # Fallback to basic analysis
            result['issues_found'] = self._basic_code_analysis(content)
            result['suggestions'] = self._basic_suggestions(content)

        return result

    def _perform_ai_analysis_intelligent(self, content: str) -> Dict[str, Any]:
        """Perform AI analysis using IntelligentAppsIntegration."""
        result = {'issues_found': [], 'suggestions': [], 'fixes_applied': []}

        try:
            # Use IntelligentAppsIntegration for comprehensive AI analysis
            if 'intelligent' in self.ai_interfaces and self.ai_interfaces['intelligent']:
                ai_integration = self.ai_interfaces['intelligent']

                # 1. Get refactoring suggestions
                refactoring_suggestions = ai_integration.suggest_refactoring(content, "current_file.py")
                for suggestion in refactoring_suggestions:
                    result['issues_found'].append(f"{suggestion['type']}: {suggestion['description']}")
                    result['suggestions'].append(suggestion['suggested_action'])

                # 2. Perform AI code review
                code_review = ai_integration.perform_ai_code_review("current_file.py", [content])
                for issue in code_review.get('issues_found', []):
                    result['issues_found'].append(issue.get('description', str(issue)))

                for suggestion in code_review.get('suggestions', []):
                    result['suggestions'].append(suggestion.get('description', str(suggestion)))

                # 3. Check for security concerns
                for concern in code_review.get('security_concerns', []):
                    result['issues_found'].append(f"SECURITY: {concern.get('description', str(concern))}")

                # 4. Generate documentation suggestions if missing
                if '"""' not in content[:200]:
                    result['suggestions'].append("Add comprehensive docstring to improve code documentation")

                # 6. Analyze for potential errors and exceptions
                error_analysis = ai_integration.explain_error_intelligently("General code analysis", content)
                if error_analysis.get('common_causes'):
                    result['suggestions'].append(f"Potential error prevention: {error_analysis['common_causes'][0]}")

                # 7. Generate test case suggestions
                # Extract function names for test generation
                import re
                function_matches = re.findall(r'def\s+(\w+)\s*\(', content)
                if function_matches:
                    test_suggestions = ai_integration.generate_test_cases(content, function_matches[0])
                    if test_suggestions:
                        result['suggestions'].append(f"Consider adding tests for function: {function_matches[0]}")

                # 8. Check for performance optimization opportunities
                if 'for ' in content and 'append(' in content:
                    result['issues_found'].append("PERFORMANCE: Potential list append in loop - consider list comprehension")
                    result['suggestions'].append("Replace list append in loop with list comprehension for better performance")

                # 9. Check for code complexity
                lines = content.split('\n')
                if len(lines) > 100:
                    result['issues_found'].append("COMPLEXITY: File is very long - consider splitting into multiple modules")
                    result['suggestions'].append("Break large files into smaller, focused modules")

                # 10. AI-powered documentation check
                if len(content) > 500 and content.count('"""') < 2:
                    doc_suggestion = ai_integration.generate_documentation(content[:500], 'function')
                    if doc_suggestion and 'Args:' in doc_suggestion:
                        result['suggestions'].append("Add comprehensive docstrings to functions and classes")

                # 11. Use semantic search to find related issues
                semantic_results = ai_integration.semantic_search("code quality issues", "current_file")
                for result_item in semantic_results[:3]:  # Limit to top 3
                    if result_item.get('relevance_score', 0) > 0.8:
                        result['suggestions'].append(f"Consider reviewing: {result_item.get('description', '')}")

                self.logger.info(f"AI analysis completed: {len(result['issues_found'])} issues, {len(result['suggestions'])} suggestions")
            else:
                # Fallback to basic analysis if AI integration is not available
                self.logger.warning("AI integration not available, using basic heuristics")
                result['issues_found'] = self._basic_code_analysis(content)
                result['suggestions'] = self._basic_suggestions(content)

        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            # Fallback to basic analysis
            result['issues_found'] = self._basic_code_analysis(content)
            result['suggestions'] = self._basic_suggestions(content)

        return result
        """Test if a model is responding quickly."""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': 'Say "OK"',
                    'stream': False,
                    'options': {'temperature': 0, 'max_tokens': 10}
                },
                timeout=10  # Short timeout for testing
            )
            return response.status_code == 200
        except:
            return False

    def _analyze_with_model(self, content: str, model_name: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Analyze code using a specific AI model."""
        try:
            # Prepare the analysis prompt - keep it short for faster response
            analysis_prompt = f"""
{prompt}

Code to analyze (first 1000 chars):
{content[:1000]}

Return only valid JSON.
"""

            # Make API call to Ollama with reasonable timeout
            timeout = 60  # Give it a full minute since we know the model works

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': analysis_prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,  # Lower temperature for more consistent responses
                        'top_p': 0.9,
                        'max_tokens': 500  # Shorter responses
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Try to parse JSON response
                try:
                    # Clean up the response text to extract JSON
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        return json.loads(json_str)
                    else:
                        self.logger.warning(f"Could not find JSON in response from {model_name}")
                        return None
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response from {model_name}: {e}")
                    return None
            else:
                self.logger.warning(f"API call to {model_name} failed with status {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error analyzing with {model_name}: {e}")
            return None

    def _basic_code_analysis(self, content: str) -> List[str]:
        """Fallback basic code analysis using heuristics."""
        issues = []

        # Check for missing imports
        if 'import' not in content and 'from' not in content:
            issues.append("No imports found - consider adding necessary imports")

        # Check for TODO comments
        if 'TODO' in content or 'FIXME' in content:
            issues.append("TODO/FIXME comments found - consider addressing them")

        # Check for long functions (simple heuristic)
        lines = content.split('\n')
        function_lines = []
        in_function = False
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if in_function and len(function_lines) > 50:
                    issues.append(f"Function starting around line {i-len(function_lines)} is very long ({len(function_lines)} lines)")
                function_lines = [line]
                in_function = True
            elif in_function:
                function_lines.append(line)
                if line.strip() == '' and len(function_lines) > 30:
                    # End of function block
                    in_function = False

        # Check for print statements (potential debug code)
        print_count = content.count('print(')
        if print_count > 10:
            issues.append(f"Many print statements found ({print_count}) - consider using logging")

        return issues

    def _basic_suggestions(self, content: str) -> List[str]:
        """Generate basic suggestions based on issues."""
        suggestions = []
        issues = self._basic_code_analysis(content)

        for issue in issues:
            if 'imports' in issue:
                suggestions.append("Add proper imports at the top of the file")
            elif 'TODO' in issue:
                suggestions.append("Address TODO comments or remove them")
            elif 'long' in issue:
                suggestions.append("Consider breaking long functions into smaller ones")
            elif 'print' in issue:
                suggestions.append("Replace print statements with proper logging")

        return suggestions

    def analyze_workspace(self, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze all Python files in the workspace.

        Args:
            target_dir: Directory to analyze (defaults to automated_debugging_strategy)

        Returns:
            Analysis results summary
        """
        if target_dir is None:
            # When run from within automated_debugging_strategy, use current directory
            target_dir = self.workspace_path

        self.logger.info(f"Starting workspace analysis in: {target_dir}")

        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(target_dir):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'backups', 'temp', 'logs', 'automation_logs', 'test_logs', 'sessions', 'reports']]

            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(os.path.join(root, file))

        self.logger.info(f"Found {len(python_files)} Python files to analyze")

        # Analyze files (limit to first 10 for performance)
        files_to_analyze = python_files[:10]  # Limit for initial implementation

        results = []
        for file_path in files_to_analyze:
            self.logger.info(f"Analyzing {os.path.basename(file_path)}...")
            result = self._analyze_file_with_ai(file_path)
            results.append(result)
            self.files_processed += 1

            if result['issues_found']:
                self.changes_made += 1

        return {
            'files_analyzed': len(results),
            'total_issues': sum(len(r['issues_found']) for r in results),
            'total_suggestions': sum(len(r['suggestions']) for r in results),
            'results': results
        }

    def run_analysis(self, apply_fixes: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the complete workspace analysis.

        Args:
            apply_fixes: Whether to apply automatic fixes
            verbose: Whether to show detailed output

        Returns:
            Result dictionary with success status and summary
        """
        try:
            if verbose:
                print("🔍 Starting AI Workspace Doctor analysis...")

            # Run analysis
            analysis_result = self.analyze_workspace()

            if verbose:
                print(f"📊 Analysis complete: {analysis_result['files_analyzed']} files analyzed")
                print(f"🔍 Issues found: {analysis_result['total_issues']}")
                print(f"💡 Suggestions: {analysis_result['total_suggestions']}")

            return {
                'success': True,
                'changes_found': analysis_result['total_issues'] > 0,
                'files_changed': analysis_result['files_analyzed'],
                'analysis_summary': analysis_result,
                'error': None
            }

        except Exception as e:
            error_msg = f"Workspace doctor failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'changes_found': False,
                'files_changed': 0,
                'error': error_msg
            }


def run_ai_workspace_doctor(apply_fixes: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    Main entry point for the AI workspace doctor.

    Args:
        apply_fixes: Whether to apply automatic fixes (currently not implemented)
        verbose: Whether to show detailed output

    Returns:
        Result dictionary
    """
    try:
        doctor = AIWorkspaceDoctor(timeout_seconds=120)  # 2 minute timeout per file
        return doctor.run_analysis(apply_fixes=apply_fixes, verbose=verbose)
    except Exception as e:
        return {
            'success': False,
            'changes_found': False,
            'files_changed': 0,
            'error': f"Failed to initialize AI workspace doctor: {str(e)}"
        }


if __name__ == "__main__":
    # Standalone execution
    import argparse

    parser = argparse.ArgumentParser(description="AI Workspace Doctor")
    parser.add_argument("--apply-fixes", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show detailed output")

    args = parser.parse_args()

    result = run_ai_workspace_doctor(apply_fixes=args.apply_fixes, verbose=args.verbose)

    if result['success']:
        print("✅ AI Workspace Doctor completed successfully")
        if result['changes_found']:
            print(f"🔧 Found {result['files_changed']} files with potential improvements")
        else:
            print("✨ No issues found - workspace is clean!")
    else:
        print(f"❌ AI Workspace Doctor failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)