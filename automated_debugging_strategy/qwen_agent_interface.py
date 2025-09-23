"""
Qwen Agent Orchestrator for Enhanced Automation

This module provides a Qwen3:1.7B agent orchestrator that coordinates between
deepseek-coder (for debugging tasks) and qwen3:1.7b (for                 payload = {
                    "model": self.qwen_model,
                    "prompt": prompt,
                    "stream": enable_streaming,  # Enable streaming only when requested
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_p    def test_connection(self) -> bool:
        \"\"\"Test connection to the specialized LLMs\"\"\"
        # Test deepseek debugger (disable streaming for quick tests)
        deepseek_test = self._call_deepseek_debugger("Hello, test connection.", enable_streaming=False)
        deepseek_ok = bool(deepseek_test.strip())

        # Test qwen optimizer (disable streaming for quick tests)
        qwen_test = self._call_qwen_optimizer("Hello, test connection.", enable_streaming=False)
        qwen_ok = bool(qwen_test.strip())-1  # No limit for local models
           Please provide the optimized code with explanations of the improvements made:\"\"\"

            response_content = self._call_qwen_optimizer(prompt, enable_streaming=True)      }
                }on tasks)
toPlease provide the optimized code:\"\"\"

            response_content = self._call_qwen_optimizer(prompt, enable_streaming=True)ovide comprehensive automated code analysis, debugging, and optimization.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re
import subprocess
import threading
import psutil

class QwenAgentInterface:
    """Qwen3:1.7B agent orchestrator coordinating deepseek-coder (debugging) and qwen3:1.7b (optimization)"""

    def __init__(self,
                 model_name: str = "qwen3:1.7b",
                 base_url: str = "http://localhost:11434",
                 api_key: str = "EMPTY",
                 workspace_path: str = None,
                 enable_thinking: bool = True,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 top_k: int = 20,
                 min_p: float = 0.0,
                 max_tokens: int = 32768,
                 deepseek_debugger_url: str = "http://localhost:11434",
                 deepseek_model: str = "deepseek-coder",
                 qwen_optimizer_url: str = "http://localhost:11434",
                 qwen_model: str = "smollm2:1.7b"):
        """
        Initialize Qwen agent orchestrator

        Args:
            model_name: Name of the Qwen orchestrator model
            base_url: Base URL for the Qwen orchestrator server
            api_key: API key (use EMPTY for local models)
            workspace_path: Path to the workspace for file operations
            enable_thinking: Whether to enable thinking mode for orchestrator
            temperature: Sampling temperature for orchestrator
            top_p: Top-p sampling parameter for orchestrator
            top_k: Top-k sampling parameter for orchestrator
            min_p: Minimum probability threshold for orchestrator
            max_tokens: Maximum output tokens for orchestrator
            deepseek_debugger_url: Base URL for deepseek-coder debugging LLM
            deepseek_model: Model name for deepseek-coder debugging
            qwen_optimizer_url: Base URL for qwen3:1.7b optimization LLM
            qwen_model: Model name for qwen3:1.7b optimization
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.workspace_path = workspace_path or os.getcwd()
        self.enable_thinking = enable_thinking

        # Qwen3 orchestrator parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens

        # Specialized LLM endpoints
        self.deepseek_debugger_url = deepseek_debugger_url
        self.deepseek_model = deepseek_model
        self.qwen_optimizer_url = qwen_optimizer_url
        self.qwen_model = qwen_model

        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Initialize agent (no longer needed - we use direct API calls)
        self.agent = None

        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.total_tokens = 0
        self.start_time = datetime.now()

        # Summary and reference tracking
        self.operation_summaries = []
        self.reference_files = {}

    def _call_deepseek_debugger(self, prompt: str, max_retries: int = 2, enable_streaming: bool = True) -> str:
        """Call deepseek-coder for debugging tasks with retry logic"""
        import requests
        import json
        from requests.exceptions import Timeout, ConnectionError

        # Live logging for troubleshooting
        self.logger.info("[DEEPSEEK] DEBUGGER REQUEST:")
        self.logger.info(f"[PROMPT] ({len(prompt)} chars):")
        try:
            with open('last_deepseek_prompt.txt', 'w', encoding='utf-8') as f:
                f.write(prompt)
        except Exception:
            pass
        if len(prompt) > 500:
            self.logger.info(f"[PROMPT START]\n{prompt[:200]}\n...\n{prompt[-200:]}\n[PROMPT END]")
        else:
            self.logger.info(f"[PROMPT]\n{prompt}\n[/PROMPT]")

        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.deepseek_model,
                    "prompt": prompt,
                    "stream": enable_streaming,  # Enable streaming by default for live runs
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": -1  # No limit for local models
                    }
                }

                self.logger.info(f"[SENDING] Request to DeepSeek (attempt {attempt + 1}/{max_retries + 1})...")
                response = requests.post(
                    f"{self.deepseek_debugger_url}/api/generate",
                    json=payload,
                    timeout=600,  # Increased timeout for local models (10 minutes for debugging)
                    stream=enable_streaming  # Enable streaming conditionally
                )

                if response.status_code == 200:
                    if enable_streaming:
                        # Handle streaming response with proper error handling
                        full_response = ""
                        try:
                            self.logger.info("[STREAMING] DeepSeek response streaming...")
                            for line in response.iter_lines(decode_unicode=True):
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        if 'response' in chunk:
                                            chunk_text = chunk['response']
                                            full_response += chunk_text
                                            # Stream output to terminal with live formatting
                                            print(chunk_text, end='', flush=True)
                                        if chunk.get('done', False):
                                            break
                                    except json.JSONDecodeError:
                                        continue

                            print()  # New line after streaming
                            self.logger.info(f"[SUCCESS] DEEPSEEK DEBUGGER RESPONSE ({len(full_response)} chars)")
                            return full_response
                        except Exception as e:
                            self.logger.warning(f"Streaming error: {e}, falling back to non-streaming")
                            # Fallback to non-streaming response
                            try:
                                result = response.json()
                                response_text = result.get('response', '')
                                self.logger.info(f"[SUCCESS] DEEPSEEK DEBUGGER RESPONSE ({len(response_text)} chars)")
                                return response_text
                            except Exception:
                                return ""
                    else:
                        # Handle non-streaming response
                        result = response.json()
                        response_text = result.get('response', '')
                        self.logger.info(f"[SUCCESS] DEEPSEEK DEBUGGER RESPONSE ({len(response_text)} chars)")
                        return response_text
                else:
                    self.logger.error(f"DeepSeek debugger API error: {response.status_code} - {response.text}")
                    return ""

            except (Timeout, ConnectionError) as e:
                if attempt < max_retries:
                    self.logger.warning(f"[WARN] DeepSeek debugger timeout/connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    self.logger.info("[RETRY] Retrying deepseek debugger call in 10 seconds...")
                    time.sleep(10)  # Wait before retrying
                    continue
                else:
                    self.logger.error(f"[ERROR] Failed to call deepseek debugger after {max_retries + 1} attempts: {e}")
                    return ""
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to call deepseek debugger: {e}")
                return ""

        return ""

    def _call_qwen_optimizer(self, prompt: str, max_retries: int = 2, enable_streaming: bool = True) -> str:
        """Call qwen3:1.7b for optimization tasks with retry logic"""
        import requests
        import json
        from requests.exceptions import Timeout, ConnectionError

        # Live logging for troubleshooting
        self.logger.info("[QWEN] OPTIMIZER REQUEST:")
        self.logger.info(f"[PROMPT] ({len(prompt)} chars):")
        # Save full prompt to file for inspection while avoiding double-long console logs
        try:
            with open('last_qwen_prompt.txt', 'w', encoding='utf-8') as f:
                f.write(prompt)
        except Exception:
            pass
        if len(prompt) > 500:
            self.logger.info(f"[PROMPT START]\n{prompt[:200]}\n...\n{prompt[-200:]}\n[PROMPT END]")
        else:
            self.logger.info(f"[PROMPT]\n{prompt}\n[/PROMPT]")

        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.qwen_model,
                    "prompt": prompt,
                    "stream": enable_streaming,  # Enable streaming by default for live runs
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": -1  # No limit for local models
                    }
                }

                self.logger.info(f"[SENDING] Request to Qwen Optimizer (attempt {attempt + 1}/{max_retries + 1})...")
                response = requests.post(
                    f"{self.qwen_optimizer_url}/api/generate",
                    json=payload,
                    timeout=900,  # Increased timeout for local models (15 minutes for complex optimization)
                    stream=enable_streaming  # Enable streaming conditionally
                )

                if response.status_code == 200:
                    if enable_streaming:
                        # Handle streaming response with proper error handling
                        full_response = ""
                        try:
                            self.logger.info("[STREAMING] Qwen response streaming...")
                            for line in response.iter_lines(decode_unicode=True):
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        if 'response' in chunk:
                                            chunk_text = chunk['response']
                                            full_response += chunk_text
                                            # Stream output to terminal with live formatting
                                            print(chunk_text, end='', flush=True)
                                        if chunk.get('done', False):
                                            break
                                    except json.JSONDecodeError:
                                        continue

                            print()  # New line after streaming
                            self.logger.info(f"[SUCCESS] QWEN OPTIMIZER RESPONSE ({len(full_response)} chars)")
                            return full_response
                        except Exception as e:
                            self.logger.warning(f"Streaming error: {e}, falling back to non-streaming")
                            # Fallback to non-streaming response
                            try:
                                result = response.json()
                                response_text = result.get('response', '')
                                self.logger.info(f"[SUCCESS] QWEN OPTIMIZER RESPONSE ({len(response_text)} chars)")
                                return response_text
                            except Exception:
                                return ""
                    else:
                        # Handle non-streaming response
                        result = response.json()
                        response_text = result.get('response', '')
                        self.logger.info(f"[SUCCESS] QWEN OPTIMIZER RESPONSE ({len(response_text)} chars)")
                        return response_text
                else:
                    self.logger.error(f"Qwen optimizer API error: {response.status_code} - {response.text}")
                    return ""

            except (Timeout, ConnectionError) as e:
                if attempt < max_retries:
                    self.logger.warning(f"[WARN] Qwen optimizer timeout/connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    self.logger.info("[RETRY] Retrying qwen optimizer call in 10 seconds...")
                    time.sleep(10)  # Wait before retrying
                    continue
                else:
                    self.logger.error(f"[ERROR] Failed to call qwen optimizer after {max_retries + 1} attempts: {e}")
                    return ""
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to call qwen optimizer: {e}")
                return ""

        return ""

    def _get_automation_tools(self) -> List[Dict]:
        """Get comprehensive tool set for automation tasks"""
        return [
            # File system tools
            {
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'description': 'Read the contents of a file',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'file_path': {
                                'type': 'string',
                                'description': 'Absolute path to the file to read'
                            },
                            'start_line': {
                                'type': 'integer',
                                'description': 'Starting line number (optional)',
                                'default': 1
                            },
                            'end_line': {
                                'type': 'integer',
                                'description': 'Ending line number (optional)',
                                'default': None
                            }
                        },
                        'required': ['file_path']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_directory',
                    'description': 'List contents of a directory',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'dir_path': {
                                'type': 'string',
                                'description': 'Path to the directory to list'
                            },
                            'recursive': {
                                'type': 'boolean',
                                'description': 'Whether to list recursively',
                                'default': False
                            }
                        },
                        'required': ['dir_path']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_files',
                    'description': 'Search for files matching patterns',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'pattern': {
                                'type': 'string',
                                'description': 'Glob pattern to match files'
                            },
                            'root_dir': {
                                'type': 'string',
                                'description': 'Root directory to search from',
                                'default': '.'
                            }
                        },
                        'required': ['pattern']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'grep_search',
                    'description': 'Search for text patterns in files',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'pattern': {
                                'type': 'string',
                                'description': 'Regex pattern to search for'
                            },
                            'file_pattern': {
                                'type': 'string',
                                'description': 'File pattern to search in (optional)',
                                'default': '*.py'
                            },
                            'case_sensitive': {
                                'type': 'boolean',
                                'description': 'Whether search is case sensitive',
                                'default': False
                            }
                        },
                        'required': ['pattern']
                    }
                }
            },
            # Code analysis tools
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_code',
                    'description': 'Analyze code for issues, complexity, and optimization opportunities',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'file_path': {
                                'type': 'string',
                                'description': 'Path to the file to analyze'
                            },
                            'analysis_type': {
                                'type': 'string',
                                'description': 'Type of analysis (syntax, complexity, performance, security)',
                                'enum': ['syntax', 'complexity', 'performance', 'security', 'all'],
                                'default': 'all'
                            }
                        },
                        'required': ['file_path']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'extract_functions',
                    'description': 'Extract and analyze functions from code',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'file_path': {
                                'type': 'string',
                                'description': 'Path to the Python file'
                            },
                            'include_docstrings': {
                                'type': 'boolean',
                                'description': 'Whether to include docstrings',
                                'default': True
                            }
                        },
                        'required': ['file_path']
                    }
                }
            },
            # Execution and testing tools
            {
                'type': 'function',
                'function': {
                    'name': 'run_code',
                    'description': 'Execute Python code and capture output',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'code': {
                                'type': 'string',
                                'description': 'Python code to execute'
                            },
                            'timeout': {
                                'type': 'integer',
                                'description': 'Execution timeout in seconds',
                                'default': 30
                            },
                            'capture_output': {
                                'type': 'boolean',
                                'description': 'Whether to capture stdout/stderr',
                                'default': True
                            }
                        },
                        'required': ['code']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'run_tests',
                    'description': 'Run unit tests and return results',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'test_file': {
                                'type': 'string',
                                'description': 'Path to test file or directory'
                            },
                            'test_pattern': {
                                'type': 'string',
                                'description': 'Test discovery pattern',
                                'default': 'test_*.py'
                            },
                            'verbose': {
                                'type': 'boolean',
                                'description': 'Verbose test output',
                                'default': True
                            }
                        },
                        'required': ['test_file']
                    }
                }
            },
            # System monitoring tools
            {
                'type': 'function',
                'function': {
                    'name': 'get_system_info',
                    'description': 'Get system information and resource usage',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'include_processes': {
                                'type': 'boolean',
                                'description': 'Include process information',
                                'default': False
                            }
                        }
                    }
                }
            },
            # Summary and reference tools
            {
                'type': 'function',
                'function': {
                    'name': 'create_summary',
                    'description': 'Create a summary of operations and results',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'operation_type': {
                                'type': 'string',
                                'description': 'Type of operation (debug, optimize, analyze)'
                            },
                            'results': {
                                'type': 'object',
                                'description': 'Results to summarize'
                            },
                            'save_to_file': {
                                'type': 'boolean',
                                'description': 'Whether to save summary to file',
                                'default': True
                            }
                        },
                        'required': ['operation_type', 'results']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'update_reference',
                    'description': 'Update reference files with new information',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'reference_type': {
                                'type': 'string',
                                'description': 'Type of reference (code_patterns, error_solutions, optimizations)'
                            },
                            'data': {
                                'type': 'object',
                                'description': 'Data to add to reference'
                            },
                            'merge': {
                                'type': 'boolean',
                                'description': 'Whether to merge with existing data',
                                'default': True
                            }
                        },
                        'required': ['reference_type', 'data']
                    }
                }
            }
        ]

    def test_connection(self) -> bool:
        """Test connection to the specialized LLMs"""
        # Test deepseek debugger (disable streaming for connection test)
        deepseek_test = self._call_deepseek_debugger("Hello, test connection.", enable_streaming=False)
        deepseek_ok = bool(deepseek_test.strip())

        # Test qwen optimizer (disable streaming for connection test)
        qwen_test = self._call_qwen_optimizer("Hello, test connection.", enable_streaming=False)
        qwen_ok = bool(qwen_test.strip())

        if deepseek_ok and qwen_ok:
            self.logger.info("Both specialized LLMs connected successfully")
            return True
        elif deepseek_ok:
            self.logger.warning("DeepSeek debugger connected, but Qwen optimizer failed")
            return True  # At least one LLM works
        elif qwen_ok:
            self.logger.warning("Qwen optimizer connected, but DeepSeek debugger failed")
            return True  # At least one LLM works
        else:
            self.logger.error("Both specialized LLMs failed connection test")
            return False

    def generate_function_fix(self, error_dict: Dict, function_info: Dict, target_file: str) -> 'LLMResponse':
        """Generate a fix for a function-level error"""
        return self._generate_code_fix("function_fix", error_dict, function_info, target_file)

    def generate_targeted_code_fix(self, error_dict: Dict, target_file: str) -> 'LLMResponse':
        """Generate a targeted code fix"""
        return self._generate_code_fix("targeted_fix", error_dict, None, target_file)

    def generate_optimization_suggestions(self, code_context: Dict, target_file: str) -> 'LLMResponse':
        """Generate optimization suggestions"""
        return self._generate_code_fix("optimization", code_context, None, target_file)

    def generate_optimization(self, code_snippet: str, context: str) -> 'LLMResponse':
        """Generate optimization for code using qwen3:1.7b optimizer"""
        try:
            self.request_count += 1

            prompt = f"""You are an expert Python developer optimizing code for performance and efficiency.

CODE TO OPTIMIZE:
{code_snippet}

CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze the code for performance bottlenecks and inefficiencies
2. Provide specific optimizations that improve execution speed, memory usage, or code clarity
3. Focus on algorithmic improvements, data structure optimizations, and Python best practices
4. Maintain the same functionality while improving performance
5. If no meaningful optimizations are possible, respond with "NO_OPTIMIZATIONS_NEEDED"

Please provide the optimized code with explanations of the improvements made:"""

            if response_content := self._call_qwen_optimizer(
                prompt, enable_streaming=True
            ):
                self.success_count += 1

                # Extract code from response
                optimized_code = self.extract_code_from_response(response_content)

                return LLMResponse(
                    success=True,
                    content=response_content,
                    extracted_code=optimized_code,
                    tokens_used=len(response_content.split())
                )
            else:
                return LLMResponse(success=False, content="", error="No response from qwen optimizer")
        except Exception as e:
            self.logger.error(f"Qwen optimization failed: {e}")
            return LLMResponse(success=False, content="", error=f'Qwen optimizer error: {str(e)}')

    def generate_targeted_optimization(self, code_snippet: str, performance_issues: List[str]) -> 'LLMResponse':
        """Generate targeted optimization for specific performance issues using qwen3:1.7b"""
        try:
            self.request_count += 1

            issues_text = '\n'.join(f'- {issue}' for issue in performance_issues)

            prompt = f"""You are an expert Python developer optimizing code for specific performance issues.

CODE TO OPTIMIZE:
{code_snippet}

IDENTIFIED PERFORMANCE ISSUES:
{issues_text}

INSTRUCTIONS:
1. Focus on the specific performance issues listed above
2. Provide targeted optimizations that address each issue
3. Maintain code functionality and readability
4. Explain the optimization benefits
5. If issues cannot be meaningfully addressed, respond with "NO_OPTIMIZATIONS_NEEDED"

Please provide the optimized code:"""

            if response_content := self._call_qwen_optimizer(
                prompt, enable_streaming=True
            ):
                self.success_count += 1

                # Extract code from response
                optimized_code = self.extract_code_from_response(response_content)

                return LLMResponse(
                    success=True,
                    content=response_content,
                    extracted_code=optimized_code,
                    tokens_used=len(response_content.split())
                )
            else:
                return LLMResponse(success=False, content="", error="No response from qwen optimizer")
        except Exception as e:
            self.logger.error(f"Qwen targeted optimization failed: {e}")
            return LLMResponse(success=False, content="", error=f'Qwen optimizer error: {str(e)}')

    def _generate_code_fix(self, fix_type: str, context_dict: Dict, function_info: Optional[Dict], target_file: str) -> 'LLMResponse':
        """Internal method to generate code fixes using appropriate specialized LLM"""
        try:
            self.request_count += 1

            # Build comprehensive prompt based on fix type
            if fix_type == "function_fix":
                prompt = self._build_function_fix_prompt(context_dict, function_info, target_file)
                response_content = self._call_deepseek_debugger(prompt)
            elif fix_type == "targeted_fix":
                prompt = self._build_targeted_fix_prompt(context_dict, target_file)
                response_content = self._call_deepseek_debugger(prompt)
            elif fix_type == "optimization":
                prompt = self._build_optimization_prompt(context_dict, target_file)
                response_content = self._call_qwen_optimizer(prompt)
            else:
                prompt = f"Analyze and fix the following code issue: {json.dumps(context_dict, indent=2)}"
                response_content = self._call_deepseek_debugger(prompt)

            if response_content:
                self.success_count += 1

                # Extract code from response
                fixed_code = self.extract_code_from_response(response_content)

                # Create summary
                self._create_operation_summary(fix_type, context_dict, fixed_code, target_file)

                return LLMResponse(
                    success=True,
                    content=response_content,
                    extracted_code=fixed_code,
                    tokens_used=len(response_content.split())
                )
            else:
                return LLMResponse(success=False, content="", error="No response from specialized LLM")
        except Exception as e:
            self.logger.error(f"Specialized LLM {fix_type} failed: {e}")
            return LLMResponse(success=False, content="", error=f'Specialized LLM error: {str(e)}')

    def _build_function_fix_prompt(self, error_dict: Dict, function_info: Dict, target_file: str) -> str:
        """Build prompt for function-level fixes"""
        return f"""You are an expert Python developer fixing code issues. Analyze this function error and provide a corrected version.

ERROR DETAILS:
- Type: {error_dict.get('error_type', 'Unknown')}
- Message: {error_dict.get('error_message', 'Unknown')}
- File: {target_file}
- Function: {getattr(function_info, 'name', 'Unknown')}

CODE CONTEXT:
{error_dict.get('code_context', '')}

INSTRUCTIONS:
1. Analyze the error and understand what needs to be fixed
2. Provide ONLY the corrected function code
3. Do not include line numbers, markers, or explanations
4. Ensure the code is syntactically correct and follows Python best practices
5. Maintain the same function signature and overall structure

Please provide the corrected function code:"""

    def _build_targeted_fix_prompt(self, error_dict: Dict, target_file: str) -> str:
        """Build prompt for targeted code fixes"""
        return f"""You are an expert Python developer fixing code issues. Analyze this error and provide a targeted fix.

ERROR DETAILS:
- Type: {error_dict.get('error_type', 'Unknown')}
- Message: {error_dict.get('error_message', 'Unknown')}
- File: {target_file}
- Line: {error_dict.get('line_number', 'Unknown')}

CODE CONTEXT:
{error_dict.get('targeted_code', '')}

INSTRUCTIONS:
1. Focus on the specific error at line {error_dict.get('line_number')}
2. Provide ONLY the corrected code block
3. Do not include line numbers or markers
4. Fix only what's necessary to resolve the error
5. Maintain proper indentation and structure

Please provide the corrected code:"""

    def _build_optimization_prompt(self, code_context: Dict, target_file: str) -> str:
        """Build prompt for code optimization"""
        return f"""You are an expert Python developer optimizing code for performance and efficiency.

CODE TO OPTIMIZE:
File: {target_file}
Context: {json.dumps(code_context, indent=2)}

INSTRUCTIONS:
1. Analyze the code for performance bottlenecks
2. Suggest specific optimizations (algorithm improvements, data structure changes, etc.)
3. Focus on computational efficiency, memory usage, and execution speed
4. Provide concrete code changes with explanations
5. Consider trade-offs between readability and performance

Please provide optimization suggestions with specific code changes:"""

    def extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code from Qwen agent response"""
        if not response:
            return None

        # Look for code blocks in markdown format
        code_pattern = r'```python\s*(.*?)\s*```'
        if match := re.search(code_pattern, response, re.DOTALL):
            return match[1].strip()

        # Look for code blocks without language specifier
        code_pattern = r'```\s*(.*?)\s*```'
        if match := re.search(code_pattern, response, re.DOTALL):
            code = match.group(1).strip()
            # Check if it looks like Python code
            if any(keyword in code for keyword in ['def ', 'class ', 'import ', 'if ', 'for ', 'while ']):
                return code

        # If no code blocks found, try to extract direct code
        lines = response.split('\n')
        code_lines = []

        in_code = False
        for line in lines:
            # Skip markdown headers and explanations
            if line.startswith('#') or line.startswith('Here') or line.startswith('The'):
                continue
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # Likely start of code
                in_code = True

            if in_code:
                code_lines.append(line)

        return '\n'.join(code_lines).strip() if code_lines else None

    def _create_operation_summary(self, operation_type: str, context: Dict, result: str, target_file: str):
        """Create and store operation summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': operation_type,
            'target_file': target_file,
            'context': context,
            'result_length': len(result) if result else 0,
            'success': bool(result)
        }

        self.operation_summaries.append(summary)

        # Save to file periodically
        if len(self.operation_summaries) % 10 == 0:
            self._save_summaries_to_file()

    def _save_summaries_to_file(self):
        """Save operation summaries to file"""
        summary_file = os.path.join(self.workspace_path, 'qwen_agent_summaries.json')

        try:
            # Load existing summaries
            existing_summaries = []
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_summaries = json.load(f)

            # Merge with new summaries
            existing_summaries.extend(self.operation_summaries)
            self.operation_summaries = []

            # Save updated summaries
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(existing_summaries, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(existing_summaries)} operation summaries to {summary_file}")

        except Exception as e:
            self.logger.error(f"Failed to save summaries: {e}")

    def update_reference_file(self, reference_type: str, data: Dict):
        """Update reference files with new information"""
        ref_file = os.path.join(self.workspace_path, f'qwen_agent_references_{reference_type}.json')

        try:
            # Load existing references
            existing_data = {}
            if os.path.exists(ref_file):
                with open(ref_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            # Merge new data
            if reference_type not in existing_data:
                existing_data[reference_type] = []

            existing_data[reference_type].append({
                'timestamp': datetime.now().isoformat(),
                'data': data
            })

            # Save updated references
            with open(ref_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Updated reference file {ref_file}")

        except Exception as e:
            self.logger.error(f"Failed to update reference file: {e}")

    def get_status_summary(self) -> Dict:
        """Get status summary of the Qwen agent orchestrator"""
        uptime = datetime.now() - self.start_time

        # Test LLM connections (disable streaming for quick tests)
        deepseek_test = self._call_deepseek_debugger("test", enable_streaming=False)
        qwen_test = self._call_qwen_optimizer("test", enable_streaming=False)

        return {
            'orchestrator_model': self.model_name,
            'orchestrator_url': self.base_url,
            'deepseek_debugger_url': self.deepseek_debugger_url,
            'deepseek_model': self.deepseek_model,
            'qwen_optimizer_url': self.qwen_optimizer_url,
            'qwen_model': self.qwen_model,
            'health_status': 'healthy' if (deepseek_test and qwen_test) else 'degraded',
            'deepseek_status': 'connected' if deepseek_test else 'disconnected',
            'qwen_status': 'connected' if qwen_test else 'disconnected',
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'success_rate': (self.success_count / self.request_count * 100) if self.request_count > 0 else 0,
            'total_tokens': self.total_tokens,
            'uptime_seconds': uptime.total_seconds(),
            'enable_thinking': self.enable_thinking,
            'temperature': self.temperature,
            'architecture': 'orchestrator_api_calls'
        }

    def log_status_summary(self):
        """Log a detailed status summary"""
        status = self.get_status_summary()

        self.logger.info("=" * 60)
        self.logger.info("QWEN AGENT ORCHESTRATOR STATUS SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Orchestrator Model: {status['orchestrator_model']}")
        self.logger.info(f"Orchestrator URL: {status['orchestrator_url']}")
        self.logger.info(f"DeepSeek Debugger: {status['deepseek_model']} @ {status['deepseek_debugger_url']} ({status['deepseek_status']})")
        self.logger.info(f"Qwen Optimizer: {status['qwen_model']} @ {status['qwen_optimizer_url']} ({status['qwen_status']})")
        self.logger.info(f"Overall Health: {status['health_status']}")
        self.logger.info(f"Requests: {status['total_requests']} total, {status['successful_requests']} successful")
        self.logger.info(f"Success Rate: {status['success_rate']:.1f}%")
        self.logger.info(f"Thinking Mode: {status['enable_thinking']}")
        self.logger.info(f"Temperature: {status['temperature']}")
        self.logger.info(f"Architecture: {status['architecture']}")
        self.logger.info("=" * 60)


class LLMResponse:
    """Response container for LLM operations"""
    def __init__(self, success: bool, content: str, error: str = "", extracted_code: str = "", tokens_used: int = 0):
        self.success = success
        self.content = content
        self.error = error
        self.extracted_code = extracted_code
        self.tokens_used = tokens_used
