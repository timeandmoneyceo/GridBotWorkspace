#!/usr/bin/env python3
"""
LLM-Driven Python Formatting Service

This service uses local LLMs to intelligently format and fix Python code,
with traditional formatting tools as fallbacks only.
"""

import os
import sys
import logging
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Import our LLM interfaces
try:
    from .qwen_agent_interface import QwenAgentInterface
    from .qwen_transformers_interface import QwenTransformersInterface
except ImportError:
    try:
        from qwen_agent_interface import QwenAgentInterface
        from qwen_transformers_interface import QwenTransformersInterface
    except ImportError:
        QwenAgentInterface = None
        QwenTransformersInterface = None


class LLMFormattingService:
    """LLM-driven Python code formatting and indentation service"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize LLM interfaces for formatting
        self.qwen_agent = None
        self.qwen_transformers = None

        if QwenAgentInterface:
            try:
                self.qwen_agent = QwenAgentInterface(temperature=0.1)  # Low temp for consistency
            except Exception as e:
                self.logger.warning(f"Failed to initialize QwenAgentInterface: {e}")

        if QwenTransformersInterface:
            try:
                self.qwen_transformers = QwenTransformersInterface()
            except Exception as e:
                self.logger.warning(f"Failed to initialize QwenTransformersInterface: {e}")

        # Formatting tools as fallbacks only
        self.black_available = self._check_tool_availability('black')
        self.autopep8_available = self._check_tool_availability('autopep8')

        self.logger.info("[LLM-FORMAT] Service initialized")
        self.logger.info(f"[LLM-FORMAT] Qwen Agent: {bool(self.qwen_agent)}")
        self.logger.info(f"[LLM-FORMAT] Qwen Transformers: {bool(self.qwen_transformers)}")
        self.logger.info(f"[LLM-FORMAT] Black available: {self.black_available}")
        self.logger.info(f"[LLM-FORMAT] AutoPEP8 available: {self.autopep8_available}")
    
    def _check_tool_availability(self, tool_name: str) -> bool:
        """Check if a formatting tool is available"""
        try:
            result = subprocess.run([tool_name, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def fix_indentation_llm_driven(self, content: str, syntax_error: str) -> Tuple[str, bool, str]:
        """Use LLM to intelligently fix indentation issues"""
        self.logger.info("[LLM-FORMAT] Starting LLM-driven indentation fix...")
        
        # Primary strategy: Use Qwen agent for contextual analysis
        if self.qwen_agent:
            try:
                result = self._qwen_agent_fix_indentation(content, syntax_error)
                if result[1]:  # If successful
                    return result
            except Exception as e:
                self.logger.warning(f"[LLM-FORMAT] Qwen agent failed: {e}")
        
        # Secondary strategy: Use Qwen transformers
        if self.qwen_transformers:
            try:
                result = self._qwen_transformers_fix_indentation(content, syntax_error)
                if result[1]:  # If successful
                    return result
            except Exception as e:
                self.logger.warning(f"[LLM-FORMAT] Qwen transformers failed: {e}")
        
        # Fallback: Traditional formatting tools (if LLM fails)
        self.logger.info("[LLM-FORMAT] LLM methods failed, using fallback formatting...")
        return self._fallback_formatting_fix(content, syntax_error)
    
    def _qwen_agent_fix_indentation(self, content: str, syntax_error: str) -> Tuple[str, bool, str]:
        """Use Qwen agent to fix indentation with contextual understanding"""
        
        if not self.qwen_agent:
            return content, False, "qwen_agent_unavailable"
        
        prompt = f"""You are a Python code formatting expert. Fix the indentation issues in this code.

CRITICAL REQUIREMENTS:
1. Maintain all functionality - do not change logic
2. Fix indentation to follow Python standards (4 spaces per level)
3. Ensure proper nesting for if/for/while/def/class statements
4. Convert tabs to spaces consistently
5. Return ONLY the corrected code, no explanations

ERROR DETAILS:
{syntax_error}

BROKEN CODE:
{content}

CORRECTED CODE:"""

        try:
            response = self.qwen_agent._call_qwen_optimizer(prompt, enable_streaming=False)
            
            if response and len(response.strip()) > 0:
                # Extract code from response
                fixed_code = self._extract_code_from_llm_response(response)
                
                if fixed_code and len(fixed_code) > len(content) * 0.5 and self._validate_indentation_fix(content, fixed_code, syntax_error):
                    self.logger.info("[LLM-FORMAT] Qwen agent successfully fixed indentation")
                    return fixed_code, True, "qwen_agent_indentation"
            
        except Exception as e:
            self.logger.error(f"[LLM-FORMAT] Qwen agent error: {e}")
        
        return content, False, "qwen_agent_failed"
    
    def _qwen_transformers_fix_indentation(self, content: str, syntax_error: str) -> Tuple[str, bool, str]:
        """Use Qwen transformers for indentation fixing"""
        
        if not self.qwen_transformers:
            return content, False, "qwen_transformers_unavailable"
        
        prompt = f"""Fix Python indentation errors. Convert tabs to 4 spaces. Maintain logic.

Error: {syntax_error}

Code:
{content}

Fixed:"""

        try:
            response = self.qwen_transformers.generate_error_fix(
                {"error": syntax_error, "type": "indentation"},
                content,
                "temp.py"
            )
            
            if response.success and response.content:
                fixed_code = self.qwen_transformers.extract_code_from_response(response.content)
                
                if fixed_code and self._validate_indentation_fix(content, fixed_code, syntax_error):
                    self.logger.info("[LLM-FORMAT] Qwen transformers successfully fixed indentation")
                    return fixed_code, True, "qwen_transformers_indentation"
        
        except Exception as e:
            self.logger.error(f"[LLM-FORMAT] Qwen transformers error: {e}")
        
        return content, False, "qwen_transformers_failed"
    
    def _fallback_formatting_fix(self, content: str, syntax_error: str) -> Tuple[str, bool, str]:
        """Fallback to traditional formatting tools if LLM fails"""
        self.logger.info("[LLM-FORMAT] Using traditional formatting as fallback...")
        
        # Try autopep8 first (better for fixing specific issues)
        if self.autopep8_available:
            try:
                result = self._autopep8_fix(content)
                if result[1]:
                    return result[0], True, "autopep8_fallback"
            except Exception as e:
                self.logger.warning(f"[LLM-FORMAT] AutoPEP8 fallback failed: {e}")
        
        # Try black as last resort
        if self.black_available:
            try:
                result = self._black_fix(content)
                if result[1]:
                    return result[0], True, "black_fallback"
            except Exception as e:
                self.logger.warning(f"[LLM-FORMAT] Black fallback failed: {e}")
        
        return content, False, "all_formatting_failed"
    
    def _autopep8_fix(self, content: str) -> Tuple[str, bool]:
        """Use autopep8 to fix formatting"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name
            
            # Run autopep8 with indentation fixes
            result = subprocess.run([
                'autopep8', '--select=E1,W191,W292,W293', '--in-place', temp_file
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                with open(temp_file, 'r') as f:
                    fixed_content = f.read()
                os.unlink(temp_file)
                return fixed_content, True
            
            os.unlink(temp_file)
            return content, False
            
        except Exception as e:
            self.logger.error(f"[LLM-FORMAT] AutoPEP8 error: {e}")
            return content, False
    
    def _black_fix(self, content: str) -> Tuple[str, bool]:
        """Use black to fix formatting"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name
            
            # Run black
            result = subprocess.run([
                'black', '--quiet', temp_file
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                with open(temp_file, 'r') as f:
                    fixed_content = f.read()
                os.unlink(temp_file)
                return fixed_content, True
            
            os.unlink(temp_file)
            return content, False
            
        except Exception as e:
            self.logger.error(f"[LLM-FORMAT] Black error: {e}")
            return content, False
    
    def _extract_code_from_llm_response(self, response: str) -> str:
        """Extract code from LLM response, handling various formats"""
        try:
            # Remove any leading/trailing whitespace
            response = response.strip()

            # Method 1: Look for ```python code blocks (most common)
            python_pattern = r'```python\n(.*?)\n```'
            if match := re.search(python_pattern, response, re.DOTALL):
                return match.group(1).strip()

            # Method 2: Look for generic ``` code blocks
            generic_pattern = r'```\n(.*?)\n```'
            if match := re.search(generic_pattern, response, re.DOTALL):
                return match.group(1).strip()

            # Method 3: Look for code blocks without newlines after ```
            python_pattern_alt = r'```python(.*?)```'
            if match := re.search(python_pattern_alt, response, re.DOTALL):
                return match.group(1).strip()

            # Method 4: Look for generic code blocks without newlines
            generic_pattern_alt = r'```(.*?)```'
            if match := re.search(generic_pattern_alt, response, re.DOTALL):
                return match.group(1).strip()

            # Method 5: If no markdown blocks found, look for Python-like code patterns
            lines = response.split('\n')
            code_lines = []
            in_code_block = False

            for line in lines:
                stripped = line.strip()

                # Start of potential code block
                if (stripped.startswith('def ') or stripped.startswith('class ') or 
                    stripped.startswith('import ') or stripped.startswith('from ')):
                    in_code_block = True
                    code_lines.append(line)
                elif in_code_block:
                    # Continue collecting code lines
                    if (stripped and not stripped.startswith('#') and 
                        (line.startswith('    ') or line.startswith('\t') or 
                         stripped.startswith(('return', 'if', 'for', 'while', 'try', 'except', 'with')))):
                        code_lines.append(line)
                    elif not stripped:
                        code_lines.append(line)  # Empty line in code
                    else:
                        break  # End of code block

            if code_lines:
                return '\n'.join(code_lines).strip()

            # Method 6: Return original response if no patterns found
            return response

        except Exception as e:
            logger.warning(f"Error extracting code from LLM response: {e}")
            return response
    
    def _validate_indentation_fix(self, original: str, fixed: str, error: str) -> bool:
        """Validate that the indentation fix is valid"""
        try:
            # Basic syntax check
            compile(fixed, '<string>', 'exec')

            # Check that we didn't lose too much content
            if len(fixed) < len(original) * 0.5:
                return False

            # Check that key structures are preserved
            original_lines = [line.strip() for line in original.split('\n') if line.strip()]
            fixed_lines = [line.strip() for line in fixed.split('\n') if line.strip()]

            # Should have similar number of non-empty lines
            return abs(len(original_lines) - len(fixed_lines)) <= 2
        except Exception:
            return False
    
    def format_python_code_llm_driven(self, content: str, context: str = "") -> Tuple[str, bool, str]:
        """Format Python code using LLM with contextual understanding"""
        self.logger.info("[LLM-FORMAT] Starting LLM-driven code formatting...")
        
        prompt = f"""Format this Python code according to PEP 8 standards. Focus on:
1. Proper indentation (4 spaces per level)
2. Line spacing and organization
3. Import formatting
4. Function and class spacing
5. Maintain all functionality

Context: {context}

Code to format:
{content}

Formatted code:"""

        try:
            if self.qwen_agent:
                response = self.qwen_agent._call_qwen_optimizer(prompt, enable_streaming=False)
                
                if response and len(response.strip()) > 0:
                    formatted_code = self._extract_code_from_llm_response(response)
                    
                    if formatted_code and self._validate_formatting(content, formatted_code):
                        self.logger.info("[LLM-FORMAT] Successfully formatted code with LLM")
                        return formatted_code, True, "llm_formatting"
        
        except Exception as e:
            self.logger.error(f"[LLM-FORMAT] LLM formatting error: {e}")
        
        # Fallback to traditional tools
        return self._fallback_formatting_fix(content, "formatting_request")
    
    def _validate_formatting(self, original: str, formatted: str) -> bool:
        """Validate that formatting preserved functionality"""
        try:
            # Both should be valid Python
            compile(original, '<string>', 'exec')
            compile(formatted, '<string>', 'exec')

            # Content length should be similar (allowing for whitespace changes)
            return (
                len(formatted) >= len(original) * 0.7
                and len(formatted) <= len(original) * 1.5
            )
        except Exception:
            return False


# Usage example and testing
if __name__ == "__main__":
    formatter = LLMFormattingService()
    
    # Test code with indentation issues
    test_code = """def test():
\t    if True:
        print("mixed tabs and spaces")
\treturn False
"""
    
    result = formatter.fix_indentation_llm_driven(test_code, "mixed indentation error")
    print(f"Success: {result[1]}")
    print(f"Method: {result[2]}")
    print("Fixed code:")
    print(result[0])