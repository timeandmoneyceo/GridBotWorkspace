"""
Debug Automation Orchestrator

This module orchestrates the automated debugging process by running files,
parsing errors, getting fixes from LLM, and applying them automatically.
"""

import os
import subprocess
import sys
import time
import logging
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import threading

try:
    # Try relative imports first (when run as module)
    from .debug_log_parser import DebugLogParser, ErrorInfo
    from .qwen_agent_interface import QwenAgentInterface, LLMResponse
    from .automated_file_editor import SafeFileEditor, EditResult
    from .function_extractor import FunctionExtractor
except ImportError:
    # Fall back to absolute imports (when run as script)
    from debug_log_parser import DebugLogParser, ErrorInfo
    from qwen_agent_interface import QwenAgentInterface, LLMResponse
    from automated_file_editor import SafeFileEditor, EditResult
    from function_extractor import FunctionExtractor

@dataclass
class DebugSession:
    """Container for debug session information"""
    target_file: str
    session_id: str
    start_time: datetime
    errors_fixed: int = 0
    total_errors: int = 0
    iterations: int = 0
    max_iterations: int = 10
    success: bool = False
    final_status: str = ""

class DebugAutomationOrchestrator:
    """Main orchestrator for automated debugging process"""
    
    def __init__(self, 
                 llm_interface: QwenAgentInterface = None,
                 file_editor: SafeFileEditor = None,
                 log_parser: DebugLogParser = None,
                 max_iterations: int = 10,
                 timeout_per_run: int = 60):
        
        self.llm_interface = llm_interface or QwenAgentInterface()
        self.file_editor = file_editor or SafeFileEditor()
        self.log_parser = log_parser or DebugLogParser()
        self.function_extractor = FunctionExtractor()
        self.max_iterations = max_iterations
        self.timeout_per_run = timeout_per_run
        
        # Error deduplication tracking
        self.processed_errors = set()  # Track processed error signatures
        self.error_attempt_counts = {}  # Track how many times each error has been attempted
        
        # Section-aware debugging state
        self.massive_function_sections = {}
        self.vscode_debugger_active = False
        self.debugger_breakpoints = []
        
        self.setup_logging()
        self.session_history = []
    
    def _generate_error_signature(self, error_info: ErrorInfo) -> str:
        """Generate a unique signature for an error to detect duplicates"""
        return f"{error_info.error_type}:{error_info.error_message}:{error_info.file_path}:{error_info.line_number}"
    
    def _is_duplicate_error(self, error_info: ErrorInfo) -> bool:
        """Check if this error has already been processed"""
        signature = self._generate_error_signature(error_info)
        return signature in self.processed_errors
    
    def _mark_error_processed(self, error_info: ErrorInfo):
        """Mark an error as processed to prevent duplicate handling"""
        signature = self._generate_error_signature(error_info)
        self.processed_errors.add(signature)
        self.error_attempt_counts[signature] = self.error_attempt_counts.get(signature, 0) + 1
        self.logger.info(f"Marked error as processed: {signature} (attempt #{self.error_attempt_counts[signature]})")
    
    def setup_logging(self):
        """Setup logging for the orchestrator without duplicating root handlers"""
        # If the root logger is already configured (by the master pipeline),
        # don't call basicConfig again. Just attach a file handler to this module logger.
        root = logging.getLogger()
        module_logger = logging.getLogger(__name__)

        # Ensure module logger has only one file handler and propagates to root
        module_logger.propagate = True
        module_logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicates
        for h in module_logger.handlers[:]:
            module_logger.removeHandler(h)

        # Always add a file handler for module-specific logs
        file_handler = logging.FileHandler('debug_orchestrator.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        module_logger.addHandler(file_handler)

        # Only add a console handler if root has no handlers (standalone run)
        if not root.handlers:
            console_handler = logging.StreamHandler()
            if hasattr(console_handler.stream, 'reconfigure'):
                try:
                    console_handler.stream.reconfigure(encoding='utf-8')
                except Exception:
                    pass
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            module_logger.addHandler(console_handler)

        self.logger = module_logger
    
    def run_file_with_debug(self, file_path: str, python_executable: str = "python") -> Tuple[str, str, int]:
        """Run a Python file and capture output and errors"""
        try:
            # Detect if this is a continuous application
            is_continuous = self.is_continuous_application(file_path)
            
            # Use different timeout strategies
            if is_continuous:
                # For continuous apps, use a shorter timeout to check startup
                startup_timeout = 30  # 30 seconds to check if it starts properly
                self.logger.info(f"Running continuous application {file_path} with startup timeout: {startup_timeout}s")
            else:
                # For regular scripts, use full timeout
                startup_timeout = self.timeout_per_run
                self.logger.info(f"Running script {file_path} with timeout: {startup_timeout}s")
            
            # Run the file with debug output
            cmd = [python_executable, "-u", file_path]  # -u for unbuffered output
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(file_path) or "."
            )
            
            # Wait for process with appropriate timeout
            try:
                stdout, stderr = process.communicate(timeout=startup_timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                
                if is_continuous:
                    # For continuous applications, timeout is expected - not an error
                    self.logger.info(f"Continuous application {file_path} is running properly (timed out as expected after {startup_timeout}s)")
                    returncode = 0  # Treat as successful
                else:
                    # For regular scripts, timeout indicates a problem
                    returncode = -1
                    self.logger.warning(f"Script {file_path} timed out after {startup_timeout} seconds")
            
            return stdout, stderr, returncode
            
        except Exception as e:
            error_msg = f"Error running file: {e}"
            self.logger.error(error_msg)
            return "", error_msg, -1
    
    def extract_file_context(self, file_path: str, error_info: ErrorInfo, 
                           context_lines: int = 10) -> str:
        """Extract relevant code context around an error"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Handle special case where line_number is 0 (general syntax error)
            if error_info.line_number <= 0:
                # For general syntax errors, return first 20 lines
                context_lines_list = [
                    f"General syntax error: {error_info.error_message}",
                    "Context (first 20 lines):",
                ]
                for i, line in enumerate(lines[:20], start=1):
                    context_lines_list.append(f"    {i:3d}: {line.rstrip()}")

                context_lines_list.extend(
                    (
                        "",
                        "Please identify and fix the syntax error. Provide ONLY the corrected code without line numbers or markers.",
                    )
                )
                return '\n'.join(context_lines_list)

            # Calculate context range for specific line errors
            start_line = max(0, error_info.line_number - context_lines - 1)
            end_line = min(len(lines), error_info.line_number + context_lines)

            context_lines_list = []
            context_lines_list.extend(
                (
                    f"Error at line {error_info.line_number}: {error_info.error_message}",
                    "Context:",
                )
            )
            for i, line in enumerate(lines[start_line:end_line], start=start_line + 1):
                marker = ">>>" if i == error_info.line_number else "   "
                context_lines_list.append(f"{marker} {i:3d}: {line.rstrip()}")

            context_lines_list.extend(
                (
                    "",
                    "Please provide ONLY the corrected code without line numbers or markers.",
                )
            )
            return '\n'.join(context_lines_list)

        except Exception as e:
            self.logger.error(f"Error extracting file context: {e}")
            return ""
    
    def attempt_error_fix(self, error_info: ErrorInfo, target_file: str) -> Tuple[bool, str]:
        """Attempt to fix a single error using LLM - tries function-level fix first"""
        # Check for duplicate error to prevent extended response bug
        if self._is_duplicate_error(error_info):
            self.logger.warning(f"Duplicate error detected - skipping to prevent extended response: {error_info.error_type}")
            return False, "Duplicate error - skipped to prevent extended response bug"
        
        # Mark error as being processed
        self._mark_error_processed(error_info)
        
        # First try function-level fix
        function_fix_success, function_fix_message = self.attempt_function_level_fix(error_info, target_file)
        if function_fix_success:
            return True, function_fix_message
        
        # Fall back to line-level fix
        self.logger.info(f"Function-level fix failed ({function_fix_message}), trying line-level fix")
        return self.attempt_line_level_fix(error_info, target_file)
    
    def attempt_function_level_fix(self, error_info: ErrorInfo, target_file: str) -> Tuple[bool, str]:
        """Attempt to fix an error by replacing the entire function"""
        try:
            # SIMPLE SYNTAX ERROR BYPASS - Handle obvious cases directly
            if error_info.error_type == "SyntaxError" and "never closed" in error_info.error_message:
                self.logger.info(f"SIMPLE SYNTAX FIX: Detected obvious syntax error - {error_info.error_message}")
                return self.attempt_simple_syntax_fix(error_info, target_file)

            # Read the source file
            with open(target_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Find the function containing the error
            function_info = self.function_extractor.find_function_containing_line(
                source_code, error_info.line_number
            )

            if not function_info:
                return False, f"Could not find function containing line {error_info.line_number}"

            self.logger.info(f"Found function {function_info.name} containing error on line {error_info.line_number}")

            # Try simple syntax fix first for obvious errors
            simple_fix_success, simple_fix_message = self.attempt_simple_syntax_fix(error_info, target_file)
            if simple_fix_success:
                return True, simple_fix_message

            # Prepare error information for LLM
            error_dict = {
                'error_type': error_info.error_type,
                'error_message': error_info.error_message,
                'line_number': error_info.line_number,
                'function_name': error_info.function_name,
                'code_context': error_info.code_context
            }

            self.logger.info(f"Requesting targeted code block fix for {error_info.error_type}: {error_info.error_message}")

            # Use targeted code block extraction instead of entire function
            # Fixed: Call with correct parameters - pass error_dict instead of error_info, and target_file only
            targeted_fix_result = self.attempt_targeted_code_block_fix(error_dict, target_file)

            # The method returns a Dict, so check the success field
            if targeted_fix_result.get('success', False):
                return True, targeted_fix_result.get('message', 'Targeted code block fix applied')

            targeted_fix_message = targeted_fix_result.get('message', 'Targeted fix failed')

            # Check if this was a system error that shouldn't be retried
            if "not retrying" in targeted_fix_message:
                return False, targeted_fix_message

            # Only fall back to full function if targeted approach fails with content issues
            self.logger.warning(f"Targeted fix failed ({targeted_fix_message}), falling back to full function fix")

            # Get function fix from LLM (fallback)
            llm_response = self.llm_interface.generate_function_fix(
                error_dict, function_info, target_file
            )

            if not llm_response.success:
                return False, f"LLM request failed: {llm_response.error}"

            # Extract code from LLM response
            fixed_function_code = self.llm_interface.extract_code_from_response(llm_response.content)

            if not fixed_function_code:
                return False, "No code found in LLM response"

            # Apply the enhanced function replacement
            edit_result = self.file_editor.apply_enhanced_function_replacement(
                target_file, function_info, fixed_function_code
            )

            if not edit_result.success:
                return False, f"Function replacement failed: {edit_result.error}"

            self.logger.info(f"Successfully applied function-level fix for {error_info.error_type}")
            return True, "Function fix applied successfully"
        except Exception as e:
            error_msg = f"Error in function-level fix: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def attempt_targeted_code_block_fix(self, error_info: Dict, file_path: str) -> Dict:
        """
        Attempt to fix error using targeted code block extraction.
        More efficient than function-level fixes for large codebases.
        """
        self.logger.info("Attempting targeted code block fix...")
        
        try:
            # Extract context around the error line
            error_line = error_info.get('line_number')
            if not error_line:
                self.logger.warning("No error line number available for targeted fix")
                return self._create_fix_result(False, "No error line number available")
            
            # Use the function extractor to find the code block containing the error
            code_block_info = self.function_extractor.find_code_block_containing_line(
                open(file_path, 'r', encoding='utf-8').read(), error_line
            )
            
            if not code_block_info:
                self.logger.warning(f"Could not find code block containing line {error_line}")
                return self._create_fix_result(False, "Could not find containing code block")
            
            self.logger.info(f"Found code block: {code_block_info.block_type} "
                           f"(lines {code_block_info.start_line}-{code_block_info.end_line})")
            
            # Extract targeted code around the error (with context)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Use the code block boundaries as base, but add some context
            context_size = 5  # Lines of context around the block
            start_line = max(1, code_block_info.start_line - context_size)
            end_line = min(len(lines), code_block_info.end_line + context_size)
            
            targeted_lines = lines[start_line-1:end_line]  # Convert to 0-indexed
            targeted_code = ''.join(targeted_lines).rstrip()
            
            self.logger.info(f"Using targeted code block: lines {start_line}-{end_line}")
            
            # Enhance error info with targeted context
            enhanced_error_info = error_info.copy()
            enhanced_error_info.update({
                'targeted_code': targeted_code,
                'block_start_line': start_line,
                'block_end_line': end_line,
                'code_block_range': f"{start_line}-{end_line}",
                'block_type': code_block_info.block_type,
                'function_name': code_block_info.parent_function or 'unknown',
                'code_context': f"Code block type: {code_block_info.block_type}, "
                               f"Parent function: {code_block_info.parent_function or 'None'}"
            })
            
            # Request fix from LLM
            self.logger.info(f"Requesting targeted fix for {error_info.get('error_type', 'Unknown')} error")
            llm_response = self.llm_interface.generate_targeted_code_fix(enhanced_error_info, file_path)
            
            if not llm_response.success:
                error_msg = llm_response.error or "LLM request failed"
                self.logger.error(f"LLM request failed: {error_msg}")
                return self._create_fix_result(False, f"LLM request failed: {error_msg}")
            
            self.logger.info("LLM FIX RECEIVED")
            
            # Extract the fixed code
            self.logger.info("Extracting code from response...")
            fixed_code = self.llm_interface.extract_code_from_response(llm_response.content)
            
            if not fixed_code or not fixed_code.strip():
                self.logger.error("No code extracted from LLM response")
                self.logger.info(f"Response content preview: {llm_response.content[:500]}...")
                return self._create_fix_result(False, "No code extracted from LLM response")
            
            self.logger.info("CODE EXTRACTED SUCCESSFULLY")
            self.logger.info(f"Fixed code length: {len(fixed_code)} characters")
            self.logger.info("-" * 50)
            self.logger.info("EXTRACTED CODE:")
            self.logger.info(fixed_code[:200] + ("..." if len(fixed_code) > 200 else ""))
            
            # Apply the fix using code block replacement
            edit_result = self.file_editor.apply_code_block_replacement(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                new_code=fixed_code
            )
            
            if edit_result.success:
                self.logger.info("Targeted code block fix applied successfully")
                return self._create_fix_result(
                    True, 
                    "Targeted code block fix applied",
                    changes=edit_result.diff,
                    backup_path=edit_result.backup_path
                )
            else:
                # Fixed: Use edit_result.error instead of edit_result.message
                error_msg = getattr(edit_result, 'error', edit_result.error)
                self.logger.error(f"Failed to apply targeted fix: {error_msg}")
                return self._create_fix_result(False, f"Failed to apply targeted fix: {error_msg}")
                
        except Exception as e:
            error_msg = f"Error in targeted code block fix: {e}"
            self.logger.error(error_msg)
            return self._create_fix_result(False, error_msg)

    def _create_fix_result(self, success: bool, message: str, changes: str = None, backup_path: str = None) -> Dict:
        """Create a standardized fix result dictionary"""
        result = {
            'success': success,
            'message': message
        }
        if changes:
            result['changes'] = changes
        if backup_path:
            result['backup_path'] = backup_path
        return result
    
    def extract_logical_code_block(self, source_lines: List[str], error_line: int, 
                                 function_start: int, function_end: int) -> Optional[Dict]:
        """Extract logical code block (try/except, if/else, loop) containing the error"""
        try:
            # Look for logical structures around the error line
            for i in range(max(function_start, error_line - 20), min(function_end, error_line + 5)):
                if i <= 0 or i > len(source_lines):
                    continue
                    
                line = source_lines[i-1].strip()
                
                # Check for try/except blocks
                if line.startswith('try:') or 'try:' in line:
                    block_end = self.find_matching_block_end(source_lines, i, ['except', 'finally'], function_end)
                    if block_end and i <= error_line <= block_end:
                        code_lines = source_lines[i-1:block_end]
                        return {
                            'block_type': 'try/except',
                            'start_line': i,
                            'end_line': block_end,
                            'code': ''.join(code_lines)
                        }
                
                # Check for if/else blocks
                if line.startswith('if ') or line.startswith('elif ') or line.startswith('else:'):
                    block_end = self.find_matching_block_end(source_lines, i, ['elif', 'else'], function_end)
                    if block_end and i <= error_line <= block_end:
                        code_lines = source_lines[i-1:block_end]
                        return {
                            'block_type': 'if/else',
                            'start_line': i,
                            'end_line': block_end,
                            'code': ''.join(code_lines)
                        }
                
                # Check for loop blocks
                if line.startswith('for ') or line.startswith('while '):
                    block_end = self.find_matching_block_end(source_lines, i, [], function_end)
                    if block_end and i <= error_line <= block_end:
                        code_lines = source_lines[i-1:block_end]
                        return {
                            'block_type': 'loop',
                            'start_line': i,
                            'end_line': block_end,
                            'code': ''.join(code_lines)
                        }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting logical code block: {e}")
            return None
    
    def find_matching_block_end(self, source_lines: List[str], start_line: int, 
                              keywords: List[str], function_end: int) -> Optional[int]:
        """Find the end of a code block by tracking indentation"""
        try:
            if start_line <= 0 or start_line > len(source_lines):
                return None
                
            start_indent = len(source_lines[start_line-1]) - len(source_lines[start_line-1].lstrip())
            
            for i in range(start_line + 1, min(function_end + 1, len(source_lines) + 1)):
                if i > len(source_lines):
                    break
                    
                line = source_lines[i-1]
                if line.strip() == '':
                    continue
                    
                current_indent = len(line) - len(line.lstrip())
                
                # If we hit a line with same or less indentation, block ends
                if current_indent <= start_indent:
                    # Check if it's a continuation keyword
                    line_stripped = line.strip()
                    is_continuation = any(line_stripped.startswith(kw) for kw in keywords)
                    
                    if not is_continuation:
                        return i - 1
                    else:
                        # Continue to find the real end
                        continue
            
            return function_end
            
        except Exception as e:
            self.logger.warning(f"Error finding block end: {e}")
            return None
    
    def attempt_line_level_fix(self, error_info: ErrorInfo, target_file: str) -> Tuple[bool, str]:
        """Attempt to fix a single error using LLM with line-level context"""
        try:
            # Extract the actual code block around the error for proper replacement
            with open(target_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()
            
            # Calculate context range around the error line
            error_line = error_info.line_number
            context_lines = 10
            block_start = max(1, error_line - context_lines)
            block_end = min(len(source_lines), error_line + context_lines)
            
            # Extract the targeted code block
            targeted_code_lines = source_lines[block_start-1:block_end]
            targeted_code = ''.join(targeted_code_lines)
            
            # Create error context for LLM (similar to targeted fix)
            error_dict = {
                'error_type': error_info.error_type,
                'error_message': error_info.error_message,
                'line_number': error_line,
                'function_name': error_info.function_name,
                'code_context': error_info.code_context,
                'block_start_line': block_start,
                'block_end_line': block_end,
                'targeted_code': targeted_code
            }
            
            self.logger.info(f"Requesting line-level fix for {error_info.error_type}: {error_info.error_message}")
            self.logger.info(f"Using targeted code block: lines {block_start}-{block_end}")
            
            # Get fix from LLM using targeted approach
            llm_response = self.llm_interface.generate_targeted_code_fix(error_dict, target_file)
            
            if not llm_response.success:
                self.logger.error("❌ LLM REQUEST FAILED")
                self.logger.error(f"Error: {llm_response.error}")
                return False, f"LLM request failed: {llm_response.error}"
            
            self.logger.info("LLM FIX RECEIVED")
            self.logger.info("Extracting code from response...")
            
            # Extract code from LLM response
            fixed_code = self.llm_interface.extract_code_from_response(llm_response.content)
            
            if not fixed_code:
                self.logger.error("❌ NO CODE FOUND IN LLM RESPONSE")
                return False, "No code found in LLM response"
            
            self.logger.info("CODE EXTRACTED SUCCESSFULLY")
            self.logger.info(f"Fixed code length: {len(fixed_code)} characters")
            self.logger.info("-" * 50)
            self.logger.info("EXTRACTED CODE:")
            self.logger.info(fixed_code)
            self.logger.info("-" * 50)
            
            # Apply the fix using code block replacement (proper approach)
            self.logger.info("APPLYING CODE FIX USING BLOCK REPLACEMENT")
            edit_result = self.file_editor.apply_code_block_replacement(
                target_file, block_start, block_end, fixed_code
            )
            
            if edit_result.success:
                self.logger.info("✅ FIX APPLIED SUCCESSFULLY")
                self.logger.info(f"Fix type: Line-level block fix for {error_info.error_type}")
                self.logger.info(f"Lines replaced: {block_start}-{block_end}")
                return True, "Line-level block fix applied successfully"
            else:
                self.logger.warning("PRIMARY FIX FAILED, TRYING ALTERNATIVE APPROACH")
                self.logger.warning(f"Primary failure reason: {edit_result.error}")
                
                # Try alternative approach - single line replacement if it's a simple case
                if error_info.line_number > 0:
                    self.logger.info(f"Attempting single line replacement at line {error_info.line_number}")
                    # Extract just the corrected line from the fixed code
                    fixed_lines = fixed_code.split('\n')
                    if len(fixed_lines) > 0:
                        # Find the line that corresponds to the error line
                        relative_error_line = error_info.line_number - block_start
                        if 0 <= relative_error_line < len(fixed_lines):
                            corrected_line = fixed_lines[relative_error_line]
                            if corrected_line.strip():
                                edit_result = self.file_editor.apply_line_replacement(
                                    target_file, error_info.line_number, corrected_line
                                )
                                if edit_result.success:
                                    self.logger.info(f"Successfully applied single line fix for {error_info.error_type}")
                                    return True, "Single line fix applied successfully"
                
                return False, f"Failed to apply fix: {edit_result.error}"
            
        except Exception as e:
            error_msg = f"Error attempting line-level fix: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def run_debug_cycle(self, target_file: str, python_executable: str = "python") -> DebugSession:
        """Run a complete debugging cycle on a file with detailed real-time logging"""
        session_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = DebugSession(
            target_file=target_file,
            session_id=session_id,
            start_time=datetime.now(),
            max_iterations=self.max_iterations
        )

        self.logger.info("=" * 100)
        self.logger.info(f"STARTING DEBUG SESSION: {session_id}")
        self.logger.info("=" * 100)
        self.logger.info(f"Target File: {target_file}")
        self.logger.info(f"Max Iterations: {self.max_iterations}")
        self.logger.info(f"Python Executable: {python_executable}")
        self.logger.info(f"Timeout per run: {self.timeout_per_run}s")
        self.logger.info("=" * 100)

        # Check if this is a continuous application
        is_continuous = self.is_continuous_application(target_file)
        if is_continuous:
            self.logger.info("DETECTED CONTINUOUS APPLICATION - Will use different success criteria")
        else:
            self.logger.info("DETECTED REGULAR SCRIPT - Will check for clean exit")

        try:
            for iteration in range(self.max_iterations):
                session.iterations = iteration + 1

                self.logger.info("\n" + "="*60)
                self.logger.info(f"DEBUG ITERATION {iteration + 1}/{self.max_iterations}")
                self.logger.info("="*60)

                # Run the file and capture errors
                self.logger.info("RUNNING FILE FOR ERROR DETECTION")
                self.logger.info(f"Command: {python_executable} {target_file}")
                self.logger.info("-" * 40)

                # Run the file
                stdout, stderr, returncode = self.run_file_with_debug(target_file, python_executable)

                # Check if successful - different criteria for continuous vs regular apps
                if is_continuous:
                    # For continuous applications: success if no errors during startup
                    success = returncode == 0 and not self._has_startup_errors(stderr)
                    if success:
                        session.success = True
                        session.final_status = "Continuous application starts successfully without errors"
                        self.logger.info("Continuous application starts successfully!")
                        break
                else:
                    # For regular scripts: success if clean exit with no errors
                    success = returncode == 0 and not stderr.strip()
                    if success:
                        session.success = True
                        session.final_status = "File runs successfully without errors"
                        self.logger.info("File runs successfully! Debug cycle complete.")
                        break

                # Parse errors from output using enhanced parser
                combined_output = stdout + "\n" + stderr
                self.logger.info(f"PARSING RUNTIME OUTPUT ({len(combined_output)} chars)")
                self.logger.info("Raw output preview:")
                self.logger.info("-" * 40)
                preview = (
                    f"{combined_output[:500]}..."
                    if len(combined_output) > 500
                    else combined_output
                )
                self.logger.info(preview)
                self.logger.info("-" * 40)

                errors = self.log_parser.parse_runtime_output(combined_output)

                if errors:
                    self.logger.info(f"DebugLogParser found {len(errors)} errors:")
                    for i, error in enumerate(errors, 1):
                        self.logger.info(f"  {i}. {error.error_type}: {error.error_message}")
                        self.logger.info(f"     File: {error.file_path}, Line: {error.line_number}")
                        if error.function_name != "unknown":
                            self.logger.info(f"     Function: {error.function_name}")
                else:
                    if is_continuous and not self._has_startup_errors(stderr):
                        # No parseable errors and no startup errors for continuous app
                        session.success = True
                        session.final_status = "Continuous application running without detectable errors"
                        self.logger.info("Continuous application running without detectable errors!")
                        break
                    elif not is_continuous:
                        # No parseable errors for regular script
                        session.final_status = f"Script failed with return code {returncode}, but no parseable errors found"
                        self.logger.warning(session.final_status)
                        break

                if not errors:
                    # No errors to fix
                    break

                session.total_errors = len(errors)
                self.logger.info(f"Found {len(errors)} errors to fix")

                # Attempt to fix each error
                fixed_any = False
                for error in errors:
                    self.logger.info(f"Attempting to fix: {error.error_type} at line {error.line_number}")

                    success, message = self.attempt_error_fix(error, target_file)

                    if success:
                        session.errors_fixed += 1
                        fixed_any = True
                        self.logger.info(f"Fixed error: {message}")
                        # Only fix one error per iteration to avoid conflicts
                        break
                    else:
                        self.logger.warning(f"Failed to fix error: {message}")

                if not fixed_any:
                    session.final_status = "Unable to fix any errors in this iteration"
                    self.logger.error("Unable to fix any errors. Stopping debug cycle.")
                    break

                # Small delay between iterations
                time.sleep(1)

            if session.iterations >= self.max_iterations and not session.success:
                if is_continuous:
                    session.final_status = f"Continuous application still has errors after {self.max_iterations} iterations"
                else:
                    session.final_status = f"Maximum iterations ({self.max_iterations}) reached without full success"

        except Exception as e:
            session.final_status = f"Debug cycle failed with exception: {e}"
            self.logger.error(session.final_status)

        self.session_history.append(session)
        self.save_session_report(session)

        return session
    
    def _get_line_indentation(self, file_path: str, line_number: int) -> Optional[str]:
        """Get the indentation of a specific line in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if 1 <= line_number <= len(lines):
                line = lines[line_number - 1]
                # Extract leading whitespace
                return line[:len(line) - len(line.lstrip())]
            return None
        except Exception:
            return None
    
    def save_session_report(self, session: DebugSession):
        """Save a detailed report of the debug session"""
        report = {
            'session_id': session.session_id,
            'target_file': session.target_file,
            'start_time': session.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'iterations': session.iterations,
            'max_iterations': session.max_iterations,
            'errors_fixed': session.errors_fixed,
            'total_errors': session.total_errors,
            'success': session.success,
            'final_status': session.final_status
        }
        
        report_file = f"debug_report_{session.session_id}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved debug report to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug report: {e}")
    
    def run_multiple_files(self, file_paths: List[str], 
                          python_executable: str = "python") -> List[DebugSession]:
        """Run debug cycles on multiple files"""
        sessions = []
        
        for file_path in file_paths:
            self.logger.info(f"Starting debug cycle for {file_path}")
            session = self.run_debug_cycle(file_path, python_executable)
            sessions.append(session)
            
            # Log session summary
            self.logger.info(f"Session {session.session_id} completed:")
            self.logger.info(f"  Success: {session.success}")
            self.logger.info(f"  Errors fixed: {session.errors_fixed}")
            self.logger.info(f"  Iterations: {session.iterations}")
            self.logger.info(f"  Status: {session.final_status}")
        
        return sessions
    
    def get_session_summary(self) -> Dict:
        """Get summary of all debug sessions"""
        total_sessions = len(self.session_history)
        successful_sessions = sum(bool(s.success)
                              for s in self.session_history)
        total_errors_fixed = sum(s.errors_fixed for s in self.session_history)
        
        return {
            'total_sessions': total_sessions,
            'successful_sessions': successful_sessions,
            'success_rate': successful_sessions / total_sessions if total_sessions > 0 else 0,
            'total_errors_fixed': total_errors_fixed,
            'average_iterations': sum(s.iterations for s in self.session_history) / total_sessions if total_sessions > 0 else 0
        }
    
    def attempt_section_aware_fix(self, error_info: ErrorInfo, function_info, target_file: str, source_lines: List[str]) -> Tuple[bool, str]:
        """Handle massive functions like run_bot with section-aware extraction"""
        try:
            error_line = error_info.line_number
            function_start = function_info.start_line
            function_end = function_info.end_line

            if section_info := self.identify_run_bot_section(
                    error_line, source_lines, function_start
                ):
                if function_info.name == 'run_bot':
                    self.logger.info(f"Error in section: {section_info['name']} (lines {section_info['start']}-{section_info['end']})")

                    # Extract the specific section with minimal context
                    section_start = max(function_start, section_info['start'] - 5)
                    section_end = min(function_end, section_info['end'] + 5)

                    section_code_lines = source_lines[section_start-1:section_end]
                    section_code = ''.join(section_code_lines)

                    # Create enhanced error context for the section
                    error_dict = {
                        'error_type': error_info.error_type,
                        'error_message': error_info.error_message,
                        'line_number': error_info.line_number,
                        'function_name': function_info.name,
                        'section_name': section_info['name'],
                        'section_description': section_info['description'],
                        'code_context': error_info.code_context,
                        'block_start_line': section_start,
                        'block_end_line': section_end,
                        'targeted_code': section_code,
                        'section_aware': True
                    }

                    # Request section-aware fix from LLM
                    llm_response = self.llm_interface.generate_targeted_code_fix(error_dict, target_file)

                    if not llm_response.success:
                        return False, f"LLM section-aware request failed: {llm_response.error}"

                    # Extract and apply the fix
                    fixed_code = self.llm_interface.extract_code_from_response(llm_response.content)
                    if not fixed_code:
                        return False, "No code found in section-aware LLM response"

                    edit_result = self.file_editor.apply_code_block_replacement(
                        target_file, section_start, section_end, fixed_code
                    )

                    if not edit_result.success:
                        return False, f"Failed to apply section-aware fix: {edit_result.error}"

                    self.logger.info(f"Successfully applied section-aware fix to {section_info['name']}")
                    return True, f"Section-aware fix applied to {section_info['name']}"
            # Fallback to extended context extraction for other massive functions
            context_lines = 25
            block_start = max(function_start, error_line - context_lines)
            block_end = min(function_end, error_line + context_lines)

            targeted_code_lines = source_lines[block_start-1:block_end]
            targeted_code = ''.join(targeted_code_lines)

            error_dict = {
                'error_type': error_info.error_type,
                'error_message': error_info.error_message,
                'line_number': error_info.line_number,
                'function_name': function_info.name,
                'code_context': error_info.code_context,
                'block_start_line': block_start,
                'block_end_line': block_end,
                'targeted_code': targeted_code,
                'massive_function': True
            }

            llm_response = self.llm_interface.generate_targeted_code_fix(error_dict, target_file)

            if not llm_response.success:
                return False, f"LLM massive function request failed: {llm_response.error}"

            fixed_code = self.llm_interface.extract_code_from_response(llm_response.content)
            if not fixed_code:
                return False, "No code found in massive function LLM response"

            edit_result = self.file_editor.apply_code_block_replacement(
                target_file, block_start, block_end, fixed_code
            )

            if edit_result.success:
                self.logger.info(f"Successfully applied massive function fix (lines {block_start}-{block_end})")
                return True, "Massive function targeted fix applied successfully"
            else:
                # Check if this is a file editor system error vs content error
                error_msg = getattr(edit_result, 'error', 'Unknown edit error')
                if "object has no attribute" not in error_msg and (
                    "missing" not in error_msg
                    or "required positional arguments" not in error_msg
                ):
                    # This is likely a content/code issue, fallback might help
                    return False, f"Failed to apply massive function fix: {error_msg}"

                # This is a file editor system bug, not an LLM generation issue
                self.logger.error(f"File editor system error in massive function fix: {error_msg}")
                return False, f"File editor system error (not retrying): {error_msg}"
        except Exception as e:
            # Log the exception and don't fall back for system errors
            if ("object has no attribute" in str(e) or 
                "missing" in str(e) and "required positional arguments" in str(e)):
                self.logger.error(f"File editor system exception in section-aware fix: {e}")
                return False, f"File editor system exception (not retrying): {e}"
            else:
                self.logger.error(f"Section-aware fix failed: {e}")
                return False, f"Section-aware fix error: {e}"
    
    def identify_run_bot_section(self, error_line: int, source_lines: List[str], function_start: int) -> Optional[Dict]:
        """Identify which section of the run_bot function contains the error"""
        try:
            # Define the known sections of run_bot function
            sections = [
                {"pattern": r"# (Initialize|Declare) global variables", "name": "Variable Declaration", "description": "Global variable initialization"},
                {"pattern": r"# (Configure|Initialize) WebSocket", "name": "WebSocket Setup", "description": "WebSocket manager and connection setup"},
                {"pattern": r"# (Fetch|Configure) Coinbase exchange", "name": "Exchange Configuration", "description": "Exchange API configuration"},
                {"pattern": r"# (Fetch initial|Initialize) market data", "name": "Market Data Initialization", "description": "Initial market data and feature computation"},
                {"pattern": r"# (Train|Initialize) machine learning", "name": "ML Model Training", "description": "Machine learning model training and setup"},
                {"pattern": r"# (Initialize|Perform) initial ETH purchase", "name": "Initial Purchase", "description": "Initial ETH purchase and grid setup"},
                {"pattern": r"# (Place initial|Initialize) base grid", "name": "Grid Initialization", "description": "Initial grid order placement"},
                {"pattern": r"# 1\. Fetch Market Data", "name": "Section 1: Market Data", "description": "Real-time market data fetching in main loop"},
                {"pattern": r"# 2\. Update Balances", "name": "Section 2: Balance Update", "description": "Account balance synchronization"},
                {"pattern": r"# 3\. Refresh Data for ML", "name": "Section 3: Data Refresh", "description": "ML prediction data refresh and WebSocket updates"},
                {"pattern": r"# 4\. Retraining", "name": "Section 4: Model Retraining", "description": "ML model retraining logic"},
                {"pattern": r"# 5\. Check Orders", "name": "Section 5: Order Management", "description": "Order status checking and management"},
                {"pattern": r"# 6\.|# Feature-based|# RSI|# Bollinger|# MACD", "name": "Section 6: Feature Trading", "description": "Feature-based trading logic (RSI, Bollinger, MACD, etc.)"},
                {"pattern": r"# 7\. Adjust Parameters", "name": "Section 7: Parameter Adjustment", "description": "Dynamic parameter adjustment based on ML predictions"},
                {"pattern": r"# 8\. Send WebSocket", "name": "Section 8: WebSocket Updates", "description": "WebSocket message sending"},
                {"pattern": r"# 9\. Process WebSocket", "name": "Section 9: WebSocket Processing", "description": "WebSocket command processing"},
                {"pattern": r"# 11\. Update Bot State", "name": "Section 11: State Update", "description": "Bot state updates and logging"}
            ]
            
            # Find which section contains the error
            for section in sections:
                section_start = self.find_section_start(source_lines, section["pattern"], function_start, error_line)
                if section_start and section_start <= error_line:
                    section_end = self.find_section_end(source_lines, section_start, function_start)
                    if section_end and error_line <= section_end:
                        return {
                            "name": section["name"],
                            "description": section["description"],
                            "start": section_start,
                            "end": section_end,
                            "pattern": section["pattern"]
                        }
            
            # If no specific section found, try to find a logical block
            return self.find_nearest_logical_block(source_lines, error_line, function_start)
            
        except Exception as e:
            self.logger.warning(f"Error identifying run_bot section: {e}")
            return None
    
    def find_section_start(self, source_lines: List[str], pattern: str, function_start: int, error_line: int) -> Optional[int]:
        """Find the start line of a section based on pattern"""
        try:
            # Search backwards from error line to function start
            for line_num in range(error_line, function_start - 1, -1):
                if line_num <= len(source_lines):
                    line = source_lines[line_num - 1]
                    if re.search(pattern, line, re.IGNORECASE):
                        return line_num
            return None
        except Exception as e:
            self.logger.warning(f"Error finding section start: {e}")
            return None
    
    def find_section_end(self, source_lines: List[str], section_start: int, function_start: int) -> Optional[int]:
        """Find the end of a section by looking for the next section or logical break"""
        try:
            base_indent = None
            section_started = False
            
            for line_num in range(section_start, min(len(source_lines) + 1, section_start + 200)):
                if line_num > len(source_lines):
                    break
                    
                line = source_lines[line_num - 1]
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Determine base indentation from first non-comment line after section start
                if not section_started and not line.strip().startswith('#'):
                    base_indent = len(line) - len(line.lstrip())
                    section_started = True
                    continue
                
                if section_started and base_indent is not None:
                    current_indent = len(line) - len(line.lstrip())
                    
                    # Check for next section marker
                    if (line.strip().startswith('# ') and 
                        ('Section' in line or re.search(r'# \d+\.', line) or 
                         'Initialize' in line or 'Fetch' in line or 'Train' in line)):
                        return line_num - 1
                    
                    # Check for significant indentation decrease (end of block)
                    if current_indent < base_indent and line.strip():
                        return line_num - 1
            
            # Default to a reasonable section size
            return min(section_start + 100, len(source_lines))
            
        except Exception as e:
            self.logger.warning(f"Error finding section end: {e}")
            return section_start + 50
    
    def find_nearest_logical_block(self, source_lines: List[str], error_line: int, function_start: int) -> Optional[Dict]:
        """Find the nearest logical code block containing the error"""
        try:
            # Look for try/except, if/else, for/while blocks
            for search_line in range(max(function_start, error_line - 30), error_line + 1):
                if search_line <= len(source_lines):
                    line = source_lines[search_line - 1].strip()
                    
                    if (line.startswith('try:') or line.startswith('if ') or 
                        line.startswith('for ') or line.startswith('while ') or
                        line.startswith('with ')):
                        
                        block_end = self.find_matching_block_end(
                            source_lines, search_line, [], function_start + 500
                        )
                        
                        if block_end and search_line <= error_line <= block_end:
                            return {
                                "name": f"Logical Block ({line[:20]}...)",
                                "description": f"Code block starting with: {line}",
                                "start": search_line,
                                "end": block_end,
                                "pattern": "logical_block"
                            }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding logical block: {e}")
            return None
    
    def _extract_errors_manually(self, output: str, target_file: str) -> List[ErrorInfo]:
        """Manual error extraction as fallback when DebugLogParser fails"""
        errors = []

        try:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                # Look for common error patterns
                if any(error_type in line for error_type in ['Error:', 'Exception:']) and ':' in line:
                    parts = line.split(':', 1)
                    error_type = parts[0].strip()
                    error_message = parts[1].strip() if len(parts) > 1 else "Unknown error"
                
                    # Try to find line number in surrounding context
                    line_number = 0
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if line_match := re.search(r'line (\d+)', lines[j]):
                            line_number = int(line_match[1])
                            break
                
                    # Try to find file path
                    file_path = target_file
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if file_match := re.search(
                            r'File "([^"]+)"', lines[j]
                        ):
                            file_path = file_match[1]
                            break
                
                    error = ErrorInfo(
                        error_type=error_type,
                        error_message=error_message,
                        file_path=file_path,
                        line_number=line_number,
                        function_name="unknown",
                        code_context="",
                        full_traceback=line,
                        timestamp=datetime.now().isoformat()
                    )
                    errors.append(error)
                    self.logger.info(f"Manual extraction: {error_type} at line {line_number}")

        except Exception as e:
            self.logger.warning(f"Manual error extraction failed: {e}")

        return errors
    
    def integrate_vscode_debugger(self, breakpoint_data: Dict) -> bool:
        """Integrate with VS Code debugger to capture precise error context"""
        try:
            self.logger.info("Integrating with VS Code debugger...")

            # Parse VS Code debugger output
            file_path = breakpoint_data.get('file_path')
            line_number = breakpoint_data.get('line_number')
            error_message = breakpoint_data.get('error_message')
            variables = breakpoint_data.get('variables', {})
            call_stack = breakpoint_data.get('call_stack', [])

            if not all([file_path, line_number, error_message]):
                return False

            # Create enhanced error info from debugger data
            enhanced_error = ErrorInfo(
                error_type="DebuggerBreakpoint",
                error_message=error_message,
                line_number=line_number,
                file_path=file_path,
                function_name="run_bot",  # Default for breakpoint debugging
                code_context=self.extract_debugger_context(file_path, line_number, variables),
                full_traceback="\\n".join(call_stack) if call_stack else "",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            if function_info := self.function_extractor.extract_function_info(
                file_path, line_number
            ):
                # Create debugger-enhanced fix prompt
                self.logger.info(f"VS Code debugger captured error at {file_path}:{line_number}")
                success, message = self.attempt_debugger_enhanced_fix(
                    enhanced_error, function_info, file_path, variables, call_stack
                )

                if success:
                    self.logger.info("Debugger-enhanced fix applied successfully")
                    return True
                else:
                    self.logger.warning(f"Debugger-enhanced fix failed: {message}")
                    return False
            else:
                self.logger.warning("Could not find function info for debugger breakpoint")
                return False

        except Exception as e:
            self.logger.error(f"VS Code debugger integration failed: {e}")
            return False
    
    def extract_debugger_context(self, file_path: str, line_number: int, variables: Dict) -> str:
        """Extract enhanced context from VS Code debugger data"""
        try:
            context_lines = []

            # Add variable values at breakpoint
            if variables:
                context_lines.append("Variables at breakpoint:")
                context_lines.extend(
                    f"  {var_name} = {var_value}"
                    for var_name, var_value in variables.items()
                )
                context_lines.append("")

            # Add code context around the line
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

            start_line = max(1, line_number - 5)
            end_line = min(len(source_lines), line_number + 5)

            context_lines.append("Code context:")
            for i in range(start_line, end_line + 1):
                marker = " -> " if i == line_number else "    "
                context_lines.append(f"{marker}Line {i}: {source_lines[i-1].rstrip()}")

            return "\\n".join(context_lines)

        except Exception as e:
            self.logger.warning(f"Error extracting debugger context: {e}")
            return f"Error at line {line_number}: {variables.get('error', 'Unknown error')}"
    
    def extract_targeted_code_block(self, source_lines: List[str], line_number: int, context_lines: int = 10) -> Optional[List[str]]:
        """Extract a targeted code block around a specific line number"""
        try:
            if not source_lines or line_number < 1 or line_number > len(source_lines):
                return None
            
            # Calculate start and end lines with context
            start_line = max(1, line_number - context_lines)
            end_line = min(len(source_lines), line_number + context_lines)
            
            # Extract the code block (convert to 0-based indexing)
            code_block = source_lines[start_line - 1:end_line]
            
            self.logger.debug(f"Extracted code block: lines {start_line}-{end_line} ({len(code_block)} lines)")
            return code_block
            
        except Exception as e:
            self.logger.error(f"Error extracting targeted code block around line {line_number}: {e}")
            return None
    
    def attempt_debugger_enhanced_fix(self, error_info: ErrorInfo, function_info, 
                                    target_file: str, variables: Dict, call_stack: List) -> Tuple[bool, str]:
        """Attempt fix using enhanced VS Code debugger information"""
        try:
            # Create comprehensive error context with debugger data
            error_dict = {
                'error_type': error_info.error_type,
                'error_message': error_info.error_message,
                'line_number': error_info.line_number,
                'function_name': function_info.name,
                'code_context': error_info.code_context,
                'debugger_variables': variables,
                'call_stack': call_stack,
                'debugger_enhanced': True
            }

            # Check if this is a massive function needing section-aware handling
            with open(target_file, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

            function_size = function_info.end_line - function_info.start_line
            if function_size > 1000 and function_info.name == 'run_bot':
                if section_info := self.identify_run_bot_section(
                    error_info.line_number, source_lines, function_info.start_line
                ):
                    error_dict['section_name'] = section_info['name']
                    error_dict['section_description'] = section_info['description']

                    # Extract section code
                    section_start = max(function_info.start_line, section_info['start'] - 3)
                    section_end = min(function_info.end_line, section_info['end'] + 3)

                    section_code_lines = source_lines[section_start-1:section_end]
                    error_dict['targeted_code'] = ''.join(section_code_lines)
                    error_dict['block_start_line'] = section_start
                    error_dict['block_end_line'] = section_end
            else:
                # Standard targeted extraction with debugger enhancement
                context_lines = 15
                block_start = max(function_info.start_line, error_info.line_number - context_lines)
                block_end = min(function_info.end_line, error_info.line_number + context_lines)

                targeted_code_lines = source_lines[block_start-1:block_end]
                error_dict['targeted_code'] = ''.join(targeted_code_lines)
                error_dict['block_start_line'] = block_start
                error_dict['block_end_line'] = block_end

            # Request debugger-enhanced fix from LLM
            llm_response = self.llm_interface.generate_targeted_code_fix(error_dict, target_file)

            if not llm_response.success:
                return False, f"LLM debugger-enhanced request failed: {llm_response.error}"

            # Extract and apply the fix
            fixed_code = self.llm_interface.extract_code_from_response(llm_response.content)
            if not fixed_code:
                return False, "No code found in debugger-enhanced LLM response"

            edit_result = self.file_editor.apply_code_block_replacement(
                target_file, error_dict['block_start_line'], error_dict['block_end_line'], fixed_code
            )

            if not edit_result.success:
                return False, f"Failed to apply debugger-enhanced fix: {edit_result.error}"

            self.logger.info("Successfully applied debugger-enhanced fix")
            return True, "Debugger-enhanced fix applied successfully"
        except Exception as e:
            self.logger.error(f"Debugger-enhanced fix failed: {e}")
            return False, f"Debugger-enhanced fix error: {e}"
            
    def integrate_with_vscode_debugger(self, file_path: str, line_number: int, 
                                     error_message: str, variables: dict, 
                                     call_stack: list) -> bool:
        """
        Integrate with VS Code debugger to process breakpoint data
        
        Args:
            file_path: Path to the file where breakpoint occurred
            line_number: Line number of the breakpoint
            error_message: Error message from the debugger
            variables: Variable values at breakpoint
            call_stack: Call stack trace
            
        Returns:
            bool: True if integration successful, False otherwise
        """
        try:
            self.logger.info("Integrating with VS Code debugger...")

            # Create enhanced error info from debugger data
            enhanced_error = ErrorInfo(
                error_type="DebuggerBreakpoint",
                error_message=error_message,
                line_number=line_number,
                file_path=file_path,
                function_name="run_bot",  # Default for breakpoint debugging
                code_context=self.extract_debugger_context(file_path, line_number, variables),
                full_traceback="\\n".join(call_stack) if call_stack else "",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Find function containing the error
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

            if function_info := self.function_extractor.extract_function_by_name(
                source_code, "run_bot"
            ):
                # Use section-aware fix for massive functions
                if function_info.name == "run_bot":
                    success, message = self.attempt_section_aware_fix(enhanced_error, function_info, file_path, source_lines)
                else:
                    success, message = self.attempt_standard_fix(enhanced_error, function_info, file_path)

                return success
            else:
                self.logger.warning("Could not identify function for debugger integration")
                return False

        except Exception as e:
            self.logger.error(f"VS Code debugger integration failed: {e}")
            return False

    def attempt_simple_syntax_fix(self, error_info: ErrorInfo, target_file: str) -> Tuple[bool, str]:
        """Handle simple syntax errors directly without complex logic"""
        try:
            self.logger.info(f"SIMPLE FIX: Processing {error_info.error_type} at line {error_info.line_number}")
            
            # Read the file
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            error_line_num = error_info.line_number
            if error_line_num > len(lines):
                return False, f"Error line {error_line_num} exceeds file length {len(lines)}"
            
            problem_line = lines[error_line_num - 1]
            self.logger.info(f"SIMPLE FIX: Problem line {error_line_num}: {problem_line.strip()}")
            
            # Handle unmatched ')' - extra closing parenthesis
            if "unmatched ')'" in error_info.error_message:
                if ')' in problem_line:
                    # For unmatched ')', there is an extra closing parenthesis
                    # Remove the last closing parenthesis
                    last_paren_pos = problem_line.rfind(')')
                    if last_paren_pos != -1:
                        fixed_line = problem_line[:last_paren_pos] + problem_line[last_paren_pos+1:]
                        lines[error_line_num - 1] = fixed_line
                        
                        # Write the fixed file
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        
                        self.logger.info(f"SIMPLE FIX APPLIED: Removed extra ')' from line {error_line_num}")
                        return True, f"Simple syntax fix applied: Removed extra ')' from line {error_line_num}"
            
            # Handle missing parenthesis "'(' was never closed"
            elif "'(' was never closed" in error_info.error_message:
                if "time.sleep(" in problem_line and not problem_line.strip().endswith(')'):
                    # Direct fix: add missing closing parenthesis
                    fixed_line = problem_line.rstrip() + ')\n'
                    lines[error_line_num - 1] = fixed_line
                    
                    # Write the fixed file
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    
                    self.logger.info(f"SIMPLE FIX APPLIED: Added missing ')' to line {error_line_num}")
                    return True, f"Simple syntax fix applied: Added missing ')' to line {error_line_num}"
            
            return False, "Simple syntax fix could not handle this error"
            
        except Exception as e:
            self.logger.error(f"Simple syntax fix error: {e}")
            return False, f"Simple syntax fix error: {e}"
    
    def is_continuous_application(self, file_path: str) -> bool:
        """Detect if a file is a continuous application that runs indefinitely"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # Check for indicators of continuous applications
            continuous_indicators = [
                # Infinite loops
                'while true:',
                'while true :',
                'while 1:',
                'while 1 :',

                # WebSocket servers
                'websocket',
                'socketio',
                'flask-socketio',
                'run_forever',
                'serve_forever',

                # Trading bots
                'trading bot',
                'grid bot',
                'bot_state',
                'run_bot',

                # Daemons/services
                'daemon',
                'service',
                'server',

                # Event loops
                'asyncio.run_forever',
                'loop.run_forever',
                'event loop',

                # Continuous monitoring
                'monitor',
                'watch',
                'continuous',

                # CLI with restart loops
                'while true:\n        try:\n            ',
                'restart',
                'reconnect',
            ]

            # Check for multiple indicators
            indicator_count = sum(
                indicator in content for indicator in continuous_indicators
            )

            # Additional check: look for main loop patterns
            main_loop_patterns = [
                r'while\s+true\s*:',  # while True:
                r'while\s+1\s*:',     # while 1:
                r'for\s+.*\s+in\s+.*:\s*$',  # potential infinite for loops
            ]

            import re
            loop_count = sum(bool(re.search(pattern, content, re.MULTILINE | re.IGNORECASE))
                         for pattern in main_loop_patterns)

            # If we find multiple indicators or clear infinite loops, classify as continuous
            is_continuous = indicator_count >= 2 or loop_count >= 1

            if is_continuous:
                self.logger.info(f"Detected continuous application: {file_path} (indicators: {indicator_count}, loops: {loop_count})")

            return is_continuous

        except Exception as e:
            self.logger.warning(f"Error detecting application type for {file_path}: {e}")
            return False
    
    def _has_startup_errors(self, stderr: str) -> bool:
        """Check if stderr contains actual startup errors (not just timeout messages)"""
        if not stderr or not stderr.strip():
            return False

        # Filter out common non-error messages that might appear in stderr
        error_lines = []
        for line in stderr.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Skip common informational messages
            skip_patterns = [
                'starting',
                'connecting',
                'listening',
                'running on',
                'server started',
                'websocket',
                'web socket',
                'info:',
                'debug:',
                'warning: connecting',
                'warning: reconnecting',
            ]

            is_error = all(
                pattern.lower() not in line.lower() for pattern in skip_patterns
            )
            # Check for actual error keywords
            error_keywords = ['error:', 'exception:', 'traceback', 'failed', 'critical:']
            has_error_keyword = any(keyword in line.lower() for keyword in error_keywords)

            if is_error and has_error_keyword:
                error_lines.append(line)

        return len(error_lines) > 0