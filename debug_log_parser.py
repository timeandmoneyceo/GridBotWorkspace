"""
Debug Log Parser Module

This module parses debug logs from GridbotBackup.py and extracts error information
for automated debugging and fixing.
"""

import re
import logging
import traceback
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ErrorInfo:
    """Container for error information extracted from logs with enhanced localized function metadata"""
    error_type: str
    error_message: str
    file_path: str
    line_number: int
    function_name: str
    code_context: str
    full_traceback: str
    timestamp: str
    severity: str = "ERROR"
    # Enhanced metadata fields with localized context
    function_signature: Optional[str] = None
    code_block_range: Optional[str] = None
    enhanced_context: Optional[str] = None
    is_websocket_error: bool = False
    # New localized fields
    is_localized: bool = False  # True if using localized code context (~30 lines vs 5000+)
    local_variables: Optional[str] = None  # Local variables at error point
    control_structures: Optional[str] = None  # Nearby if/while/for/try blocks

class DebugLogParser:
    """Parser for extracting error information from debug logs"""
    
    def __init__(self, log_file_path: str = None):
        self.log_file_path = log_file_path
        self.setup_logging()
        
        # Common error patterns
        self.error_patterns = {
            'python_exception': re.compile(
                r'(\w+Error|Exception): (.+?)(?:\n|$)', 
                re.MULTILINE | re.DOTALL
            ),
            'traceback': re.compile(
                r'Traceback \(most recent call last\):(.*?)(?=\n\w+Error|\n\w+Exception|\Z)', 
                re.MULTILINE | re.DOTALL
            ),
            'file_line': re.compile(
                r'File "([^"]+)", line (\d+), in (\w+)'
            ),
            'syntax_error': re.compile(
                r'SyntaxError: (.+?) \(line (\d+)\)'
            ),
            'import_error': re.compile(
                r'(ImportError|ModuleNotFoundError): (.+)'
            ),
            'attribute_error': re.compile(
                r'AttributeError: (.+)'
            ),
            # Enhanced function metadata patterns - Updated for localized context
            'enhanced_error': re.compile(
                r'\[(?:ERROR|WS_ERROR):([^:]+):(\d+)\]\[([^\]]+)\] (.+?)(?:\n|$)',
                re.MULTILINE
            ),
            'localized_error': re.compile(
                r'\[(?:ERROR|WS_ERROR):([^:]+):(\d+)\]\[Lines (\d+)-(\d+)\] (.+?)(?:\n|$)',
                re.MULTILINE
            ),
            'function_signature': re.compile(
                r'\[(?:ERROR_SIGNATURE|WS_ERROR_SIGNATURE):([^\]]+)\] (.+?)(?:\n|$)',
                re.MULTILINE
            ),
            'code_context': re.compile(
                r'\[(?:ERROR_CONTEXT|WS_ERROR_CONTEXT|CODE_CONTEXT):([^\]]+)\] (.+?)(?=\n\[|\Z)',
                re.MULTILINE | re.DOTALL
            ),
            'local_variables': re.compile(
                r'\[(?:ERROR_VARS|WS_ERROR_VARS|LOCAL_VARS):([^\]]+)\] (.+?)(?:\n|$)',
                re.MULTILINE
            ),
            'control_flow': re.compile(
                r'\[(?:ERROR_FLOW|WS_ERROR_FLOW|CONTROL_FLOW):([^\]]+)\] (.+?)(?:\n|$)',
                re.MULTILINE
            ),
            'websocket_error': re.compile(
                r'\[WS_ERROR:([^:]+):(\d+)\]',
                re.MULTILINE
            )
        }
    
    def setup_logging(self):
        """Setup logging for the parser"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('debug_log_parser.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def read_log_file(self, file_path: str = None) -> str:
        """Read the log file content"""
        target_file = file_path or self.log_file_path
        if not target_file:
            raise ValueError("No log file path specified")
        
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Log file not found: {target_file}")
            return ""
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
            return ""
    
    def _extract_enhanced_metadata(self, log_content: str, error_info: ErrorInfo) -> ErrorInfo:
        """Extract enhanced function metadata from logs and enrich error info with localized context"""
        # Look for enhanced error patterns around the error
        lines = log_content.split('\n')

        # Find lines containing enhanced metadata
        for line in lines:
            if enhanced_match := self.error_patterns['enhanced_error'].match(line):
                func_name, line_num, code_range, message = enhanced_match.groups()
                if func_name == error_info.function_name:
                    error_info.code_block_range = code_range
                    error_info.enhanced_context = message

            if localized_match := self.error_patterns['localized_error'].match(
                line
            ):
                func_name, line_num, start_line, end_line, message = localized_match.groups()
                if func_name == error_info.function_name:
                    error_info.code_block_range = f"Lines {start_line}-{end_line}"
                    error_info.enhanced_context = message
                    # Mark as localized (much smaller code range)
                    error_info.is_localized = True

            if sig_match := self.error_patterns['function_signature'].match(line):
                func_name, signature = sig_match.groups()
                if func_name == error_info.function_name:
                    error_info.function_signature = signature

            if context_match := self.error_patterns['code_context'].match(line):
                func_name, code = context_match.groups()
                if func_name == error_info.function_name:
                    error_info.code_context = code.strip()

            if vars_match := self.error_patterns['local_variables'].match(line):
                func_name, vars_data = vars_match.groups()
                if func_name == error_info.function_name:
                    error_info.local_variables = vars_data.strip()

            if flow_match := self.error_patterns['control_flow'].match(line):
                func_name, flow_data = flow_match.groups()
                if func_name == error_info.function_name:
                    error_info.control_structures = flow_data.strip()

            # WebSocket error detection
            ws_match = self.error_patterns['websocket_error'].match(line)
            if ws_match:
                error_info.is_websocket_error = True

        return error_info
    
    def extract_python_errors(self, log_content: str) -> List[ErrorInfo]:
        """Extract Python errors from log content"""
        self.logger.debug("DebugLogParser.extract_python_errors() starting error extraction")
        errors = []

        # Check for any common error indicators first
        if 'error' not in log_content.lower() and 'exception' not in log_content.lower() and 'syntax' not in log_content.lower():
            self.logger.debug("No obvious error keywords found, checking for syntax patterns")

        # Look for direct syntax errors (no traceback format) - more flexible patterns
        syntax_error_patterns = [
            # Standard Python syntax error format
            re.compile(
                r'File "([^"]+)", line (\d+)\s*\n.*?\n.*?\^\s*\n(\w+Error): (.+?)(?=\n|\Z)', 
                re.MULTILINE | re.DOTALL
            ),
            # Alternative syntax error format
            re.compile(
                r'(\w+Error): (.+?) \(line (\d+)\)', 
                re.MULTILINE
            ),
            # Simple error detection - if we see "SyntaxError" anywhere
            re.compile(
                r'(SyntaxError|IndentationError|TabError): (.+?)(?=\n|\Z)', 
                re.MULTILINE
            )
        ]

        for pattern in syntax_error_patterns:
            syntax_matches = pattern.findall(log_content)
            self.logger.debug(f"Pattern found {len(syntax_matches)} matches")

            for match in syntax_matches:
                if len(match) == 4:  # File, line, error_type, message
                    file_path, line_number, error_type, error_message = match
                    error_info = ErrorInfo(
                        error_type=error_type,
                        error_message=error_message.strip(),
                        file_path=file_path,
                        line_number=int(line_number),
                        function_name="unknown",
                        code_context="",
                        full_traceback=f"{error_type}: {error_message}",
                        timestamp=datetime.now().isoformat()
                    )
                    errors.append(error_info)
                    self.logger.debug(f"Parsed syntax error: {error_type} in {file_path}:{line_number}")
                elif len(match) == 3:  # Error_type, message, line
                    error_type, error_message, line_number = match
                    error_info = ErrorInfo(
                        error_type=error_type,
                        error_message=error_message.strip(),
                        file_path="unknown",
                        line_number=int(line_number),
                        function_name="unknown",
                        code_context="",
                        full_traceback=f"{error_type}: {error_message}",
                        timestamp=datetime.now().isoformat()
                    )
                    errors.append(error_info)
                    self.logger.debug(f"Parsed error: {error_type} at line {line_number}")
                elif len(match) == 2:  # Error_type, message
                    error_type, error_message = match
                    error_info = ErrorInfo(
                        error_type=error_type,
                        error_message=error_message.strip(),
                        file_path="unknown",
                        line_number=0,
                        function_name="unknown",
                        code_context="",
                        full_traceback=f"{error_type}: {error_message}",
                        timestamp=datetime.now().isoformat()
                    )
                    errors.append(error_info)
                    self.logger.debug(f"Parsed basic error: {error_type}")

        # If we found syntax errors, return early as they are the most critical
        if errors:
            self.logger.info(f"Found {len(errors)} syntax/direct errors, skipping traceback parsing")
            return errors

        # Look for complete traceback blocks (including the final error line)
        traceback_pattern = re.compile(
            r'Traceback \(most recent call last\):(.*?)(\w+Error: .*?)(?=\n\n|\Z)', 
            re.MULTILINE | re.DOTALL
        )

        traceback_matches = traceback_pattern.findall(log_content)
        self.logger.debug(f"Found {len(traceback_matches)} traceback patterns")

        for traceback_body, error_line in traceback_matches:
            try:
                if error_info := self._parse_full_traceback(
                    traceback_body, error_line
                ):
                    # Extract enhanced metadata for this error
                    error_info = self._extract_enhanced_metadata(log_content, error_info)
                    errors.append(error_info)
                    self.logger.debug(f"Parsed error with enhanced metadata: {error_info.error_type} in {error_info.file_path}:{error_info.line_number}")
            except Exception as e:
                self.logger.warning(f"Failed to parse traceback: {e}")

        # If no tracebacks found, look for any error-like content
        if not errors:
            self.logger.debug("No tracebacks found, looking for any error content")

            # Very broad error detection - any line containing common error words
            error_lines = []
            for line in log_content.split('\n'):
                line_lower = line.lower()
                if any(word in line_lower for word in ['error', 'exception', 'traceback', 'failed', 'syntax']):
                    error_lines.append(line.strip())

            if error_lines:
                # Create a generic error from the first error-like line
                error_line = error_lines[0]
                error_info = ErrorInfo(
                    error_type="GenericError",
                    error_message=error_line,
                    file_path="unknown",
                    line_number=0,
                    function_name="unknown",
                    code_context="",
                    full_traceback=error_line,
                    timestamp=datetime.now().isoformat()
                )
                errors.append(error_info)
                self.logger.debug(f"Created generic error from: {error_line}")

        # Also look for standalone error messages
        error_matches = self.error_patterns['python_exception'].findall(log_content)
        for error_type, error_msg in error_matches:
            if all(error_type not in err.full_traceback for err in errors):
                error_info = ErrorInfo(
                    error_type=error_type,
                    error_message=error_msg.strip(),
                    file_path="unknown",
                    line_number=0,
                    function_name="unknown",
                    code_context="",
                    full_traceback=f"{error_type}: {error_msg}",
                    timestamp=datetime.now().isoformat()
                )
                errors.append(error_info)

        return errors
    
    def _parse_full_traceback(self, traceback_body: str, error_line: str) -> Optional[ErrorInfo]:
        """Parse a complete traceback with separate error line"""
        # Find the last file reference (usually the error location)
        file_matches = self.error_patterns['file_line'].findall(traceback_body)
        if not file_matches:
            return None

        # Get the last (most relevant) file reference
        file_path, line_number, function_name = file_matches[-1]

        if error_match := self.error_patterns['python_exception'].search(
            error_line
        ):
            error_type, error_message = error_match.groups()
        elif ':' in error_line:
            error_type, error_message = error_line.split(':', 1)
            error_type = error_type.strip()
            error_message = error_message.strip()
        else:
            error_type = "UnknownError"
            error_message = error_line.strip()

        # Extract code context from the traceback body
        code_context = self._extract_code_context(traceback_body)

        full_traceback = traceback_body + error_line

        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            line_number=int(line_number),
            function_name=function_name,
            code_context=code_context,
            full_traceback=full_traceback,
            timestamp=datetime.now().isoformat()
        )

    def _parse_traceback(self, traceback_text: str) -> Optional[ErrorInfo]:
        """Parse a single traceback to extract error information"""
        lines = traceback_text.strip().split('\n')

        # Find the last file reference (usually the error location)
        file_matches = self.error_patterns['file_line'].findall(traceback_text)
        if not file_matches:
            return None

        # Get the last (most relevant) file reference
        file_path, line_number, function_name = file_matches[-1]

        # Extract error type and message from the last line
        error_line = lines[-1] if lines else ""
        if error_match := self.error_patterns['python_exception'].search(
            error_line
        ):
            error_type, error_message = error_match.groups()
        elif ':' in error_line:
            error_type, error_message = error_line.split(':', 1)
            error_type = error_type.strip()
            error_message = error_message.strip()
        else:
            error_type = "UnknownError"
            error_message = error_line

        # Try to extract code context
        code_context = self._extract_code_context(traceback_text)

        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            line_number=int(line_number),
            function_name=function_name,
            code_context=code_context,
            full_traceback=traceback_text,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_code_context(self, traceback_text: str) -> str:
        """Extract code context from traceback"""
        lines = traceback_text.split('\n')
        code_lines = [
            line.strip()
            for line in lines
            if line.strip()
            and not line.startswith('File ')
            and not line.startswith('Traceback')
            and (line.startswith('    ') or line.startswith('\t'))
        ]
        return '\n'.join(code_lines)
    
    def parse_runtime_output(self, output: str) -> List[ErrorInfo]:
        """Parse runtime output for errors"""
        self.logger.info(f"DebugLogParser.parse_runtime_output() called with {len(output)} characters")
        self.logger.debug(f"Output preview: {output[:200]}...")
        errors = self.extract_python_errors(output)
        self.logger.info(f"DebugLogParser found {len(errors)} errors in output")
        return errors
    
    def categorize_errors(self, errors: List[ErrorInfo]) -> Dict[str, List[ErrorInfo]]:
        """Categorize errors by type for better handling"""
        categories = {
            'syntax_errors': [],
            'import_errors': [],
            'attribute_errors': [],
            'runtime_errors': [],
            'other_errors': []
        }
        
        for error in errors:
            if error.error_type in ['SyntaxError', 'IndentationError']:
                categories['syntax_errors'].append(error)
            elif error.error_type in ['ImportError', 'ModuleNotFoundError']:
                categories['import_errors'].append(error)
            elif error.error_type in ['AttributeError']:
                categories['attribute_errors'].append(error)
            elif error.error_type in ['RuntimeError', 'ValueError', 'TypeError', 'KeyError']:
                categories['runtime_errors'].append(error)
            else:
                categories['other_errors'].append(error)
        
        return categories
    
    def generate_error_summary(self, errors: List[ErrorInfo]) -> Dict:
        """Generate a summary of errors for LLM processing"""
        categorized = self.categorize_errors(errors)
        
        summary = {
            'total_errors': len(errors),
            'error_categories': {k: len(v) for k, v in categorized.items()},
            'errors_by_file': {},
            'most_common_errors': {},
            'critical_errors': []
        }
        
        # Group by file
        for error in errors:
            if error.file_path not in summary['errors_by_file']:
                summary['errors_by_file'][error.file_path] = []
            summary['errors_by_file'][error.file_path].append({
                'type': error.error_type,
                'message': error.error_message,
                'line': error.line_number,
                'function': error.function_name
            })
        
        # Count error types
        error_counts = {}
        for error in errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        
        summary['most_common_errors'] = dict(sorted(error_counts.items(), 
                                                   key=lambda x: x[1], reverse=True))
        
        # Identify critical errors (those that prevent execution)
        critical_types = ['SyntaxError', 'IndentationError', 'ImportError', 'ModuleNotFoundError']
        summary['critical_errors'] = [
            error for error in errors if error.error_type in critical_types
        ]
        
        return summary
    
    def save_errors_to_json(self, errors: List[ErrorInfo], output_file: str):
        """Save errors to JSON file for processing"""
        error_data = [
            {
                'error_type': error.error_type,
                'error_message': error.error_message,
                'file_path': error.file_path,
                'line_number': error.line_number,
                'function_name': error.function_name,
                'code_context': error.code_context,
                'full_traceback': error.full_traceback,
                'timestamp': error.timestamp,
                'severity': error.severity,
            }
            for error in errors
        ]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(errors)} errors to {output_file}")