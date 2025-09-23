"""
Serena MCP Integration Module

This module provides integration with Serena MCP server for semantic code analysis
and editing capabilities in the GridBot automation pipeline.
"""

import os
import json
import logging
import asyncio
import subprocess
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import tempfile
import shutil

@dataclass
class SerenaToolResult:
    """Result from a Serena tool execution"""
    success: bool
    tool_name: str
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class SymbolInfo:
    """Information about a code symbol"""
    name: str
    kind: str  # 'function', 'class', 'variable', 'method', etc.
    file_path: str
    start_line: int
    end_line: int
    definition: str
    references: List[Dict] = None

@dataclass
class SemanticEdit:
    """A semantic code edit operation"""
    operation: str  # 'replace_symbol', 'insert_after_symbol', 'insert_before_symbol', etc.
    symbol_name: str
    file_path: str
    new_code: str
    context: Dict = None

class SerenaMCPClient:
    """Client for interacting with Serena MCP server"""

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 30):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._session_active = False

    def is_server_running(self) -> bool:
        """Check if Serena MCP server is running"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def start_server(self, project_path: str = None) -> bool:
        """Start Serena MCP server if not running"""
        if self.is_server_running():
            self.logger.info("Serena MCP server already running")
            return True

        try:
            env = os.environ.copy() | {
                'SERENA_PROJECT_PATH': project_path or os.getcwd(),
                'SERENA_CONTEXT': 'desktop-app',
                'SERENA_HOST': 'localhost',
                'SERENA_PORT': '8000',
                'PYTHONPATH': os.getcwd(),
            }
            # Try to start Serena server using uvx
            cmd = ["uvx", "--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server"]

            if project_path:
                cmd.extend(["--project", project_path])

            # Use desktop-app context which is the default and should work
            cmd.extend(["--context", "desktop-app"])

            # Use default modes (interactive, editing) - don't specify to avoid issues

            self.logger.info(f"Starting Serena MCP server: {' '.join(cmd)}")
            self.logger.info(f"Environment variables set: SERENA_PROJECT_PATH={env.get('SERENA_PROJECT_PATH')}, SERENA_CONTEXT={env.get('SERENA_CONTEXT')}")

            # Start server in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=project_path or os.getcwd()
            )

            # Wait for server to start
            time.sleep(10)  # Increased wait time

            if self.is_server_running():
                self.logger.info("Serena MCP server started successfully")
                return True
            else:
                self.logger.error("Failed to start Serena MCP server")
                # Get error output
                if process.poll() is None:
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                        self.logger.error(f"Server stdout: {stdout.decode() if stdout else 'None'}")
                        self.logger.error(f"Server stderr: {stderr.decode() if stderr else 'None'}")
                    except Exception:
                        pass
                return False

        except Exception as e:
            self.logger.error(f"Error starting Serena server: {e}")
            return False

    def activate_project(self, project_path: str) -> SerenaToolResult:
        """Activate a project in Serena"""
        return self._call_tool("activate_project", {
            "project_path": project_path
        })

    def find_symbol(self, symbol_name: str, file_path: Optional[str] = None,
                   symbol_type: Optional[str] = None) -> SerenaToolResult:
        """Find symbols by name"""
        params = {"symbol_name": symbol_name}
        if file_path:
            params["file_path"] = file_path
        if symbol_type:
            params["symbol_type"] = symbol_type

        return self._call_tool("find_symbol", params)

    def get_symbols_overview(self, file_path: str) -> SerenaToolResult:
        """Get overview of all symbols in a file"""
        return self._call_tool("get_symbols_overview", {
            "file_path": file_path
        })

    def find_referencing_symbols(self, symbol_name: str, file_path: str,
                               line_number: int) -> SerenaToolResult:
        """Find symbols that reference the given symbol"""
        return self._call_tool("find_referencing_symbols", {
            "symbol_name": symbol_name,
            "file_path": file_path,
            "line_number": line_number
        })

    def read_file(self, file_path: str, start_line: Optional[int] = None,
                 end_line: Optional[int] = None) -> SerenaToolResult:
        """Read file content, optionally with line range"""
        params = {"file_path": file_path}
        if start_line is not None:
            params["start_line"] = start_line
        if end_line is not None:
            params["end_line"] = end_line

        return self._call_tool("read_file", params)

    def replace_symbol_body(self, symbol_name: str, file_path: str,
                          new_body: str) -> SerenaToolResult:
        """Replace the body of a symbol (function, class, etc.)"""
        return self._call_tool("replace_symbol_body", {
            "symbol_name": symbol_name,
            "file_path": file_path,
            "new_body": new_body
        })

    def insert_after_symbol(self, symbol_name: str, file_path: str,
                          code_to_insert: str) -> SerenaToolResult:
        """Insert code after a symbol"""
        return self._call_tool("insert_after_symbol", {
            "symbol_name": symbol_name,
            "file_path": file_path,
            "code_to_insert": code_to_insert
        })

    def insert_before_symbol(self, symbol_name: str, file_path: str,
                           code_to_insert: str) -> SerenaToolResult:
        """Insert code before a symbol"""
        return self._call_tool("insert_before_symbol", {
            "symbol_name": symbol_name,
            "file_path": file_path,
            "code_to_insert": code_to_insert
        })

    def execute_shell_command(self, command: str, cwd: Optional[str] = None) -> SerenaToolResult:
        """Execute a shell command"""
        params = {"command": command}
        if cwd:
            params["cwd"] = cwd

        return self._call_tool("execute_shell_command", params)

    def _call_tool(self, tool_name: str, params: Dict) -> SerenaToolResult:
        """Call a Serena tool via MCP"""
        start_time = time.time()

        try:
            # For now, implement basic HTTP-based tool calling
            # In a real implementation, this would use the MCP protocol
            import requests

            payload = {
                "tool": tool_name,
                "parameters": params
            }

            response = requests.post(
                f"{self.server_url}/tools",
                json=payload,
                timeout=self.timeout
            )

            execution_time = time.time() - start_time

            if response.status_code != 200:
                return SerenaToolResult(
                    success=False,
                    tool_name=tool_name,
                    data=None,
                    error=f"HTTP {response.status_code}: {response.text}",
                    execution_time=execution_time
                )

            result_data = response.json()
            return SerenaToolResult(
                success=True,
                tool_name=tool_name,
                data=result_data.get("result"),
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return SerenaToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=str(e),
                execution_time=execution_time
            )

class SerenaCodeAnalyzer:
    """High-level code analysis using Serena tools"""

    def __init__(self, client: SerenaMCPClient):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def analyze_file_symbols(self, file_path: str) -> List[SymbolInfo]:
        """Analyze all symbols in a file"""
        self.logger.info(f"Analyzing symbols in {file_path}")

        # Get symbols overview
        overview_result = self.client.get_symbols_overview(file_path)
        if not overview_result.success:
            self.logger.error(f"Failed to get symbols overview: {overview_result.error}")
            return []

        symbols = []
        try:
            # Parse the overview data
            if isinstance(overview_result.data, list):
                for symbol_data in overview_result.data:
                    symbol = SymbolInfo(
                        name=symbol_data.get("name", ""),
                        kind=symbol_data.get("kind", ""),
                        file_path=file_path,
                        start_line=symbol_data.get("start_line", 0),
                        end_line=symbol_data.get("end_line", 0),
                        definition=symbol_data.get("definition", "")
                    )
                    symbols.append(symbol)
        except Exception as e:
            self.logger.error(f"Error parsing symbols overview: {e}")

        return symbols

    def find_function_definition(self, function_name: str, file_path: Optional[str] = None) -> Optional[SymbolInfo]:
        """Find a function definition"""
        self.logger.info(f"Finding function definition: {function_name}")

        result = self.client.find_symbol(function_name, file_path, "function")
        if not result.success:
            self.logger.error(f"Failed to find function: {result.error}")
            return None

        try:
            if isinstance(result.data, list) and result.data:
                symbol_data = result.data[0]  # Take first match
                return SymbolInfo(
                    name=symbol_data.get("name", function_name),
                    kind="function",
                    file_path=symbol_data.get("file_path", file_path or ""),
                    start_line=symbol_data.get("start_line", 0),
                    end_line=symbol_data.get("end_line", 0),
                    definition=symbol_data.get("definition", "")
                )
        except Exception as e:
            self.logger.error(f"Error parsing function definition: {e}")

        return None

    def get_function_references(self, function_name: str, file_path: str,
                              line_number: int) -> List[Dict]:
        """Get all references to a function"""
        self.logger.info(f"Finding references to {function_name} in {file_path}:{line_number}")

        result = self.client.find_referencing_symbols(function_name, file_path, line_number)
        if not result.success:
            self.logger.error(f"Failed to find references: {result.error}")
            return []

        try:
            if isinstance(result.data, list):
                return result.data
        except Exception as e:
            self.logger.error(f"Error parsing references: {e}")

        return []

    def read_function_code(self, function_info: SymbolInfo) -> Optional[str]:
        """Read the full code of a function"""
        if not function_info or not function_info.file_path:
            return None

        result = self.client.read_file(
            function_info.file_path,
            function_info.start_line,
            function_info.end_line
        )

        if result.success:
            return result.data
        self.logger.error(f"Failed to read function code: {result.error}")
        return None

class SerenaCodeEditor:
    """High-level code editing using Serena tools"""

    def __init__(self, client: SerenaMCPClient, analyzer: SerenaCodeAnalyzer):
        self.client = client
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)

    def replace_function_body(self, function_name: str, file_path: str,
                            new_body: str) -> bool:
        """Replace the body of a function"""
        self.logger.info(f"Replacing function body: {function_name} in {file_path}")

        result = self.client.replace_symbol_body(function_name, file_path, new_body)
        if result.success:
            self.logger.info("Function body replaced successfully")
            return True
        else:
            self.logger.error(f"Failed to replace function body: {result.error}")
            return False

    def insert_code_after_function(self, function_name: str, file_path: str,
                                 code_to_insert: str) -> bool:
        """Insert code after a function definition"""
        self.logger.info(f"Inserting code after function: {function_name}")

        result = self.client.insert_after_symbol(function_name, file_path, code_to_insert)
        if result.success:
            self.logger.info("Code inserted successfully")
            return True
        else:
            self.logger.error(f"Failed to insert code: {result.error}")
            return False

    def insert_code_before_function(self, function_name: str, file_path: str,
                                  code_to_insert: str) -> bool:
        """Insert code before a function definition"""
        self.logger.info(f"Inserting code before function: {function_name}")

        result = self.client.insert_before_symbol(function_name, file_path, code_to_insert)
        if result.success:
            self.logger.info("Code inserted successfully")
            return True
        else:
            self.logger.error(f"Failed to insert code: {result.error}")
            return False

    def apply_semantic_edit(self, edit: SemanticEdit) -> bool:
        """Apply a semantic edit operation"""
        self.logger.info(f"Applying semantic edit: {edit.operation} on {edit.symbol_name}")

        if edit.operation == "replace_symbol_body":
            return self.replace_function_body(edit.symbol_name, edit.file_path, edit.new_code)
        elif edit.operation == "insert_after_symbol":
            return self.insert_code_after_function(edit.symbol_name, edit.file_path, edit.new_code)
        elif edit.operation == "insert_before_symbol":
            return self.insert_code_before_function(edit.symbol_name, edit.file_path, edit.new_code)
        else:
            self.logger.error(f"Unknown edit operation: {edit.operation}")
            return False

class SerenaIntegration:
    """Main integration class for Serena in GridBot automation"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize Serena components
        server_url = self.config.get('serena_server_url', 'http://localhost:24282')
        self.client = SerenaMCPClient(server_url=server_url)

        self.analyzer = SerenaCodeAnalyzer(self.client)
        self.editor = SerenaCodeEditor(self.client, self.analyzer)

        # Project path for Serena
        self.project_path = self.config.get('project_path', os.getcwd())

    def initialize(self) -> bool:
        """Initialize Serena integration"""
        self.logger.info("Initializing Serena integration...")

        # Check if server is running, start if needed
        if not self.client.is_server_running():
            self.logger.info("Serena server not running, attempting to start...")
            if not self.client.start_server(self.project_path):
                self.logger.error("Failed to start Serena server")
                return False

        # Activate project
        self.logger.info(f"Activating project: {self.project_path}")
        result = self.client.activate_project(self.project_path)
        if not result.success:
            self.logger.error(f"Failed to activate project: {result.error}")
            return False

        self.logger.info("Serena integration initialized successfully")
        return True

    def analyze_codebase(self, file_paths: List[str]) -> Dict[str, List[SymbolInfo]]:
        """Analyze symbols in multiple files"""
        self.logger.info(f"Analyzing codebase: {len(file_paths)} files")

        analysis_results = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                symbols = self.analyzer.analyze_file_symbols(file_path)
                analysis_results[file_path] = symbols
                self.logger.info(f"Found {len(symbols)} symbols in {file_path}")
            else:
                self.logger.warning(f"File not found: {file_path}")

        return analysis_results

    def find_buggy_functions(self, file_path: str, error_patterns: List[str]) -> List[SymbolInfo]:
        """Find functions that might contain bugs based on error patterns"""
        self.logger.info(f"Searching for potentially buggy functions in {file_path}")

        symbols = self.analyzer.analyze_file_symbols(file_path)
        potentially_buggy = []

        for symbol in symbols:
            if symbol.kind == 'function':
                if code := self.analyzer.read_function_code(symbol):
                    # Check for error patterns
                    for pattern in error_patterns:
                        if pattern.lower() in code.lower():
                            potentially_buggy.append(symbol)
                            break

        self.logger.info(f"Found {len(potentially_buggy)} potentially buggy functions")
        return potentially_buggy

    def apply_semantic_fix(self, file_path: str, function_name: str,
                          fix_code: str, operation: str = "replace_symbol_body") -> bool:
        """Apply a semantic fix to a function"""
        self.logger.info(f"Applying semantic fix to {function_name} in {file_path}")

        edit = SemanticEdit(
            operation=operation,
            symbol_name=function_name,
            file_path=file_path,
            new_code=fix_code
        )

        return self.editor.apply_semantic_edit(edit)

    def run_tests_with_semantic_context(self, test_command: str) -> Tuple[bool, str]:
        """Run tests with semantic context awareness"""
        self.logger.info("Running tests with semantic context")

        result = self.client.execute_shell_command(test_command, self.project_path)

        if not result.success:
            return False, result.error or "Test execution failed"
        output = result.data.get('output', '') if isinstance(result.data, dict) else str(result.data)
        return True, output

    def get_code_context_for_error(self, file_path: str, line_number: int,
                                 context_lines: int = 5) -> Dict:
        """Get semantic context around an error location"""
        self.logger.info(f"Getting code context around {file_path}:{line_number}")

        # Read file around the error line
        result = self.client.read_file(file_path, line_number - context_lines, line_number + context_lines)
        if not result.success:
            return {"error": result.error}

        # Find symbols in the context
        symbols_result = self.client.get_symbols_overview(file_path)
        symbols_in_context = []
        if symbols_result.success and isinstance(symbols_result.data, list):
            symbols_in_context.extend(
                symbol
                for symbol in symbols_result.data
                if (
                    symbol.get('start_line', 0)
                    <= line_number
                    <= symbol.get('end_line', 0)
                    or abs(symbol.get('start_line', 0) - line_number)
                    <= context_lines
                )
            )
        return {
            "code_context": result.data,
            "symbols_in_context": symbols_in_context,
            "line_number": line_number
        }

# Utility functions for integration
def create_serena_config(automation_config: Dict) -> Dict:
    """Create Serena-specific configuration from automation config"""
    return {
        'serena_server_url': automation_config.get('serena', {}).get('server_url', 'http://localhost:8000'),
        'project_path': automation_config.get('project_path', os.getcwd()),
        'enabled': automation_config.get('serena', {}).get('enabled', True),
        'auto_start_server': automation_config.get('serena', {}).get('auto_start_server', True),
        'timeout': automation_config.get('serena', {}).get('timeout', 30)
    }

def is_serena_available() -> bool:
    """Check if Serena is available and configured"""
    try:
        # Try to import required modules
        import requests
        return True
    except ImportError:
        return False
