"""
Optimization Automation System

This module handles the optimization phase after successful debugging,
analyzing code performance and applying LLM-generated optimizations.
"""

import os
import ast
import time
import logging
import json
import subprocess
import profile
import pstats
import cProfile
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile

try:
    # Try relative imports first (when run as module)
    from .llm_interface import SmollLLMInterface, LLMResponse
    from .qwen_agent_interface import QwenAgentInterface, LLMResponse as QwenLLMResponse
    from .automated_file_editor import SafeFileEditor, EditResult
except ImportError:
    # Fall back to absolute imports (when run as script)
    from llm_interface import SmollLLMInterface, LLMResponse
    from qwen_agent_interface import QwenAgentInterface, LLMResponse as QwenLLMResponse
    from automated_file_editor import SafeFileEditor, EditResult

@dataclass
class PerformanceMetrics:
    """Container for performance measurement data"""
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    function_calls: Optional[int] = None
    profile_data: Optional[Dict] = None

@dataclass
class OptimizationCandidate:
    """Container for code sections that could be optimized"""
    function_name: str
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    performance_issues: List[str]
    optimization_priority: int  # 1-10, 10 being highest priority
    estimated_impact: str  # "low", "medium", "high"
    optimization_metadata: Optional[Dict] = None  # Additional metadata for optimization

@dataclass
class OptimizationResult:
    """Container for optimization attempt results"""
    candidate: OptimizationCandidate
    success: bool
    original_performance: PerformanceMetrics
    optimized_performance: Optional[PerformanceMetrics]
    improvement_ratio: Optional[float]
    applied: bool
    error: Optional[str] = None

class CodeOptimizationAnalyzer:
    """Analyzes code for optimization opportunities"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_ast_for_optimizations(self, file_path: str) -> List[OptimizationCandidate]:
        """Analyze AST to find optimization opportunities"""
        candidates = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if candidate := self._analyze_function(
                        node, source_code, file_path
                    ):
                        candidates.append(candidate)
                elif isinstance(node, ast.For):
                    if candidate := self._analyze_loop(
                        node, source_code, file_path
                    ):
                        candidates.append(candidate)

        except Exception as e:
            self.logger.error(f"Error analyzing AST: {e}")

        return candidates
    
    def _analyze_function(self, func_node: ast.FunctionDef, source_code: str, 
                         file_path: str) -> Optional[OptimizationCandidate]:
        """Analyze a function for optimization opportunities"""
        issues = []
        priority = 1
        impact = "low"

        # Get function source lines
        lines = source_code.split('\n')
        start_line = func_node.lineno
        end_line = func_node.end_lineno or start_line

        # Check for common performance issues
        for node in ast.walk(func_node):
            # Nested loops
            if isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.For) and inner_node != node:
                        issues.append("Nested loops detected - consider optimization")
                        priority = max(priority, 6)
                        impact = "medium"

            # String concatenation in loops
            if isinstance(node, ast.For):
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, ast.AugAssign) and isinstance(inner_node.op, ast.Add) and self._is_string_operation(inner_node):
                        issues.append("String concatenation in loop - consider join()")
                        priority = max(priority, 7)
                        impact = "high"

            # Frequent function calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr in ['append', 'extend'] and self._is_in_loop(node, func_node)):
                issues.append("List operations in loop - consider list comprehension")
                priority = max(priority, 5)
                impact = "medium"

            # Large list/dict creations
            if isinstance(node, (ast.ListComp, ast.DictComp)) and len(ast.dump(node)) > 200:
                issues.append("Complex comprehension - consider breaking down")
                priority = max(priority, 4)

        # Check function length
        if end_line - start_line > 50:
            issues.append("Long function - consider refactoring")
            priority = max(priority, 3)

        if issues:
            func_code = '\n'.join(lines[start_line-1:end_line])

            return OptimizationCandidate(
                function_name=func_node.name,
                file_path=file_path,
                line_start=start_line,
                line_end=end_line,
                code_snippet=func_code,
                performance_issues=issues,
                optimization_priority=priority,
                estimated_impact=impact
            )

        return None
    
    def _analyze_loop(self, loop_node: ast.For, source_code: str, 
                     file_path: str) -> Optional[OptimizationCandidate]:
        """Analyze a loop for optimization opportunities"""
        issues = []
        priority = 1
        
        # Check for inefficient operations inside loop
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ['len', 'range', 'enumerate']:
                issues.append("Function call in loop condition")
                priority = max(priority, 4)
        
        if issues:
            lines = source_code.split('\n')
            start_line = loop_node.lineno
            end_line = loop_node.end_lineno or start_line
            loop_code = '\n'.join(lines[start_line-1:end_line])
            
            return OptimizationCandidate(
                function_name="<loop>",
                file_path=file_path,
                line_start=start_line,
                line_end=end_line,
                code_snippet=loop_code,
                performance_issues=issues,
                optimization_priority=priority,
                estimated_impact="medium"
            )
        
        return None
    
    def _is_string_operation(self, node: ast.AugAssign) -> bool:
        """Check if an augmented assignment is likely a string operation"""
        # This is a heuristic - in practice you'd want more sophisticated analysis
        return True  # Simplified for example
    
    def _is_in_loop(self, node: ast.AST, func_node: ast.FunctionDef) -> bool:
        """Check if a node is inside a loop"""
        for parent in ast.walk(func_node):
            if isinstance(parent, (ast.For, ast.While)):
                for child in ast.walk(parent):
                    if child == node:
                        return True
        return False

class PerformanceProfiler:
    """Profiles code execution for performance metrics"""
    
    def __init__(self, python_executable: str = "python"):
        self.python_executable = python_executable
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the profiler"""
        self.logger = logging.getLogger(__name__)
    
    def profile_file_execution(self, file_path: str) -> PerformanceMetrics:
        """Profile the execution of a Python file using the configured executable"""
        try:
            # Create a temporary profiling script
            profile_script = f"""
import cProfile
import pstats
import time
import sys
import os

# Add the file's directory to path
sys.path.insert(0, '{os.path.dirname(file_path)}')

start_time = time.time()

# Profile the execution
profiler = cProfile.Profile()
profiler.enable()

try:
    exec(open('{file_path}').read())
except Exception as e:
    print(f"Execution error: {{e}}")

profiler.disable()
end_time = time.time()

# Save profile data
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')

# Print basic metrics
print(f"EXECUTION_TIME:{{end_time - start_time}}")
print(f"TOTAL_CALLS:{{stats.total_calls}}")

# Save detailed profile
stats.dump_stats('profile_output.pstats')
"""
            
            # Write profile script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(profile_script)
                profile_script_path = f.name
            
            try:
                # Run the profiling script with longer timeout for GridBot
                result = subprocess.run(
                    [self.python_executable, profile_script_path],
                    capture_output=True,
                    text=True,
                    timeout=360  # 6 minutes - longer than GridBot's typical runtime
                )
                
                # Parse the output
                execution_time = 0.0
                total_calls = 0
                
                for line in result.stdout.split('\n'):
                    if line.startswith('EXECUTION_TIME:'):
                        execution_time = float(line.split(':')[1])
                    elif line.startswith('TOTAL_CALLS:'):
                        total_calls = int(line.split(':')[1])
                
                return PerformanceMetrics(
                    execution_time=execution_time,
                    function_calls=total_calls
                )
                
            finally:
                # Clean up temporary files
                if os.path.exists(profile_script_path):
                    os.remove(profile_script_path)
                if os.path.exists('profile_output.pstats'):
                    os.remove('profile_output.pstats')
        
        except Exception as e:
            self.logger.error(f"Error profiling file: {e}")
            return PerformanceMetrics(execution_time=0.0)

class LogCollector:
    """Collects and analyzes logs for log-driven optimization"""
    
    def __init__(self):
        self.setup_logging()
        self.log_patterns = {
            'performance': [
                r'execution time', r'cpu usage', r'memory usage', r'slow', r'performance',
                r'efficiency', r'benchmark', r'profiling'
            ],
            'bottlenecks': [
                r'bottleneck', r'blocking', r'waiting', r'delay', r'lag', r'timeout',
                r'hanging', r'freeze', r'stuck', r'inefficient'
            ],
            'resource_usage': [
                r'high memory', r'memory leak', r'cpu intensive', r'disk i/o',
                r'network latency', r'database slow', r'cache miss'
            ],
            'function_calls': [
                r'frequent calls', r'repeated operations', r'loop overhead',
                r'recursive calls', r'stack overflow', r'infinite loop'
            ]
        }
    
    def setup_logging(self):
        """Setup logging for the log collector"""
        self.logger = logging.getLogger(__name__)
    
    def collect_recent_logs(self, max_age_hours: int = 24) -> List[str]:
        """Enhanced log collection with real-time GridBot execution metrics"""
        import glob
        import os
        import time
        from datetime import datetime, timedelta

        self.logger.info(
            "[LOG_COLLECT] Collecting execution logs for optimization analysis"
        )

        log_files = []
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        # Enhanced log patterns to capture GridBot execution data
        log_patterns = [
            "*.log",
            "*_log.txt", 
            "*_report.json",
            "*_session.json",
            "debug_*.json",
            "optimization_*.txt",
            "gridbot*.log",
            "websocket*.log",
            "master_automation.log",
            "trading_*.log"
        ]

        # Search in multiple directories
        search_dirs = [
            ".",  # Current directory
            "../",  # Parent directory (main workspace)
            "logs/",  # Logs subdirectory
            "../logs/",  # Parent logs
            os.path.dirname(__file__),  # This script's directory
            os.path.dirname(os.path.dirname(__file__))  # Workspace directory
        ]

        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue

            for pattern in log_patterns:
                search_pattern = os.path.join(search_dir, pattern)
                for log_file in glob.glob(search_pattern):
                    if os.path.exists(log_file) and os.path.isfile(log_file):
                        try:
                            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                            if file_time > cutoff_time:
                                log_files.append(log_file)
                        except OSError:
                            continue

        # Remove duplicates and sort by modification time (newest first)
        unique_files = list(set(log_files))
        unique_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        self.logger.info(f"[LOG_COLLECT] Found {len(unique_files)} recent log files")

        if runtime_metrics := self._collect_runtime_metrics():
            # Create a temporary metrics file
            metrics_file = "temp_runtime_metrics.log"
            with open(metrics_file, 'w') as f:
                f.write('\\n'.join(runtime_metrics))
            unique_files.insert(0, metrics_file)  # Add at beginning for priority

        return unique_files
    
    def _collect_runtime_metrics(self) -> List[str]:
        """Collect real-time performance metrics from running GridBot processes"""
        metrics = []

        try:
            import psutil
            current_time = time.time()

            self.logger.info("[RUNTIME] Scanning for active GridBot processes")

            # Find GridBot-related processes
            gridbot_processes = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('GridbotBackup.py' in str(cmd) or 'gridbot_websocket_server.py' in str(cmd) for cmd in cmdline):
                        gridbot_processes += 1

                        # Get detailed process metrics
                        memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                        cpu_percent = proc.cpu_percent()

                        process_name = 'GridbotBackup.py' if 'GridbotBackup.py' in str(cmdline) else 'gridbot_websocket_server.py'

                        metrics.append(f"RUNTIME_METRIC: {process_name} - PID: {proc.info['pid']}, Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")

                        # Get additional process stats if available
                        try:
                            with proc.oneshot():
                                create_time = proc.create_time()
                                runtime_hours = (current_time - create_time) / 3600
                                metrics.append(f"RUNTIME_METRIC: {process_name} - Runtime: {runtime_hours:.1f}h")

                                # Get I/O stats
                                io_counters = proc.io_counters()
                                read_mb = io_counters.read_bytes / (1024 * 1024)
                                write_mb = io_counters.write_bytes / (1024 * 1024)
                                metrics.append(f"RUNTIME_METRIC: {process_name} - I/O: Read {read_mb:.1f}MB, Write {write_mb:.1f}MB")

                                # Get network connections for WebSocket server
                                if 'websocket' in process_name.lower():
                                    try:
                                        connections = proc.connections()
                                        active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
                                        listening_ports = len([c for c in connections if c.status == 'LISTEN'])
                                        metrics.append(f"RUNTIME_METRIC: {process_name} - Connections: {active_connections} active, {listening_ports} listening")
                                    except (psutil.AccessDenied, AttributeError):
                                        pass

                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if gridbot_processes > 0:
                self.logger.info(f"[RUNTIME] ✅ Found {gridbot_processes} active GridBot processes")
            else:
                self.logger.info("[RUNTIME] No active GridBot processes detected")

        except ImportError:
            metrics.append("RUNTIME_METRIC: psutil not available - install for enhanced monitoring")
            self.logger.warning("[RUNTIME] psutil not available for process monitoring")
        except Exception as e:
            metrics.append(f"RUNTIME_METRIC_ERROR: {e}")
            self.logger.warning(f"[RUNTIME] Error collecting runtime metrics: {e}")

        return metrics
    
    def extract_log_content(self, log_files: List[str], max_lines_per_file: int = 1000) -> str:
        """Extract relevant content from log files"""
        all_content = []
        total_lines = 0
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                    # Take the most recent lines first
                    recent_lines = lines[-max_lines_per_file:] if len(lines) > max_lines_per_file else lines
                    
                    # Filter for performance-related content but exclude optimization logs to prevent feedback loop
                    relevant_lines = []
                    for line in recent_lines:
                        line_lower = line.lower()
                        # Skip optimization-related log lines to prevent feedback loop
                        if any(skip_pattern in line_lower for skip_pattern in [
                            'applied log-driven optimization', 'auto-applying log-driven optimization',
                            'optimization completed', 'optimization phase', 'optimization result',
                            'optimization system', 'optimization candidate', 'optimization automation',
                            'logperformanceissue', 'log-driven optimization', 'optimization for',
                            'applied optimization', 'optimization report', 'multi-file optimization'
                        ]):
                            continue
                        
                        # Also skip lines from optimization automation system logger
                        if 'optimization_automation_system' in line_lower:
                            continue
                            
                        if any(any(pattern in line_lower for pattern in patterns) 
                              for patterns in self.log_patterns.values()):
                            relevant_lines.append(line.strip())
                    
                    if relevant_lines:
                        all_content.append(f"=== {log_file} ===")
                        all_content.extend(relevant_lines[:200])  # Limit per file
                        all_content.append("")
                        
                        total_lines += len(relevant_lines)
                        
            except Exception as e:
                self.logger.warning(f"Error reading log file {log_file}: {e}")
        
        content = "\n".join(all_content)
        self.logger.info(f"Extracted {total_lines} relevant log lines from {len(log_files)} files")
        return content
    
    def analyze_logs_for_optimization_opportunities(self, log_content: str) -> Dict[str, Any]:
        """Analyze log content to identify real optimization opportunities with performance metrics"""
        analysis = {
            'performance_issues': [],
            'bottlenecks': [],
            'resource_problems': [],
            'function_patterns': [],
            'recommendations': [],
            'execution_metrics': {},
            'error_patterns': [],
            'optimization_targets': []
        }

        lines = log_content.split('\n')

        # Track execution timing patterns
        execution_times = []
        memory_usage = []
        error_counts = {}
        function_calls = {}

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Extract timestamp and execution timing
            if 'took' in line_stripped and any(unit in line_stripped for unit in ['ms', 'seconds', 'minutes']):
                try:
                    # Parse execution time from logs like "Operation took 1.23 seconds"
                    import re
                    if time_match := re.search(
                        r'(\d+\.?\d*)\s*(ms|seconds?|minutes?)', line_stripped
                    ):
                        time_val = float(time_match.group(1))
                        time_unit = time_match.group(2)

                        # Convert to milliseconds
                        if 'second' in time_unit:
                            time_val *= 1000
                        elif 'minute' in time_unit:
                            time_val *= 60000

                        execution_times.append(time_val)

                        # Flag slow operations (>5 seconds)
                        if time_val > 5000:
                            analysis['bottlenecks'].append({
                                'type': 'slow_execution',
                                'duration_ms': time_val,
                                'log_line': line_stripped,
                                'severity': 'high' if time_val > 10000 else 'medium'
                            })
                except Exception:
                    pass

            # Track memory usage patterns
            if 'memory' in line_stripped.lower() or 'mb' in line_stripped.lower():
                try:
                    import re
                    if mem_match := re.search(
                        r'(\d+\.?\d*)\s*mb', line_stripped.lower()
                    ):
                        mem_val = float(mem_match.group(1))
                        memory_usage.append(mem_val)

                        # Flag high memory usage (>500MB)
                        if mem_val > 500:
                            analysis['resource_problems'].append({
                                'type': 'high_memory',
                                'memory_mb': mem_val,
                                'log_line': line_stripped,
                                'severity': 'high' if mem_val > 1000 else 'medium'
                            })
                except Exception:
                    pass

            # Track error patterns
            if any(error_term in line_stripped.lower() for error_term in ['error', 'exception', 'failed', 'timeout']):
                error_type = 'unknown'
                if 'timeout' in line_stripped.lower():
                    error_type = 'timeout'
                elif 'connection' in line_stripped.lower():
                    error_type = 'connection'
                elif 'memory' in line_stripped.lower():
                    error_type = 'memory'
                elif 'syntax' in line_stripped.lower():
                    error_type = 'syntax'

                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                analysis['error_patterns'].append({
                    'type': error_type,
                    'log_line': line_stripped,
                    'frequency': error_counts[error_type]
                })

            # Track function call patterns
            if 'calling' in line_stripped.lower() or 'function' in line_stripped.lower():
                try:
                    import re
                    if func_match := re.search(r'(\w+)\(', line_stripped):
                        func_name = func_match.group(1)
                        function_calls[func_name] = function_calls.get(func_name, 0) + 1

                        # Flag frequently called functions (>100 calls)
                        if function_calls[func_name] > 100:
                            analysis['function_patterns'].append({
                                'type': 'frequent_calls',
                                'function': func_name,
                                'call_count': function_calls[func_name],
                                'optimization_potential': 'high'
                            })
                except Exception:
                    pass

        # Calculate performance metrics
        if execution_times:
            analysis['execution_metrics'] = {
                'avg_execution_time_ms': sum(execution_times) / len(execution_times),
                'max_execution_time_ms': max(execution_times),
                'total_operations': len(execution_times),
                'slow_operations_count': len([t for t in execution_times if t > 5000])
            }

        if memory_usage:
            analysis['execution_metrics']['avg_memory_mb'] = sum(memory_usage) / len(memory_usage)
            analysis['execution_metrics']['max_memory_mb'] = max(memory_usage)

        # Generate intelligent optimization targets
        self._generate_optimization_targets(analysis)

        return analysis
    
    def _generate_optimization_targets(self, analysis: Dict[str, Any]):
        """Generate specific optimization targets based on analysis"""
        targets = []

        # Performance-based targets
        if analysis['execution_metrics'].get('avg_execution_time_ms', 0) > 3000:
            targets.append({
                'type': 'performance_optimization',
                'target': 'reduce_execution_time',
                'priority': 'high',
                'description': f"Average execution time {analysis['execution_metrics']['avg_execution_time_ms']:.0f}ms is above threshold",
                'suggested_techniques': ['caching', 'algorithm_optimization', 'parallel_processing']
            })

        # Memory optimization targets
        if analysis['execution_metrics'].get('max_memory_mb', 0) > 500:
            targets.append({
                'type': 'memory_optimization',
                'target': 'reduce_memory_usage',
                'priority': 'medium',
                'description': f"Peak memory usage {analysis['execution_metrics']['max_memory_mb']:.0f}MB exceeds threshold",
                'suggested_techniques': ['object_pooling', 'lazy_loading', 'memory_profiling']
            })

        if error_types := {
            error['type']
            for error in analysis['error_patterns']
            if error['frequency'] > 5
        }:
            targets.append({
                'type': 'error_handling',
                'target': 'improve_error_resilience',
                'priority': 'high',
                'description': f"Frequent errors detected: {', '.join(error_types)}",
                'suggested_techniques': ['retry_mechanisms', 'circuit_breakers', 'graceful_degradation']
            })

        if frequent_functions := [
            fp
            for fp in analysis['function_patterns']
            if fp.get('call_count', 0) > 50
        ]:
            targets.append({
                'type': 'function_optimization',
                'target': 'optimize_hot_paths',
                'priority': 'medium',
                'description': f"Functions with high call frequency detected: {len(frequent_functions)} candidates",
                'suggested_techniques': ['memoization', 'inlining', 'vectorization']
            })

        analysis['optimization_targets'] = targets

class OptimizationAutomationSystem:
    """Main system for automated code optimization"""
    
    def __init__(self,
                 llm_interface: QwenAgentInterface = None,
                 file_editor: SafeFileEditor = None,
                 min_improvement_threshold: float = 0.05,  # 5% minimum improvement
                 python_executable: str = None,
                 optimization_mode: str = "static"):  # "static" or "log-driven"
        
        # Setup logging first so it's available for detection
        self.setup_logging()
        
        self.llm_interface = llm_interface or QwenAgentInterface()
        self.file_editor = file_editor or SafeFileEditor()
        self.analyzer = CodeOptimizationAnalyzer()
        
        # Use virtual environment Python if available
        if python_executable is None:
            python_executable = self._detect_python_executable()
        self.python_executable = python_executable
        
        self.profiler = PerformanceProfiler(python_executable=self.python_executable)
        self.min_improvement_threshold = min_improvement_threshold
        
        # New: Optimization mode (static vs log-driven)
        self.optimization_mode = optimization_mode
        # Always provide a log_collector to avoid None checks downstream
        self.log_collector = LogCollector()
        
        self.optimization_history = []
    
    def setup_logging(self):
        """Setup logging for the optimization system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimization_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _detect_python_executable(self) -> str:
        """Detect the correct Python executable, preferring virtual environment"""
        import sys
        import os

        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # First try to use the current Python executable (if in venv)
            current_python = sys.executable

            self.logger.info(f"Using virtual environment Python: {current_python}")
            return current_python

        # Try common virtual environment locations
        venv_paths = [
            r"C:\Users\805Sk\GridBotWorkspace\.venv\Scripts\python.exe",
            "python"
        ]

        for python_path in venv_paths:
            if os.path.exists(python_path):
                self.logger.info(f"Found Python executable: {python_path}")
                return python_path

        self.logger.warning("Using system Python as fallback")
        return "python"
    
    def analyze_file_for_optimizations(self, file_path: str) -> List[OptimizationCandidate]:
        """Analyze a file and find optimization opportunities"""
        self.logger.info(f"Analyzing {file_path} for optimization opportunities")
        
        candidates = self.analyzer.analyze_ast_for_optimizations(file_path)
        
        # Sort by priority (highest first)
        candidates.sort(key=lambda c: c.optimization_priority, reverse=True)
        
        self.logger.info(f"Found {len(candidates)} optimization candidates")
        
        return candidates
    
    def generate_optimization(self, candidate: OptimizationCandidate) -> LLMResponse:
        """Generate an optimization for a candidate using LLM"""
        # Create context for the LLM
        context = f"""
Performance Issues Identified:
{chr(10).join(f"- {issue}" for issue in candidate.performance_issues)}

Priority: {candidate.optimization_priority}/10
Estimated Impact: {candidate.estimated_impact}

Current implementation:
```python
{candidate.code_snippet}
```

Please analyze this code and provide an optimized version that addresses the identified performance issues.
Focus on improving execution speed, memory usage, and overall efficiency while maintaining functionality.
"""
        
        return self.llm_interface.generate_optimization(candidate.code_snippet, context)
    
    def extract_optimization_targets(self, candidate: OptimizationCandidate, file_path: str) -> List[Dict]:
        """
        Intelligently extract targeted sections from massive functions for optimization.
        Specifically designed to handle massive functions like run_bot() by identifying
        logical sections and preserving context.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            function_lines = lines[candidate.line_start-1:candidate.line_end]
            
            # Detect if this is the massive run_bot function
            is_run_bot = 'def run_bot(' in '\n'.join(function_lines[:5])
            
            if is_run_bot:
                return self.extract_run_bot_sections(function_lines, candidate.line_start)
            else:
                return self.extract_generic_function_sections(function_lines, candidate.line_start)
                
        except Exception as e:
            self.logger.error(f"Error extracting optimization targets: {e}")
            return []
    
    def extract_run_bot_sections(self, function_lines: List[str], start_line: int) -> List[Dict]:
        """Extract logical sections from the massive run_bot function"""
        sections = []
        current_section = []
        current_section_name = "Initialization"
        section_start_line = start_line
        
        # Known section markers in run_bot function
        section_markers = {
            "# 1. Fetch Market Data": "Market Data Fetching",
            "# 2. Update Balances": "Balance Updates", 
            "# 3. Refresh Data for ML Predictions": "ML Data Refresh",
            "# 4. Retraining": "Model Retraining",
            "# 5. Check Orders": "Order Management",
            "# 6. Feature-Based Trading Logic": "Feature Trading",
            "# 7. Adjust Parameters": "Parameter Adjustment",
            "# 8. Send WebSocket Updates": "WebSocket Communication",
            "# 9. Process WebSocket Commands": "Command Processing",
            "# 10. Check Grid Stagnation": "Grid Management",
            "# 11. Update Bot State": "State Updates"
        }
        
        for i, line in enumerate(function_lines):
            # Check for section markers
            for marker, section_name in section_markers.items():
                if marker in line:
                    # Save previous section if it has substantial content
                    if current_section and len(current_section) > 10:
                        sections.append({
                            'name': current_section_name,
                            'code': '\n'.join(current_section),
                            'line_range': (section_start_line, start_line + i - 1),
                            'size': len(current_section)
                        })
                    
                    # Start new section
                    current_section = [line]
                    current_section_name = section_name
                    section_start_line = start_line + i
                    break
            else:
                current_section.append(line)
        
        # Add final section
        if current_section and len(current_section) > 10:
            sections.append({
                'name': current_section_name,
                'code': '\n'.join(current_section),
                'line_range': (section_start_line, start_line + len(function_lines) - 1),
                'size': len(current_section)
            })
        
        # Also extract nested loops and complex blocks
        sections.extend(self.extract_nested_loops(function_lines, start_line))
        sections.extend(self.extract_string_operations(function_lines, start_line))
        sections.extend(self.extract_repeated_calls(function_lines, start_line))
        
        # Sort by potential optimization impact (larger sections first)
        sections.sort(key=lambda x: x['size'], reverse=True)
        
        self.logger.info(f"Extracted {len(sections)} sections from run_bot function")
        for section in sections[:5]:  # Log top 5 sections
            self.logger.info(f"  Section: {section['name']} ({section['size']} lines)")
        
        return sections[:8]  # Return top 8 sections for optimization
    
    def extract_generic_function_sections(self, function_lines: List[str], start_line: int) -> List[Dict]:
        """Extract sections from regular functions"""
        sections = []
        
        # Extract by logical blocks (try/except, if/else, loops)
        sections.extend(self.extract_nested_loops(function_lines, start_line))
        sections.extend(self.extract_string_operations(function_lines, start_line))
        sections.extend(self.extract_repeated_calls(function_lines, start_line))
        
        # If function is still large, split by complexity
        if len(function_lines) > 50:
            chunk_size = 30
            for i in range(0, len(function_lines), chunk_size):
                chunk = function_lines[i:i+chunk_size]
                if len(chunk) > 10:
                    sections.append({
                        'name': f"Function Block {i//chunk_size + 1}",
                        'code': '\n'.join(chunk),
                        'line_range': (start_line + i, start_line + i + len(chunk) - 1),
                        'size': len(chunk)
                    })
        
        return sections[:6]  # Return top 6 sections
    
    def extract_nested_loops(self, lines: List[str], start_line: int) -> List[Dict]:
        """Extract nested loops and complex iterations"""
        sections = []
        in_loop = False
        loop_start = 0
        loop_lines = []
        indent_level = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())
            
            # Detect loop start
            if any(keyword in stripped for keyword in ['for ', 'while ']):
                if not in_loop:
                    in_loop = True
                    loop_start = i
                    loop_lines = [line]
                    indent_level = current_indent
                else:
                    loop_lines.append(line)
            elif in_loop:
                if current_indent > indent_level or stripped.startswith(('if ', 'elif ', 'else:', 'try:', 'except', 'finally:')):
                    loop_lines.append(line)
                elif current_indent <= indent_level and stripped and not stripped.startswith('#'):
                    # End of loop block
                    if len(loop_lines) > 5:  # Only consider substantial loops
                        sections.append({
                            'name': f"Loop Block (lines {start_line + loop_start + 1}-{start_line + i})",
                            'code': '\n'.join(loop_lines),
                            'line_range': (start_line + loop_start, start_line + i - 1),
                            'size': len(loop_lines)
                        })
                    in_loop = False
                    loop_lines = []
                else:
                    loop_lines.append(line)
        
        # Handle loop at end of function
        if in_loop and len(loop_lines) > 5:
            sections.append({
                'name': f"Loop Block (lines {start_line + loop_start + 1}-{start_line + len(lines)})",
                'code': '\n'.join(loop_lines),
                'line_range': (start_line + loop_start, start_line + len(lines) - 1),
                'size': len(loop_lines)
            })
        
        return sections
    
    def extract_string_operations(self, lines: List[str], start_line: int) -> List[Dict]:
        """Extract string concatenation and formatting operations"""
        sections = []
        string_block = []
        block_start = 0
        
        for i, line in enumerate(lines):
            if any(op in line for op in ['.join(', '+=', 'f"', '".format(', '%s', '%d', '%f']):
                if not string_block:
                    block_start = i
                string_block.append(line)
            elif string_block:
                if len(string_block) > 3:  # Only consider substantial string operations
                    sections.append({
                        'name': f"String Operations (lines {start_line + block_start + 1}-{start_line + i})",
                        'code': '\n'.join(string_block),
                        'line_range': (start_line + block_start, start_line + i - 1),
                        'size': len(string_block)
                    })
                string_block = []
        
        return sections
    
    def extract_repeated_calls(self, lines: List[str], start_line: int) -> List[Dict]:
        """Extract blocks with repeated function calls that could be optimized"""
        sections = []
        call_patterns = {}
        
        # Track repeated patterns
        for i, line in enumerate(lines):
            # Look for repeated logger calls, API calls, etc.
            if any(pattern in line for pattern in ['logger.', 'exchange.', '.fetch_', '.create_', 'bot_state']):
                for pattern in ['logger.', 'exchange.', '.fetch_', '.create_', 'bot_state']:
                    if pattern in line:
                        if pattern not in call_patterns:
                            call_patterns[pattern] = []
                        call_patterns[pattern].append((i, line))
        
        # Extract blocks with high call frequency
        for pattern, calls in call_patterns.items():
            if len(calls) > 5:  # Only patterns with multiple occurrences
                # Group consecutive calls
                groups = []
                current_group = [calls[0]]
                
                for j in range(1, len(calls)):
                    if calls[j][0] - current_group[-1][0] <= 10:  # Within 10 lines
                        current_group.append(calls[j])
                    else:
                        if len(current_group) > 2:
                            groups.append(current_group)
                        current_group = [calls[j]]
                
                if len(current_group) > 2:
                    groups.append(current_group)
                
                # Create sections for significant groups
                for group in groups:
                    if len(group) > 3:
                        start_idx = group[0][0]
                        end_idx = group[-1][0]
                        section_lines = lines[start_idx:end_idx+1]
                        
                        sections.append({
                            'name': f"Repeated {pattern.rstrip('.')} Calls (lines {start_line + start_idx + 1}-{start_line + end_idx + 1})",
                            'code': '\n'.join(section_lines),
                            'line_range': (start_line + start_idx, start_line + end_idx),
                            'size': len(section_lines)
                        })
        
        return sections
    
    def combine_optimization_results(self, optimized_sections: List[Dict]) -> str:
        """Combine multiple section optimizations into coherent recommendations"""
        combined = [
            "# SECTION-BASED OPTIMIZATION RECOMMENDATIONS",
            "# Multiple code sections have been analyzed and optimized separately",
            "",
        ]
        for i, section in enumerate(optimized_sections, 1):
            combined.append(f"## Section {i}: {section['original']['name']}")
            combined.append(f"Lines {section['line_range'][0]}-{section['line_range'][1]}")
            combined.append("")
            combined.append("### Optimized Code:")
            combined.append("```python")
            combined.append(section['optimized'])
            combined.append("```")
            combined.append("")

        combined.extend(
            (
                "# INTEGRATION NOTES:",
                "# - Each section can be applied independently",
            )
        )
        combined.extend(
            (
                "# - Test each optimization separately before combining",
                "# - Maintain function context and variable scope",
            )
        )
        combined.append("# - Preserve error handling and logging")

        return '\n'.join(combined)
    
    def generate_targeted_optimization(self, candidate: OptimizationCandidate, file_path: str) -> LLMResponse:
        """Generate optimization using targeted code extraction for better accuracy"""
        import signal
        import threading

        # Set up timeout handling
        result = {'response': None, 'error': None}

        def optimization_worker():
            try:
                # For large functions, extract the most important section instead of processing multiple sections
                if len(candidate.code_snippet.split('\n')) > 100:  # If function is too large
                    self.logger.info(f"Large function detected ({len(candidate.code_snippet.split('\n'))} lines), using targeted single-section extraction")

                    # Extract targeted optimization sections
                    targeted_sections = self.extract_optimization_targets(candidate, file_path)

                    if targeted_sections:
                        # Select the most important section (largest or highest priority)
                        best_section = max(targeted_sections, key=lambda x: x['size'])
                        self.logger.info(f"Selected best section for optimization: {best_section['name']} ({best_section['size']} lines)")

                        # Optimize only this best section
                        section_response = self.llm_interface.generate_targeted_optimization(
                            best_section['code'], candidate.performance_issues
                        )

                        if getattr(section_response, 'success', False):
                            # Return the section optimization directly
                            # Normalize to current LLMResponse if needed
                            try:
                                if isinstance(section_response, LLMResponse):
                                    result['response'] = section_response
                                else:
                                    result['response'] = LLMResponse(
                                        content=getattr(section_response, 'content', ''),
                                        success=getattr(section_response, 'success', False),
                                        error=getattr(section_response, 'error', None)
                                    )
                            except Exception:
                                result['response'] = None
                            return
                        else:
                            self.logger.warning("Section optimization failed, falling back to full function")

                    # Fallback to original approach if targeted extraction fails
                    self.logger.warning("Targeted optimization failed, falling back to full function")

                # Original approach for smaller functions or fallback
                context = f"""
Optimization target: {candidate.function_name}
Performance issues identified:
{chr(10).join(f'- {issue}' for issue in candidate.performance_issues)}
Estimated impact: {candidate.estimated_impact}
Priority: {candidate.optimization_priority}

Focus on the specific performance issues mentioned above.
Code to optimize:
{candidate.code_snippet}

Please analyze this code and provide an optimized version that addresses the identified performance issues.
Focus on improving execution speed, memory usage, and overall efficiency while maintaining functionality.

CRITICAL: If no meaningful optimizations are possible, respond with "NO_OPTIMIZATIONS_NEEDED" only."""

                result['response'] = self.llm_interface.generate_optimization(candidate.code_snippet, context)

            except Exception as e:
                self.logger.error(f"Error in optimization worker thread: {e}")
                result['error'] = str(e)

        # Start optimization in a separate thread with dynamic timeout
        base_timeout = 120  # 2 minutes base timeout (reduced from 300)
        self.logger.info(f"Starting optimization for {candidate.function_name} with {base_timeout}s base timeout (dynamic extension enabled)")
        worker_thread = threading.Thread(target=optimization_worker)
        worker_thread.daemon = True
        worker_thread.start()

        # Dynamic timeout monitoring
        check_interval = 15  # Check every 15 seconds
        total_wait_time = 0
        response_detected = False

        while worker_thread.is_alive() and total_wait_time < base_timeout:
            worker_thread.join(timeout=check_interval)
            total_wait_time += check_interval

            # Check if model has started responding - improved detection
            if not response_detected:
                # If worker thread has been running for >30s, assume model started responding
                response_value = result.get('response') if isinstance(result, dict) else None
                if total_wait_time > 30 or response_value:
                    response_detected = True
                    extension = 600  # 10 minutes extension for streaming responses
                    base_timeout += extension
                    self.logger.info(f"Model responding for {candidate.function_name} - extending timeout by {extension}s")

            if total_wait_time % 60 == 0:  # Log progress every minute
                self.logger.info(f"Optimization in progress for {candidate.function_name}... {total_wait_time}s elapsed")

        if worker_thread.is_alive():
            self.logger.error(f"Optimization for {candidate.function_name} timed out after {total_wait_time} seconds")
            return LLMResponse(content="", success=False, error=f"Optimization timed out after {total_wait_time} seconds")

        if result['error']:
            return LLMResponse(content="", success=False, error=result['error'])

        if result['response']:
            resp = result['response']
            if isinstance(resp, LLMResponse):
                return resp
            return LLMResponse(
                content=getattr(resp, 'content', ''),
                success=getattr(resp, 'success', False),
                error=getattr(resp, 'error', None)
            )

        return LLMResponse(content="", success=False, error="No response from optimization worker")
    
    def test_optimization(self, original_file: str, optimized_code: str, 
                         candidate: OptimizationCandidate) -> Tuple[bool, PerformanceMetrics, PerformanceMetrics]:
        """Test an optimization by comparing performance"""
        try:
            # Create temporary files for testing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Read original file and replace the optimized section
                with open(original_file, 'r', encoding='utf-8') as orig_f:
                    original_content = orig_f.read()
                
                # Replace the candidate section with optimized code
                lines = original_content.split('\n')
                new_lines = (lines[:candidate.line_start-1] + 
                           optimized_code.split('\n') + 
                           lines[candidate.line_end:])
                
                f.write('\n'.join(new_lines))
                optimized_file = f.name
            
            try:
                # Profile original performance
                original_metrics = self.profiler.profile_file_execution(original_file)
                
                # Profile optimized performance
                optimized_metrics = self.profiler.profile_file_execution(optimized_file)
                
                # Check if optimization is valid (file still runs)
                if optimized_metrics.execution_time == 0.0:
                    return False, original_metrics, optimized_metrics
                
                return True, original_metrics, optimized_metrics
                
            finally:
                # Clean up temporary file
                if os.path.exists(optimized_file):
                    os.remove(optimized_file)
        
        except Exception as e:
            self.logger.error(f"Error testing optimization: {e}")
            return False, PerformanceMetrics(0.0), PerformanceMetrics(0.0)
    

    
    def save_optimization_candidate(self, candidate: OptimizationCandidate, llm_response: str, optimized_code: str) -> str:
        """Save optimization candidate details to a file for review"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_candidate_{candidate.function_name}_{timestamp}.txt"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"OPTIMIZATION CANDIDATE: {candidate.function_name}\n")
                f.write(f"File: {candidate.file_path}\n")
                f.write(f"Lines: {candidate.line_start}-{candidate.line_end}\n")
                f.write(f"Priority: {candidate.optimization_priority}/10\n")
                f.write(f"Estimated Impact: {candidate.estimated_impact}\n")
                f.write("\nPERFORMANCE ISSUES:\n")
                for issue in candidate.performance_issues:
                    f.write(f"- {issue}\n")

                f.write("\nORIGINAL CODE:\n")
                f.write("-" * 50 + "\n")
                f.write(candidate.code_snippet)
                f.write("\n" + "-" * 50 + "\n")

                f.write("\nLLM RESPONSE:\n")
                f.write("-" * 50 + "\n")
                f.write(llm_response)
                f.write("\n" + "-" * 50 + "\n")

                if optimized_code:
                    f.write("\nEXTRACTED OPTIMIZED CODE:\n")
                    f.write("-" * 50 + "\n")
                    f.write(optimized_code)
                    f.write("\n" + "-" * 50 + "\n")
                else:
                    f.write("\nNO OPTIMIZED CODE EXTRACTED\n")

            return filename

        except Exception as e:
            self.logger.error(f"Error saving optimization candidate: {e}")
            return ""

    def _optimize_file_static(self, file_path: str) -> List[OptimizationResult]:
        """Static optimization process for a single file"""
        self.logger.info(f"Static optimization for {file_path}")

        results = []

        # Get baseline performance
        baseline_performance = self.profiler.profile_file_execution(file_path)
        self.logger.info(f"Baseline execution time: {baseline_performance.execution_time:.4f}s")

        # Find optimization candidates
        candidates = self.analyze_file_for_optimizations(file_path)

        if not candidates:
            self.logger.info("No optimization candidates found")
            return results

        # Process candidates by priority (high to low)
        sorted_candidates = sorted(candidates, key=lambda x: x.optimization_priority, reverse=True)

        for i, candidate in enumerate(sorted_candidates[:10]):  # Limit to top 10 candidates
            self.logger.info(f"[OPTIMIZE] Processing optimization {i+1}/10 for {candidate.function_name} "
                           f"(priority: {candidate.optimization_priority})")
            self.logger.info(f"[INFO] Function: {candidate.function_name} | Lines: {candidate.line_end - candidate.line_start + 1} | Issues: {len(candidate.performance_issues)}")
            self.logger.info(f"[ISSUES] Performance Issues: {', '.join(candidate.performance_issues)}")

            # Add timeout protection for each candidate
            import time
            import threading
            start_time = time.time()

            # Use a thread-based timeout for the entire optimization process
            result_container = {'result': None, 'error': None}

            def optimization_worker():
                try:
                    # Generate optimization with priority-based timeout
                    timeout_multiplier = 1.0 if candidate.optimization_priority >= 6 else 0.5
                    self.logger.info(f"[START] Calling LLM for optimization of {candidate.function_name}...")
                    self.logger.info(f"[PRIORITY] Priority: {candidate.optimization_priority}, Timeout Multiplier: {timeout_multiplier}")
                    llm_response = self.generate_targeted_optimization(candidate, file_path)
                    result_container['result'] = llm_response
                except Exception as e:
                    result_container['error'] = str(e)

            # Start worker thread with dynamic timeout system
            worker_thread = threading.Thread(target=optimization_worker, daemon=True)
            worker_thread.start()

            # Dynamic timeout monitoring with improved response detection
            check_interval = 10  # Check every 10 seconds

            total_wait_time = 0
            response_detected = False
            last_response_check = time.time()
            # Define timeouts for this scope (fix NameError)
            base_timeout = 120  # 2 minutes base timeout (reduced from 300)
            extension_timeout = 600  # extend by 10 minutes when activity detected

            while worker_thread.is_alive() and total_wait_time < base_timeout:
                worker_thread.join(timeout=check_interval)
                total_wait_time += check_interval

                # Check if model has started responding (result container has data)
                # Also check if enough time has passed to indicate response started
                current_time = time.time()
                if not response_detected and (total_wait_time > 30 or result_container.get('result')):
                    response_detected = True
                    # Give much more time once model starts responding (for streaming)
                    extension_timeout = 600  # 10 minutes extension for streaming responses
                    base_timeout += extension_timeout
                    self.logger.info(f"Model started responding for {candidate.function_name} - extending timeout by {extension_timeout}s")

                if total_wait_time % 60 == 0:  # Log every minute
                    self.logger.info(f"[PROGRESS] Optimization for {candidate.function_name} running... {total_wait_time}s elapsed (timeout: {base_timeout}s)")
                elif total_wait_time % 30 == 0:  # Quick update every 30 seconds
                    self.logger.info(f"[WORKING] Still optimizing {candidate.function_name}... {total_wait_time}s")

            if worker_thread.is_alive():
                self.logger.error(f"Optimization for {candidate.function_name} timed out after {total_wait_time}s - skipping")
                continue

            if result_container['error']:
                self.logger.error(f"Optimization worker error for {candidate.function_name}: {result_container['error']}")
                continue

            llm_response = result_container['result']
            if not llm_response:
                self.logger.error(f"No response received for {candidate.function_name}")
                continue

            try:
                elapsed_time = time.time() - start_time
                self.logger.info(f"LLM response received for {candidate.function_name} after {elapsed_time:.1f}s")

                if not llm_response.success:
                    self.logger.warning(f"LLM optimization failed for {candidate.function_name}: {llm_response.error}")
                    result = OptimizationResult(
                        candidate=candidate,
                        success=False,
                        original_performance=baseline_performance,
                        optimized_performance=None,
                        improvement_ratio=None,
                        applied=False,
                        error=f"LLM optimization failed: {llm_response.error}"
                    )
                    results.append(result)
                    continue

                # Extract optimized code with import filtering
                optimized_code = self.llm_interface.extract_code_from_response(llm_response.content)

                self.logger.info(f"[OPTIMIZATION] Extracted code length: {len(optimized_code) if optimized_code else 0}")
                if optimized_code:
                    self.logger.info(f"[OPTIMIZATION] Code preview: {optimized_code[:200]}...")
                else:
                    self.logger.warning("[OPTIMIZATION] No code extracted from response")
                    self.logger.info(f"[OPTIMIZATION] Response preview: {llm_response.content[:500]}...")

                # Check if the optimized code is the same as the original (meaningless optimization)
                if optimized_code and self._is_same_code(candidate.code_snippet, optimized_code):
                    self.logger.warning(f"[OPTIMIZATION] LLM returned identical code for {candidate.function_name} - skipping meaningless optimization")
                    result = OptimizationResult(
                        candidate=candidate,
                        success=False,
                        original_performance=baseline_performance,
                        optimized_performance=None,
                        improvement_ratio=None,
                        applied=False,
                        error="LLM returned identical code - no meaningful optimization possible"
                    )
                    results.append(result)
                    continue

                # Save optimization details for review
                saved_file = self.save_optimization_candidate(candidate, llm_response.content, optimized_code)
                self.logger.info(f"Optimization details saved to: {saved_file}")

                if not optimized_code:
                    result = OptimizationResult(
                        candidate=candidate,
                        success=False,
                        original_performance=baseline_performance,
                        optimized_performance=None,
                        improvement_ratio=None,
                        applied=False,
                        error="No optimized code found in LLM response - check saved file"
                    )
                    results.append(result)
                    continue

                # Enhanced optimization application logic - More selective auto-application
                priority_check = candidate.optimization_priority >= 5  # Increased threshold
                issues_text = ' '.join(candidate.performance_issues).lower()
                string_concat_check = "string concatenation" in issues_text and "join()" not in optimized_code.lower()
                list_comp_check = "list comprehension" in optimized_code.lower() and len(optimized_code) < len(candidate.code_snippet) * 1.2
                loop_opt_check = any(term in optimized_code.lower() for term in ["enumerate", "zip", "itertools", "sum(", "any(", "all("])
                memory_check = any(term in issues_text for term in ["memory", "cache", "dict", "set"]) and any(term in optimized_code.lower() for term in ["cache", "dict(", "set("])
                performance_check = any(term in optimized_code.lower() for term in ["# optimization", "# improved", "# faster"])
                size_check = len(optimized_code) > 50 and len(optimized_code) < len(candidate.code_snippet) * 1.5

                # More conservative application logic
                should_apply = (
                    (priority_check and size_check) or  # High priority with reasonable size
                    (string_concat_check and len(optimized_code) > 0) or  # String concat fixes
                    (list_comp_check and loop_opt_check) or  # List comp with loop optimization
                    (memory_check and performance_check)  # Memory optimizations with performance comments
                )

                self.logger.info(f"[APPLY_LOGIC] Should apply decision: {should_apply}")
                self.logger.info(f"[APPLY_LOGIC] - Priority >= 5: {priority_check} (priority: {candidate.optimization_priority})")
                self.logger.info(f"[APPLY_LOGIC] - String concatenation fix: {string_concat_check}")
                self.logger.info(f"[APPLY_LOGIC] - List comprehension + loop opt: {list_comp_check and loop_opt_check}")
                self.logger.info(f"[APPLY_LOGIC] - Memory optimization: {memory_check and performance_check}")
                self.logger.info(f"[APPLY_LOGIC] - Size reasonable: {size_check} (orig: {len(candidate.code_snippet)}, opt: {len(optimized_code)})")

                applied = False
                optimization_error = None

                if should_apply and optimized_code:
                    self.logger.info(f"Applying optimization for {candidate.function_name} (priority: {candidate.optimization_priority})")
                    self.logger.info(f"[APPLY_OPTIMIZATION] Original code snippet: {len(candidate.code_snippet)} chars")
                    self.logger.info(f"[APPLY_OPTIMIZATION] Optimized code: {len(optimized_code)} chars") 

                    # Validate optimized code before applying
                    validation_error = self._validate_optimized_code(optimized_code, candidate)
                    if validation_error:
                        optimization_error = f"Validation failed: {validation_error}"
                        self.logger.warning(f"[APPLY_OPTIMIZATION] {optimization_error}")
                        applied = False
                    else:
                        # Apply the optimization
                        edit_result = self.file_editor.apply_code_block_replacement(
                            file_path, 
                            candidate.line_start, 
                            candidate.line_end, 
                            optimized_code
                        )
                        applied = edit_result.success
                        if applied:
                            self.logger.info(f"[APPLY_OPTIMIZATION] Successfully applied optimization to {candidate.function_name}")

                            # SELF-HEALING: Validate the applied optimization by running debug process
                            validation_result = self._validate_applied_optimization(file_path, candidate, optimized_code)
                            if not validation_result['success']:
                                self.logger.warning(f"[SELF-HEALING] Optimization validation failed: {validation_result['error']}")

                                # Attempt auto-correction
                                correction_result = self._attempt_optimization_correction(
                                    file_path, candidate, optimized_code, validation_result['error']
                                )

                                if correction_result['success']:
                                    self.logger.info(f"[SELF-HEALING] Auto-correction successful for {candidate.function_name}")
                                    applied = True
                                else:
                                    self.logger.error(f"[SELF-HEALING] Auto-correction failed: {correction_result['error']}")
                                    # Rollback to previous state
                                    rollback_success = self._rollback_optimization(file_path, candidate)
                                    if rollback_success:
                                        self.logger.info(f"[SELF-HEALING] Successfully rolled back optimization for {candidate.function_name}")
                                        applied = False
                                        optimization_error = f"Optimization rolled back due to validation failure: {validation_result['error']}"
                                    else:
                                        self.logger.error(
                                            "[SELF-HEALING] Rollback failed - file may be in inconsistent state"
                                        )
                                        applied = False
                                        optimization_error = "Critical error: optimization failed and rollback unsuccessful"
                            else:
                                self.logger.info(f"[SELF-HEALING] Optimization validation passed for {candidate.function_name}")
                        else:
                            optimization_error = f"Failed to apply optimization: {edit_result.error}"
                            self.logger.warning(f"[APPLY_OPTIMIZATION] {optimization_error}")
                else:
                    self.logger.info(f"Optimization saved to file for manual review (priority: {candidate.optimization_priority})")

                # Create result with application status
                result = OptimizationResult(
                    candidate=candidate,
                    success=True,
                    original_performance=baseline_performance,
                    optimized_performance=PerformanceMetrics(0.0),  # Placeholder
                    improvement_ratio=None,  # Unknown without testing
                    applied=applied,
                    error=optimization_error
                )
                results.append(result)

            except Exception as e:
                result = OptimizationResult(
                    candidate=candidate,
                    success=False,
                    original_performance=baseline_performance,
                    optimized_performance=None,
                    improvement_ratio=None,
                    applied=False,
                    error=f"Optimization process failed: {e}"
                )
                results.append(result)
                self.logger.error(f"Error optimizing {candidate.function_name}: {e}")

        # Generate comprehensive optimization report
        applied_count = sum(bool(r.applied)
                        for r in results)
        total_candidates = len(results)
        efficiency_rate = (applied_count / total_candidates * 100) if total_candidates > 0 else 0

        self.logger.info(f"Optimization Summary: {applied_count}/{total_candidates} applied ({efficiency_rate:.1f}% efficiency)")

        self.optimization_history.extend(results)
        self.save_optimization_report(file_path, results)

        return results
    
    def optimize_file_log_driven(self, file_path: str) -> List[OptimizationResult]:
        """Log-driven optimization process for a single file"""
        self.logger.info(f"Log-driven optimization for {file_path}")
        
        results = []
        
        # Get baseline performance
        baseline_performance = self.profiler.profile_file_execution(file_path)
        self.logger.info(f"Baseline execution time: {baseline_performance.execution_time:.4f}s")
        
        # Collect and analyze logs
        log_files = self.log_collector.collect_recent_logs()
        log_content = self.log_collector.extract_log_content(log_files)
        
        if not log_content:
            self.logger.info("No relevant log content found for optimization")
            return results
        
        # Analyze logs for optimization opportunities
        log_analysis = self.log_collector.analyze_logs_for_optimization_opportunities(log_content)
        
        # Generate optimization candidates based on log analysis
        candidates = []
        for issue in log_analysis.get('performance_issues', []):
            candidate = OptimizationCandidate(
                function_name="LogPerformanceIssue",
                file_path=file_path,
                line_start=0,
                line_end=0,
                code_snippet="",
                performance_issues=[issue],
                optimization_priority=5,
                estimated_impact="medium"
            )
            candidates.append(candidate)
        
        # For actual optimization, fallback to static analysis results if no log-driven candidates
        if not candidates:
            self.logger.info("No log-driven optimization candidates found, falling back to static analysis")
            return self._optimize_file_static(file_path)
        
        # Process real optimization candidates with LLM-powered code generation
        for candidate in candidates:
            try:
                self.logger.info(f"[OPTIMIZE] Analyzing optimization candidate: {candidate.function_name}")
                self.logger.info(f"[OPTIMIZE] Performance issues: {len(candidate.performance_issues)}")
                
                # Generate LLM-powered optimization
                optimization_result = self._generate_llm_optimization(candidate, file_path, log_analysis)
                
                if optimization_result['success']:
                    # Apply optimization using Serena semantic editing
                    applied_result = self._apply_optimization_with_serena(
                        file_path, candidate, optimization_result['optimized_code']
                    )
                    
                    if applied_result['success']:
                        # Measure actual performance improvement
                        improved_performance = self._measure_performance_improvement(
                            file_path, baseline_performance
                        )
                        
                        result = OptimizationResult(
                            candidate=candidate,
                            success=True,
                            original_performance=baseline_performance,
                            optimized_performance=improved_performance,
                            improvement_ratio=applied_result.get('improvement_ratio', 0.0),
                            applied=True,
                            error=None
                        )
                        
                        self.logger.info(f"[OPTIMIZE] ✅ Successfully applied optimization to {candidate.function_name}")
                        self.logger.info(f"[OPTIMIZE] 📈 Improvement ratio: {applied_result.get('improvement_ratio', 0.0):.2%}")
                    else:
                        result = OptimizationResult(
                            candidate=candidate,
                            success=False,
                            original_performance=baseline_performance,
                            optimized_performance=None,
                            improvement_ratio=None,
                            applied=False,
                            error=applied_result['error']
                        )
                        self.logger.warning(f"[OPTIMIZE] ❌ Failed to apply optimization: {applied_result['error']}")
                else:
                    result = OptimizationResult(
                        candidate=candidate,
                        success=False,
                        original_performance=baseline_performance,
                        optimized_performance=None,
                        improvement_ratio=None,
                        applied=False,
                        error=optimization_result['error']
                    )
                    self.logger.warning(f"[OPTIMIZE] ❌ Failed to generate optimization: {optimization_result['error']}")
                
                results.append(result)
                
            except Exception as e:
                # Create failed OptimizationResult
                failed_result = OptimizationResult(
                    candidate=candidate,
                    success=False,
                    original_performance=baseline_performance,
                    optimized_performance=None,
                    improvement_ratio=None,
                    applied=False,
                    error=str(e)
                )
                results.append(failed_result)
                self.logger.error(f"[OPTIMIZE] 💥 Error in optimization process: {e}")
        
        return results
    
    def optimize_file(self, file_path: str) -> List[OptimizationResult]:
        """Run the complete optimization process on a file"""
        self.logger.info(f"Starting optimization process for {file_path} (mode: {self.optimization_mode})")
        
        if self.optimization_mode == "log-driven":
            return self.optimize_file_log_driven(file_path)
        else:
            return self._optimize_file_static(file_path)
    
    def optimize_files(self, file_paths: List[str]) -> Dict[str, List[OptimizationResult]]:
        """Run the complete optimization process on multiple files simultaneously
        
        Args:
            file_paths: List of file paths to optimize
            
        Returns:
            Dictionary mapping file paths to their optimization results
        """
        self.logger.info(f"Starting multi-file optimization process for {len(file_paths)} files (mode: {self.optimization_mode})")
        self.logger.info(f"Target files: {file_paths}")

        all_results = {}
        total_candidates = 0
        total_applied = 0

        # Process all files
        for file_path in file_paths:
            self.logger.info("=" * 60)
            self.logger.info(f"OPTIMIZING FILE: {file_path}")
            self.logger.info("=" * 60)

            try:
                # Optimize this specific file
                file_results = self.optimize_file(file_path)
                all_results[file_path] = file_results

                # Update totals
                total_candidates += len(file_results)
                total_applied += sum(bool(r.applied)
                                 for r in file_results)

                # Log file-specific summary
                applied_count = sum(bool(r.applied)
                                for r in file_results)
                self.logger.info(f"File {file_path}: {applied_count}/{len(file_results)} optimizations applied")

            except Exception as e:
                self.logger.error(f"Failed to optimize {file_path}: {e}")
                all_results[file_path] = []

        # Generate comprehensive multi-file optimization report
        efficiency_rate = (total_applied / total_candidates * 100) if total_candidates > 0 else 0

        self.logger.info("=" * 80)
        self.logger.info("MULTI-FILE OPTIMIZATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Files processed: {len(file_paths)}")
        self.logger.info(f"Total candidates: {total_candidates}")
        self.logger.info(f"Total applied: {total_applied}")
        self.logger.info(f"Efficiency rate: {efficiency_rate:.1f}%")

        # Save comprehensive multi-file report
        self.save_multi_file_optimization_report(all_results)

        return all_results
    
    def optimize_files_alternating(self, file_paths: List[str], cycles: int = 2) -> Dict[str, List[OptimizationResult]]:
        """Run alternating optimization: static analysis → log-driven → static analysis → etc.
        
        Args:
            file_paths: List of file paths to optimize
            cycles: Number of complete alternation cycles to run
            
        Returns:
            Dictionary mapping file paths to their optimization results
        """
        self.logger.info(f"Starting alternating optimization for {len(file_paths)} files ({cycles} cycles)")

        all_results = {}

        for cycle in range(cycles):
            self.logger.info("=" * 100)
            self.logger.info(f"ALTERNATION CYCLE {cycle + 1}/{cycles}")
            self.logger.info("=" * 100)

            # Alternate between modes
            if cycle % 2 == 0:
                # Even cycles: static analysis
                self.optimization_mode = "static"
                self.log_collector = None
                mode_name = "STATIC PATTERN ANALYSIS"
            else:
                # Odd cycles: log-driven
                self.optimization_mode = "log-driven"
                self.log_collector = LogCollector()
                mode_name = "LOG-DRIVEN CIRCULAR PROMPTING"

            self.logger.info(f"Mode: {mode_name}")

            # Run optimization with current mode
            cycle_results = self.optimize_files(file_paths)

            # Merge results
            for file_path, results in cycle_results.items():
                if file_path not in all_results:
                    all_results[file_path] = []
                all_results[file_path].extend(results)

        # Final summary
        total_candidates = sum(len(results) for results in all_results.values())
        total_applied = sum(sum(bool(r.applied)
                            for r in results) for results in all_results.values())
        efficiency_rate = (total_applied / total_candidates * 100) if total_candidates > 0 else 0

        self.logger.info("=" * 100)
        self.logger.info("ALTERNATING OPTIMIZATION COMPLETE")
        self.logger.info("=" * 100)
        self.logger.info(f"Total cycles: {cycles}")
        self.logger.info(f"Total candidates: {total_candidates}")
        self.logger.info(f"Total applied: {total_applied}")
        self.logger.info(f"Overall efficiency: {efficiency_rate:.1f}%")

        return all_results
    
    def save_optimization_report(self, file_path: str, results: List[OptimizationResult]) -> str:
        """Save optimization results to a file for review"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_report_{os.path.basename(file_path)}_{timestamp}.json"
        
        try:
            report_data = {
                'file_path': file_path,
                'timestamp': timestamp,
                'total_candidates': len(results),
                'applied_optimizations': sum(bool(r.applied)
                                         for r in results),
                'success_rate': (sum(bool(r.success)
                                 for r in results) / len(results) * 100) if results else 0,
                'results': [asdict(r) for r in results]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Optimization report saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving optimization report: {e}")
            return ""
    
    def save_multi_file_optimization_report(self, all_results: Dict[str, List[OptimizationResult]]) -> str:
        """Save multi-file optimization results to a file for review"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"multi_file_optimization_report_{timestamp}.json"
        
        try:
            total_candidates = sum(len(results) for results in all_results.values())
            total_applied = sum(sum(bool(r.applied)
                                for r in results) for results in all_results.values())
            
            report_data = {
                'timestamp': timestamp,
                'total_files': len(all_results),
                'total_candidates': total_candidates,
                'total_applied': total_applied,
                'efficiency_rate': (total_applied / total_candidates * 100) if total_candidates > 0 else 0,
                'file_results': {}
            }
            
            for file_path, results in all_results.items():
                report_data['file_results'][file_path] = {
                    'candidates': len(results),
                    'applied': sum(bool(r.applied)
                               for r in results),
                    'results': [asdict(r) for r in results]
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Multi-file optimization report saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving multi-file optimization report: {e}")
            return ""
    
    def _is_same_code(self, original_code: str, optimized_code: str) -> bool:
        """Check if optimized code is essentially the same as original (ignoring whitespace)"""
        import re
        
        def normalize_code(code: str) -> str:
            # Remove comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Remove extra whitespace
            code = re.sub(r'\s+', ' ', code)
            # Remove leading/trailing whitespace
            return code.strip()
        
        normalized_original = normalize_code(original_code)
        normalized_optimized = normalize_code(optimized_code)
        
        return normalized_original == normalized_optimized
    
    def _validate_optimized_code(self, optimized_code: str, candidate: OptimizationCandidate) -> Optional[str]:
        """Validate that optimized code is syntactically correct and maintains structure"""
        try:
            # Try to parse the optimized code
            ast.parse(optimized_code)
            
            # Check that it has similar structure (same number of functions/classes)
            original_tree = ast.parse(candidate.code_snippet)
            optimized_tree = ast.parse(optimized_code)
            
            original_functions = [node for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef)]
            optimized_functions = [node for node in ast.walk(optimized_tree) if isinstance(node, ast.FunctionDef)]
            
            if len(original_functions) != len(optimized_functions):
                return f"Function count mismatch: {len(original_functions)} -> {len(optimized_functions)}"
            
            return None  # Valid
            
        except SyntaxError as e:
            return f"Syntax error in optimized code: {e}"
        except Exception as e:
            return f"Validation error: {e}"
    
    def _validate_applied_optimization(self, file_path: str, candidate: OptimizationCandidate, optimized_code: str) -> Dict[str, Any]:
        """Validate that an applied optimization doesn't break the code by running a quick syntax check"""
        try:
            # First, basic syntax check
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Try to parse the entire file
            ast.parse(file_content)
            
            # Run a quick import test if it's a Python module
            if file_path.endswith('.py'):
                import subprocess
                import sys
                
                # Try to compile the file (syntax check)
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', file_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    return {
                        'success': False,
                        'error': f'Syntax error: {result.stderr.strip()}'
                    }
            
            return {'success': True}
            
        except SyntaxError as e:
            return {
                'success': False,
                'error': f'Syntax error: {e}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Validation error: {e}'
            }
    
    def _attempt_optimization_correction(self, file_path: str, candidate: OptimizationCandidate, 
                                       original_optimized_code: str, validation_error: str) -> Dict[str, Any]:
        """Attempt to auto-correct a failed optimization using LLM"""
        try:
            self.logger.info(f"[AUTO-CORRECTION] Attempting to fix optimization for {candidate.function_name}")

            # Create correction prompt
            correction_prompt = f"""
The following optimization was applied but caused errors. Please provide a corrected version:

ORIGINAL CODE:
```python
{candidate.code_snippet}
```

APPLIED OPTIMIZATION (that caused errors):
```python
{original_optimized_code}
```

VALIDATION ERROR:
{validation_error}

Please provide a corrected version of the optimization that:
1. Maintains the same functionality as the original code
2. Fixes the validation error
3. Preserves proper Python syntax and indentation
4. Does not introduce new bugs

Provide only the corrected code, no explanations."""

            # Get correction from LLM
            correction_response = self.llm_interface.generate_optimization(original_optimized_code, correction_prompt)

            if not correction_response.success:
                return {
                    'success': False,
                    'error': f'LLM correction failed: {correction_response.error}'
                }

            # Extract corrected code
            corrected_code = self.llm_interface.extract_code_from_response(correction_response.content)

            if not corrected_code:
                return {
                    'success': False,
                    'error': 'No corrected code found in LLM response'
                }

            if validation_error := self._validate_optimized_code(
                corrected_code, candidate
            ):
                return {
                    'success': False,
                    'error': f'Corrected code still has validation errors: {validation_error}'
                }

            # Apply the corrected optimization
            edit_result = self.file_editor.apply_code_block_replacement(
                file_path,
                candidate.line_start,
                candidate.line_end,
                corrected_code
            )

            if not edit_result.success:
                return {
                    'success': False,
                    'error': f'Failed to apply corrected optimization: {edit_result.error}'
                }

            # Final validation
            final_validation = self._validate_applied_optimization(file_path, candidate, corrected_code)
            if not final_validation['success']:
                return {
                    'success': False,
                    'error': f'Corrected optimization still fails validation: {final_validation["error"]}'
                }

            return {
                'success': True,
                'corrected_code': corrected_code
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Auto-correction failed: {e}'
            }
    
    def _rollback_optimization(self, file_path: str, candidate: OptimizationCandidate) -> bool:
        """Rollback an optimization by restoring the original code"""
        try:
            self.logger.info(f"[ROLLBACK] Rolling back optimization for {candidate.function_name}")
            
            # The file editor should have created a backup - try to restore it
            # For now, we'll implement a simple rollback by re-applying the original code
            edit_result = self.file_editor.apply_code_block_replacement(
                file_path,
                candidate.line_start,
                candidate.line_end,
                candidate.code_snippet
            )
            
            if edit_result.success:
                self.logger.info(f"[ROLLBACK] Successfully rolled back {candidate.function_name}")
                return True
            else:
                self.logger.error(f"[ROLLBACK] Failed to rollback {candidate.function_name}: {edit_result.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"[ROLLBACK] Rollback failed with exception: {e}")
            return False
    
    def optimize_file(self, file_path: str) -> List[OptimizationResult]:
        """Run the complete optimization process on a file"""
        self.logger.info(f"Starting optimization process for {file_path} (mode: {self.optimization_mode})")
        
        if self.optimization_mode == "log-driven":
            return self.optimize_file_log_driven(file_path)
        else:
            return self._optimize_file_static(file_path)
    
    def optimize_files(self, file_paths: List[str]) -> Dict[str, List[OptimizationResult]]:
        """Run the complete optimization process on multiple files simultaneously
        
        Args:
            file_paths: List of file paths to optimize
            
        Returns:
            Dictionary mapping file paths to their optimization results
        """
        self.logger.info(f"Starting multi-file optimization process for {len(file_paths)} files (mode: {self.optimization_mode})")
        self.logger.info(f"Target files: {file_paths}")

        all_results = {}
        total_candidates = 0
        total_applied = 0

        # Process all files
        for file_path in file_paths:
            self.logger.info("=" * 60)
            self.logger.info(f"OPTIMIZING FILE: {file_path}")
            self.logger.info("=" * 60)

            try:
                # Optimize this specific file
                file_results = self.optimize_file(file_path)
                all_results[file_path] = file_results

                # Update totals
                total_candidates += len(file_results)
                total_applied += sum(bool(r.applied)
                                 for r in file_results)

                # Log file-specific summary
                applied_count = sum(bool(r.applied)
                                for r in file_results)
                self.logger.info(f"File {file_path}: {applied_count}/{len(file_results)} optimizations applied")

            except Exception as e:
                self.logger.error(f"Failed to optimize {file_path}: {e}")
                all_results[file_path] = []

        # Generate comprehensive multi-file optimization report
        efficiency_rate = (total_applied / total_candidates * 100) if total_candidates > 0 else 0

        self.logger.info("=" * 80)
        self.logger.info("MULTI-FILE OPTIMIZATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Files processed: {len(file_paths)}")
        self.logger.info(f"Total candidates: {total_candidates}")
        self.logger.info(f"Total applied: {total_applied}")
        self.logger.info(f"Efficiency rate: {efficiency_rate:.1f}%")

        # Save comprehensive multi-file report
        self.save_multi_file_optimization_report(all_results)

        return all_results
    
    def optimize_files_alternating(self, file_paths: List[str], cycles: int = 2) -> Dict[str, List[OptimizationResult]]:
        """Run alternating optimization: static analysis → log-driven → static analysis → etc.
        
        Args:
            file_paths: List of file paths to optimize
            cycles: Number of complete alternation cycles to run
            
        Returns:
            Dictionary mapping file paths to their optimization results
        """
        self.logger.info(f"Starting alternating optimization for {len(file_paths)} files ({cycles} cycles)")

        all_results = {}

        for cycle in range(cycles):
            self.logger.info("=" * 100)
            self.logger.info(f"ALTERNATION CYCLE {cycle + 1}/{cycles}")
            self.logger.info("=" * 100)

            # Alternate between modes
            if cycle % 2 == 0:
                # Even cycles: static analysis
                self.optimization_mode = "static"
                self.log_collector = None
                mode_name = "STATIC PATTERN ANALYSIS"
            else:
                # Odd cycles: log-driven
                self.optimization_mode = "log-driven"
                self.log_collector = LogCollector()
                mode_name = "LOG-DRIVEN CIRCULAR PROMPTING"

            self.logger.info(f"Mode: {mode_name}")

            # Run optimization with current mode
            cycle_results = self.optimize_files(file_paths)

            # Merge results
            for file_path, results in cycle_results.items():
                if file_path not in all_results:
                    all_results[file_path] = []
                all_results[file_path].extend(results)

        # Final summary
        total_candidates = sum(len(results) for results in all_results.values())
        total_applied = sum(sum(bool(r.applied)
                            for r in results) for results in all_results.values())
        efficiency_rate = (total_applied / total_candidates * 100) if total_candidates > 0 else 0

        self.logger.info("=" * 100)
        self.logger.info("ALTERNATING OPTIMIZATION COMPLETE")
        self.logger.info("=" * 100)
        self.logger.info(f"Total cycles: {cycles}")
        self.logger.info(f"Total candidates: {total_candidates}")
        self.logger.info(f"Total applied: {total_applied}")
        self.logger.info(f"Overall efficiency: {efficiency_rate:.1f}%")

        return all_results
