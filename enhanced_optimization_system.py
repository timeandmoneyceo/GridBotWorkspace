"""
Enhanced Optimization System with Real-Time Logging and LLM Integration

This module provides comprehensive code optimization capabilities with:
1. Real-time log collection from GridBot execution
2. Intelligent optimization trigger analysis  
3. LLM-powered code generation
4. Serena semantic editing integration
5. Comprehensive real-time logging
6. Performance measurement and rollback mechanisms
"""

import os
import time
import json
import ast
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import existing classes
try:
    # Try relative imports first (when run as module)
    from .optimization_automation_system import (
        OptimizationResult, OptimizationCandidate, PerformanceMetrics,
        OptimizationAutomationSystem
    )
except ImportError:
    # Fall back to absolute imports (when run as script)
    from optimization_automation_system import (
        OptimizationResult, OptimizationCandidate, PerformanceMetrics,
        OptimizationAutomationSystem
    )


class EnhancedOptimizationSystem(OptimizationAutomationSystem):
    """Enhanced optimization system with comprehensive real-time capabilities"""
    
    def __init__(self, llm_interface=None, file_editor=None, **kwargs):
        # Ensure we have valid instances before calling super
        try:
            from .qwen_agent_interface import QwenAgentInterface
        except ImportError:
            from qwen_agent_interface import QwenAgentInterface
        try:
            from .automated_file_editor import SafeFileEditor
        except ImportError:
            from automated_file_editor import SafeFileEditor
        
        if llm_interface is None:
            llm_interface = QwenAgentInterface()
        if file_editor is None:
            file_editor = SafeFileEditor()
            
        super().__init__(llm_interface, file_editor, **kwargs)
        self.optimization_sessions = []
        self.performance_history = {}
        self.rollback_stack = []
        
        # Initialize log collector if not present in parent
        if not hasattr(self, 'log_collector') or self.log_collector is None:
            # Import and create the real LogCollector
            try:
                from .optimization_automation_system import LogCollector
            except ImportError:
                from optimization_automation_system import LogCollector
            self.log_collector = LogCollector()
        
    def optimize_file_enhanced(self, file_path: str) -> List[OptimizationResult]:
        """Enhanced optimization with comprehensive real-time logging and LLM integration"""
        session_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_start = time.time()

        self.logger.info(f"[OPTIMIZE] Starting enhanced optimization session: {session_id}")
        self.logger.info(f"[OPTIMIZE] Target file: {file_path}")
        self.logger.info(f"[OPTIMIZE] Timestamp: {datetime.now().isoformat()}")

        session = {
            'session_id': session_id,
            'file_path': file_path,
            'start_time': datetime.now(),
            'results': [],
            'metrics': {},
            'status': 'running'
        }

        try:
            # Phase 1: Real-time metrics collection
            self.logger.info("[PHASE 1] Collecting real-time execution metrics...")
            runtime_metrics = self._collect_comprehensive_metrics(file_path)
            session['metrics']['runtime'] = runtime_metrics

            # Phase 2: Enhanced log analysis
            self.logger.info("[PHASE 2] Performing enhanced log analysis...")
            log_analysis = self._perform_enhanced_log_analysis(file_path)
            session['metrics']['log_analysis'] = log_analysis

            # Phase 3: Intelligent optimization candidate generation
            self.logger.info("[PHASE 3] Generating intelligent optimization candidates...")
            candidates = self._generate_intelligent_candidates(file_path, log_analysis, runtime_metrics)

            if not candidates:
                self.logger.info("[INFO] [OPTIMIZE] No optimization opportunities identified")
                session['status'] = 'no_candidates'
                return []

            self.logger.info(f"[OPTIMIZE] Generated {len(candidates)} optimization candidates")

            # Phase 4: LLM-powered optimization generation
            results = []
            for idx, candidate in enumerate(candidates, 1):
                self.logger.info(f"[CANDIDATE {idx}/{len(candidates)}] Processing: {candidate.function_name}")

                result = self._process_candidate_enhanced(candidate, file_path, log_analysis, idx, len(candidates))
                results.append(result)
                session['results'].append(result)

                # Real-time progress reporting
                progress = (idx / len(candidates)) * 100
                self.logger.info(f"[PROGRESS] {progress:.1f}% complete ({idx}/{len(candidates)})")

            # Phase 5: Performance validation and reporting
            session_duration = time.time() - session_start
            session['duration'] = session_duration
            session['status'] = 'completed'

            self._generate_comprehensive_report(session, results)

            return results

        except Exception as e:
            session['status'] = 'failed'
            session['error'] = str(e)
            self.logger.error(f"[OPTIMIZE] Session failed: {e}")
            return []

        finally:
            self.optimization_sessions.append(session)
    
    def _collect_comprehensive_metrics(self, file_path: str) -> Dict[str, Any]:
        """Collect comprehensive real-time metrics from GridBot processes"""
        self.logger.info("[METRICS] Scanning system for GridBot processes...")

        metrics = {
            'processes': [],
            'system_stats': {},
            'file_stats': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            import psutil

            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            metrics['system_stats'] = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }

            self.logger.info(f"[METRICS] System CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")

            # Find and analyze GridBot processes
            gridbot_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('GridbotBackup.py' in str(cmd) or 'gridbot_websocket_server.py' in str(cmd) for cmd in cmdline):

                        process_info = {
                            'pid': proc.info['pid'],
                            'name': 'GridbotBackup.py' if 'GridbotBackup.py' in str(cmdline) else 'gridbot_websocket_server.py',
                            'memory_mb': proc.info['memory_info'].rss / (1024 * 1024),
                            'cpu_percent': proc.cpu_percent()
                        }

                        # Get additional metrics
                        try:
                            with proc.oneshot():
                                process_info['create_time'] = proc.create_time()
                                process_info['runtime_hours'] = (time.time() - proc.create_time()) / 3600

                                # I/O metrics
                                io_counters = proc.io_counters()
                                process_info['io_read_mb'] = io_counters.read_bytes / (1024 * 1024)
                                process_info['io_write_mb'] = io_counters.write_bytes / (1024 * 1024)

                                # Network connections for WebSocket server
                                if 'websocket' in process_info['name'].lower():
                                    connections = proc.connections()
                                    process_info['connections_active'] = len([c for c in connections if c.status == 'ESTABLISHED'])
                                    process_info['connections_listening'] = len([c for c in connections if c.status == 'LISTEN'])

                        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                            pass

                        gridbot_processes.append(process_info)

                        self.logger.info(f"[PROCESS] {process_info['name']}: PID {process_info['pid']}, "
                                       f"Memory {process_info['memory_mb']:.1f}MB, "
                                       f"CPU {process_info['cpu_percent']:.1f}%")

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            metrics['processes'] = gridbot_processes
            self.logger.info(f"[METRICS] Found {len(gridbot_processes)} active GridBot processes")

        except ImportError:
            self.logger.warning("[METRICS] psutil not available for enhanced monitoring")
        except Exception as e:
            self.logger.warning(f"[METRICS] Error collecting metrics: {e}")

        # File-specific metrics
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                metrics['file_stats'] = {
                    'size_bytes': stat.st_size,
                    'size_kb': stat.st_size / 1024,
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'age_hours': (time.time() - stat.st_mtime) / 3600
                }

                self.logger.info(f"[FILE] Size: {metrics['file_stats']['size_kb']:.1f}KB, "
                               f"Age: {metrics['file_stats']['age_hours']:.1f}h")
        except Exception as e:
            self.logger.warning(f"[FILE] Could not get file stats: {e}")

        return metrics
    
    def _perform_enhanced_log_analysis(self, file_path: str) -> Dict[str, Any]:
        """Perform enhanced log analysis with intelligent pattern recognition"""
        self.logger.info("[LOG_ANALYSIS] Starting enhanced log analysis...")

        # Use enhanced log collection
        log_files = self.log_collector.collect_recent_logs(max_age_hours=6)  # Recent logs only
        self.logger.info(f"[LOG_ANALYSIS] Collected {len(log_files)} recent log files")

        log_content = self.log_collector.extract_log_content(log_files, max_lines_per_file=2000)

        if not log_content:
            self.logger.warning("[LOG_ANALYSIS] No log content available")
            return {'performance_issues': [], 'optimization_targets': []}

        content_size = len(log_content)
        self.logger.info(f"[LOG_ANALYSIS] Analyzing {content_size} characters of log content")

        # Enhanced analysis with pattern recognition
        analysis = self._analyze_logs_with_intelligence(log_content)

        # Log detailed analysis results
        performance_issues = analysis.get('performance_issues', [])
        optimization_targets = analysis.get('optimization_targets', [])
        error_patterns = analysis.get('error_patterns', [])
        execution_metrics = analysis.get('execution_metrics', {})

        self.logger.info(f"[LOG_ANALYSIS] Performance issues: {len(performance_issues)}")
        self.logger.info(f"[LOG_ANALYSIS] Optimization targets: {len(optimization_targets)}")
        self.logger.info(f"[LOG_ANALYSIS] Error patterns: {len(error_patterns)}")

        if execution_metrics:
            avg_time = execution_metrics.get('avg_execution_time_ms', 0)
            max_time = execution_metrics.get('max_execution_time_ms', 0)
            self.logger.info(f"[LOG_ANALYSIS] Avg execution: {avg_time:.1f}ms, Max: {max_time:.1f}ms")

        if high_priority_targets := [
            t for t in optimization_targets if t.get('priority') == 'high'
        ]:
            self.logger.warning(f"[LOG_ANALYSIS] {len(high_priority_targets)} HIGH PRIORITY optimization targets found!")
            for target in high_priority_targets:
                self.logger.warning(f"[CRITICAL] {target.get('type', 'Unknown')}: {target.get('description', 'No description')}")

        return analysis
    
    def _analyze_logs_with_intelligence(self, log_content: str) -> Dict[str, Any]:
        """Intelligent log analysis with advanced pattern recognition"""
        analysis = {
            'performance_issues': [],
            'optimization_targets': [],
            'error_patterns': [],
            'execution_metrics': {},
            'bottlenecks': [],
            'resource_usage': [],
            'function_patterns': []
        }

        lines = log_content.split('\\n')

        # Advanced pattern recognition
        execution_times = []
        memory_readings = []
        error_counts = {}
        function_calls = {}

        # Performance patterns
        performance_patterns = {
            'slow_execution': re.compile(r'(\\d+\\.?\\d*)\\s*(ms|seconds?|minutes?)'),
            'memory_usage': re.compile(r'(\\d+\\.?\\d*)\\s*mb', re.IGNORECASE),
            'timeout_pattern': re.compile(r'timeout.*?(\\d+)\\s*(s|seconds?|ms)', re.IGNORECASE),
            'connection_issues': re.compile(r'connection.*?(failed|timeout|refused|reset)', re.IGNORECASE),
            'api_errors': re.compile(r'(api|http).*?(error|failed|timeout)', re.IGNORECASE),
            'database_slow': re.compile(r'(database|query|sql).*?(slow|timeout)', re.IGNORECASE)
        }

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            line_lower = line_stripped.lower()

            # Extract execution timing patterns
            time_match = performance_patterns['slow_execution'].search(line_stripped)
            if time_match and any(keyword in line_lower for keyword in ['took', 'elapsed', 'duration', 'time']):
                try:
                    time_val = float(time_match.group(1))
                    time_unit = time_match.group(2).lower()

                    # Convert to milliseconds
                    if 'second' in time_unit:
                        time_val *= 1000
                    elif 'minute' in time_unit:
                        time_val *= 60000

                    execution_times.append(time_val)

                    # Flag slow operations
                    if time_val > 3000:  # > 3 seconds
                        severity = 'critical' if time_val > 10000 else 'high' if time_val > 5000 else 'medium'
                        analysis['bottlenecks'].append({
                            'type': 'slow_execution',
                            'duration_ms': time_val,
                            'line_number': line_num + 1,
                            'log_line': line_stripped,
                            'severity': severity,
                            'optimization_potential': 'high' if time_val > 5000 else 'medium'
                        })
                except ValueError:
                    pass

            if mem_match := performance_patterns['memory_usage'].search(
                line_stripped
            ):
                try:
                    mem_val = float(mem_match.group(1))
                    memory_readings.append(mem_val)

                    if mem_val > 100:  # > 100MB
                        analysis['resource_usage'].append({
                            'type': 'high_memory',
                            'memory_mb': mem_val,
                            'line_number': line_num + 1,
                            'log_line': line_stripped,
                            'severity': 'high' if mem_val > 500 else 'medium'
                        })
                except ValueError:
                    pass

            # Error pattern analysis
            if any(error_term in line_lower for error_term in ['error', 'exception', 'failed', 'timeout', 'refused']):
                error_type = self._classify_error(line_lower)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

                analysis['error_patterns'].append({
                    'type': error_type,
                    'line_number': line_num + 1,
                    'log_line': line_stripped,
                    'frequency': error_counts[error_type],
                    'severity': self._assess_error_severity(error_type, line_lower)
                })

            # Function call pattern analysis
            if any(keyword in line_lower for keyword in ['calling', 'executing', 'running']):
                try:
                    if func_match := re.search(r'(\\w+)\\(', line_stripped):
                        func_name = func_match[1]
                        function_calls[func_name] = function_calls.get(func_name, 0) + 1
                except re.error:
                    pass  # Skip invalid regex matches

        # Calculate execution metrics
        if execution_times:
            analysis['execution_metrics'] = {
                'avg_execution_time_ms': sum(execution_times) / len(execution_times),
                'max_execution_time_ms': max(execution_times),
                'min_execution_time_ms': min(execution_times),
                'total_operations': len(execution_times),
                'slow_operations_count': len([t for t in execution_times if t > 3000]),
                'performance_trend': self._calculate_performance_trend(execution_times)
            }

        if memory_readings:
            analysis['execution_metrics'].update({
                'avg_memory_mb': sum(memory_readings) / len(memory_readings),
                'max_memory_mb': max(memory_readings),
                'memory_trend': self._calculate_memory_trend(memory_readings)
            })

        # Generate intelligent optimization targets
        analysis['optimization_targets'] = self._generate_intelligent_targets(analysis)

        return analysis
    
    def _classify_error(self, line_lower: str) -> str:
        """Classify error types intelligently"""
        if 'timeout' in line_lower:
            return 'timeout'
        elif any(term in line_lower for term in ['connection', 'network', 'socket']):
            return 'connection'
        elif any(term in line_lower for term in ['memory', 'oom', 'allocation']):
            return 'memory'
        elif any(term in line_lower for term in ['api', 'http', 'rest']):
            return 'api'
        elif any(term in line_lower for term in ['database', 'sql', 'query']):
            return 'database'
        elif any(term in line_lower for term in ['syntax', 'parse', 'compile']):
            return 'syntax'
        elif 'permission' in line_lower or 'access' in line_lower:
            return 'permission'
        else:
            return 'general'
    
    def _assess_error_severity(self, error_type: str, line_lower: str) -> str:
        """Assess error severity"""
        critical_keywords = ['critical', 'fatal', 'crash', 'abort', 'emergency']
        high_keywords = ['error', 'exception', 'failed', 'timeout']

        if any(kw in line_lower for kw in critical_keywords):
            return 'critical'
        elif error_type in {'memory', 'connection', 'api'} or any(
            kw in line_lower for kw in high_keywords
        ):
            return 'high'
        else:
            return 'medium'
    
    def _calculate_performance_trend(self, execution_times: List[float]) -> str:
        """Calculate performance trend"""
        if len(execution_times) < 3:
            return 'insufficient_data'
        
        recent_avg = sum(execution_times[-5:]) / min(5, len(execution_times))
        overall_avg = sum(execution_times) / len(execution_times)
        
        if recent_avg > overall_avg * 1.2:
            return 'degrading'
        elif recent_avg < overall_avg * 0.8:
            return 'improving'
        else:
            return 'stable'
    
    def _calculate_memory_trend(self, memory_readings: List[float]) -> str:
        """Calculate memory usage trend"""
        if len(memory_readings) < 3:
            return 'insufficient_data'
        
        recent_avg = sum(memory_readings[-5:]) / min(5, len(memory_readings))
        overall_avg = sum(memory_readings) / len(memory_readings)
        
        if recent_avg > overall_avg * 1.3:
            return 'increasing'
        elif recent_avg < overall_avg * 0.7:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_intelligent_targets(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent optimization targets based on comprehensive analysis"""
        targets = []

        # Performance optimization targets
        execution_metrics = analysis.get('execution_metrics', {})
        if execution_metrics.get('avg_execution_time_ms', 0) > 2000:
            targets.append({
                'type': 'performance_optimization',
                'target': 'reduce_execution_time',
                'priority': 'high',
                'description': f"Average execution time {execution_metrics['avg_execution_time_ms']:.0f}ms exceeds threshold",
                'suggested_techniques': ['async_processing', 'caching', 'algorithm_optimization', 'database_indexing'],
                'estimated_improvement': '20-40%'
            })

        # Memory optimization targets
        if execution_metrics.get('max_memory_mb', 0) > 200:
            targets.append({
                'type': 'memory_optimization',
                'target': 'reduce_memory_usage',
                'priority': 'medium',
                'description': f"Peak memory usage {execution_metrics['max_memory_mb']:.0f}MB is high",
                'suggested_techniques': ['object_pooling', 'lazy_loading', 'memory_profiling', 'garbage_collection'],
                'estimated_improvement': '15-30%'
            })

        # Error handling improvements
        error_patterns = analysis.get('error_patterns', [])
        critical_errors = [e for e in error_patterns if e.get('severity') == 'critical']
        frequent_errors = {}
        for error in error_patterns:
            error_type = error.get('type', 'unknown')
            frequent_errors[error_type] = frequent_errors.get(error_type, 0) + 1

        high_frequency_errors = {k: v for k, v in frequent_errors.items() if v > 3}

        if critical_errors or high_frequency_errors:
            targets.append({
                'type': 'error_handling',
                'target': 'improve_error_resilience',
                'priority': 'high' if critical_errors else 'medium',
                'description': f"Error patterns detected: {len(critical_errors)} critical, {len(high_frequency_errors)} frequent types",
                'suggested_techniques': ['retry_mechanisms', 'circuit_breakers', 'graceful_degradation', 'error_monitoring'],
                'estimated_improvement': '25-50%'
            })

        # Bottleneck optimization targets
        bottlenecks = analysis.get('bottlenecks', [])
        if high_priority_bottlenecks := [
            b for b in bottlenecks if b.get('severity') in ['high', 'critical']
        ]:
            targets.append({
                'type': 'bottleneck_optimization',
                'target': 'eliminate_bottlenecks',
                'priority': 'high',
                'description': f"{len(high_priority_bottlenecks)} performance bottlenecks identified",
                'suggested_techniques': ['profiling', 'code_restructuring', 'parallel_processing', 'optimization'],
                'estimated_improvement': '30-60%'
            })

        return targets
    
    def _generate_intelligent_candidates(self, file_path: str, log_analysis: Dict, runtime_metrics: Dict) -> List[OptimizationCandidate]:
        """Generate intelligent optimization candidates"""
        candidates = []
        
        optimization_targets = log_analysis.get('optimization_targets', [])
        
        for idx, target in enumerate(optimization_targets):
            # Create more specific candidates based on target analysis
            candidate = OptimizationCandidate(
                function_name=f"{target.get('type', 'Optimization')}_{idx+1}",
                file_path=file_path,
                line_start=0,  # Will be determined by code analysis
                line_end=0,
                code_snippet="",  # Will be populated during code analysis
                performance_issues=[target.get('description', 'Performance optimization needed')],
                optimization_priority=self._calculate_priority_from_target(target),
                estimated_impact=target.get('priority', 'medium')
            )
            
            # Add target-specific metadata
            candidate.optimization_metadata = {
                'target_type': target.get('type'),
                'suggested_techniques': target.get('suggested_techniques', []),
                'estimated_improvement': target.get('estimated_improvement', 'Unknown'),
                'analysis_source': 'log_analysis'
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def _process_candidate_enhanced(self, candidate: OptimizationCandidate, file_path: str, 
                                  log_analysis: Dict, idx: int, total: int) -> OptimizationResult:
        """Process optimization candidate with enhanced LLM integration"""
        start_time = time.time()
        
        self.logger.info(f"[CANDIDATE {idx}/{total}] Starting: {candidate.function_name}")
        self.logger.info(f"[CANDIDATE {idx}/{total}] Priority: {candidate.optimization_priority}/10")
        self.logger.info(f"[CANDIDATE {idx}/{total}] Impact: {candidate.estimated_impact}")
        
        try:
            # Generate baseline performance
            baseline_performance = self.profiler.profile_file_execution(file_path)
            
            # Enhanced LLM optimization generation
            self.logger.info(f"[LLM {idx}/{total}] Generating AI-powered optimization...")
            optimization_result = self._generate_enhanced_llm_optimization(
                candidate, file_path, log_analysis
            )
            
            if not optimization_result['success']:
                self.logger.warning(f"[LLM {idx}/{total}] Generation failed: {optimization_result['error']}")
                return OptimizationResult(
                    candidate=candidate,
                    success=False,
                    original_performance=baseline_performance,
                    optimized_performance=None,
                    improvement_ratio=None,
                    applied=False,
                    error=optimization_result['error']
                )
            
            self.logger.info(f"[LLM {idx}/{total}] Generated optimization ({len(optimization_result['optimized_code'])} chars)")
            
            # Apply optimization with Serena integration
            self.logger.info(f"[APPLY {idx}/{total}] Applying optimization...")
            apply_result = self._apply_optimization_enhanced(
                file_path, candidate, optimization_result['optimized_code']
            )
            
            if apply_result['success']:
                # Measure performance improvement
                self.logger.info(f"[MEASURE {idx}/{total}] Measuring performance improvement...")
                improved_performance = self._measure_performance_detailed(file_path, baseline_performance)
                
                improvement_ratio = apply_result.get('improvement_ratio', 0.0)
                
                processing_time = time.time() - start_time
                self.logger.info(f"[SUCCESS {idx}/{total}] Applied in {processing_time:.2f}s, "
                               f"Improvement: {improvement_ratio:.2%}")
                
                return OptimizationResult(
                    candidate=candidate,
                    success=True,
                    original_performance=baseline_performance,
                    optimized_performance=improved_performance,
                    improvement_ratio=improvement_ratio,
                    applied=True,
                    error=None
                )
            else:
                self.logger.warning(f"[APPLY {idx}/{total}] Application failed: {apply_result['error']}")
                return OptimizationResult(
                    candidate=candidate,
                    success=False,
                    original_performance=baseline_performance,
                    optimized_performance=None,
                    improvement_ratio=None,
                    applied=False,
                    error=apply_result['error']
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"[ERROR {idx}/{total}] Failed after {processing_time:.2f}s: {e}")
            return OptimizationResult(
                candidate=candidate,
                success=False,
                original_performance=PerformanceMetrics(0.0),
                optimized_performance=None,
                improvement_ratio=None,
                applied=False,
                error=str(e)
            )
    
    def _generate_enhanced_llm_optimization(self, candidate: OptimizationCandidate, 
                                          file_path: str, log_analysis: Dict) -> Dict[str, Any]:
        """Generate enhanced LLM optimization with comprehensive prompting"""
        try:
            # Read file content for context
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Build enhanced optimization prompt
            prompt = self._build_enhanced_optimization_prompt(
                candidate, file_content, log_analysis
            )
            
            # Call LLM with enhanced parameters using the correct method
            if self.llm_interface:
                self.logger.info(f"[LLM] Sending optimization request ({len(prompt)} chars)")
                
                # Use the public generate_optimization method instead of internal call
                response = self.llm_interface.generate_optimization(
                    candidate.code_snippet,
                    f"Optimization context:\n{prompt}\n\nFile: {file_path}\nFunction: {candidate.function_name}"
                )
                
                if response and response.success and response.content:
                    optimized_code = response.extracted_code or response.content
                    
                    self.logger.info(f"[LLM] Generated optimized code ({len(optimized_code)} chars)")
                    
                    return {
                        'success': True,
                        'optimized_code': optimized_code,
                        'llm_response': response.content,
                        'prompt_used': prompt
                    }
                else:
                    error_msg = response.error if response else 'No response from LLM'
                    return {
                        'success': False,
                        'error': f'LLM optimization failed: {error_msg}'
                    }
            else:
                return {
                    'success': False,
                    'error': 'LLM interface not available'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"LLM generation error: {e}"
            }
    
    def _build_enhanced_optimization_prompt(self, candidate: OptimizationCandidate, 
                                          file_content: str, log_analysis: Dict) -> str:
        """Build comprehensive optimization prompt"""
        
        # Extract metadata
        metadata = getattr(candidate, 'optimization_metadata', {})
        target_type = metadata.get('target_type', 'general')
        techniques = metadata.get('suggested_techniques', [])
        estimated_improvement = metadata.get('estimated_improvement', 'Unknown')

        # Build context sections
        performance_context = ""
        if execution_metrics := log_analysis.get('execution_metrics', {}):
            performance_context = f"""
CURRENT PERFORMANCE METRICS:
- Average execution time: {execution_metrics.get('avg_execution_time_ms', 0):.1f}ms
- Maximum execution time: {execution_metrics.get('max_execution_time_ms', 0):.1f}ms
- Performance trend: {execution_metrics.get('performance_trend', 'unknown')}
- Memory usage: {execution_metrics.get('avg_memory_mb', 0):.1f}MB average
"""

        error_context = ""
        if error_patterns := log_analysis.get('error_patterns', []):
            critical_errors = [e for e in error_patterns if e.get('severity') == 'critical']
            error_context = f"""
ERROR PATTERNS DETECTED:
- Total errors: {len(error_patterns)}
- Critical errors: {len(critical_errors)}
- Most common error types: {', '.join({e.get('type', 'unknown') for e in error_patterns[:5]})}
"""

        optimization_context = ""
        if techniques:
            optimization_context = f"""
RECOMMENDED OPTIMIZATION TECHNIQUES:
{chr(10).join(f'- {technique}' for technique in techniques)}

ESTIMATED IMPROVEMENT POTENTIAL: {estimated_improvement}
"""

        return f"""You are an expert Python performance optimization specialist with deep knowledge of GridBot trading systems, WebSocket servers, and real-time financial applications.

OPTIMIZATION TASK: {target_type.replace('_', ' ').title()}
TARGET FILE: {candidate.file_path}

{performance_context}
{error_context}
{optimization_context}

PERFORMANCE ISSUES TO ADDRESS:
{chr(10).join(f'- {issue}' for issue in candidate.performance_issues)}

CODE TO ANALYZE AND OPTIMIZE:
```python
{file_content[:4000]}  # First 4000 chars for context
```

OPTIMIZATION REQUIREMENTS:
1. **Preserve Functionality**: Maintain exact same behavior and API
2. **Focus on Performance**: Target the specific issues identified above
3. **GridBot Compatibility**: Ensure compatibility with trading systems and WebSocket connections
4. **Error Handling**: Improve error resilience and recovery
5. **Resource Efficiency**: Optimize memory usage and CPU utilization
6. **Real-time Performance**: Maintain low-latency requirements for trading operations

SPECIFIC OPTIMIZATION GUIDELINES:
- For WebSocket servers: Optimize connection handling, message processing, and resource cleanup
- For trading logic: Enhance algorithm efficiency, reduce API call overhead, improve data processing
- For error handling: Add retry mechanisms, circuit breakers, and graceful degradation
- For memory usage: Implement object pooling, lazy loading, and efficient data structures
- For execution time: Use async processing, caching, and optimized algorithms

OUTPUT FORMAT:
Provide a complete, optimized version of the most critical function or code section that addresses the performance issues.
Include detailed comments explaining the optimizations made.

OPTIMIZED CODE:
```python
# Your optimized Python code here
```

OPTIMIZATION EXPLANATION:
Brief explanation of the key optimizations applied and expected performance improvements.
"""
    
    def _parse_enhanced_llm_response(self, response: str) -> str:
        """Parse enhanced LLM response to extract optimized code"""
        # Extract code from markdown blocks
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end != -1:
                return response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end != -1:
                return response[start:end].strip()
        
        # Fallback: try to find Python-like code patterns
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
        
        return '\n'.join(code_lines).strip()
    
    def _apply_optimization_enhanced(self, file_path: str, candidate: OptimizationCandidate, 
                                   optimized_code: str) -> Dict[str, Any]:
        """Apply optimization with enhanced error handling, validation, and correction"""
        try:
            # Create comprehensive backup
            backup_info = self._create_comprehensive_backup(file_path)
            
            # First, try to correct common AI generation errors
            corrected_code = self._correct_ai_generated_code(optimized_code)
            if corrected_code != optimized_code:
                self.logger.info("[CORRECTION] Applied automatic corrections to AI-generated code")
                optimized_code = corrected_code
            
            # Validate optimized code before applying
            validation_result = self._validate_optimized_code(optimized_code, file_path)
            if not validation_result['valid']:
                self.logger.warning(f"[VALIDATION] Code validation failed: {validation_result['error']}")
                
                # Try one more correction attempt
                final_corrected = self._aggressive_code_correction(optimized_code)
                if final_corrected != optimized_code:
                    validation_result = self._validate_optimized_code(final_corrected, file_path)
                    if validation_result['valid']:
                        self.logger.info("[CORRECTION] Aggressive correction succeeded")
                        optimized_code = final_corrected
                    else:
                        return {
                            'success': False,
                            'error': f"Code validation failed even after correction: {validation_result['error']}"
                        }
                else:
                    return {
                        'success': False,
                        'error': f"Code validation failed: {validation_result['error']}"
                    }
            
            # Apply using Serena if available, otherwise use traditional method
            if hasattr(self.file_editor, 'serena_client') and self.file_editor.serena_client:
                apply_result = self._apply_with_serena_enhanced(file_path, candidate, optimized_code)
            else:
                apply_result = self._apply_traditional_enhanced(file_path, candidate, optimized_code)
            
            if apply_result['success']:
                # Post-application validation
                post_validation = self._validate_post_optimization(file_path, backup_info)
                if post_validation['valid']:
                    improvement_ratio = self._calculate_improvement_ratio_enhanced(
                        candidate, validation_result.get('metrics', {})
                    )
                    
                    return {
                        'success': True,
                        'improvement_ratio': improvement_ratio,
                        'backup_info': backup_info,
                        'validation': post_validation
                    }
                else:
                    # Rollback on post-validation failure
                    self._rollback_from_backup(file_path, backup_info)
                    return {
                        'success': False,
                        'error': f"Post-optimization validation failed: {post_validation['error']}",
                        'rolled_back': True
                    }
            else:
                return apply_result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Optimization application error: {e}"
            }
    
    def _generate_comprehensive_report(self, session: Dict, results: List[OptimizationResult]):
        """Generate comprehensive optimization report"""
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"optimization_report_enhanced_{report_time}.json"

        # Calculate comprehensive statistics
        total_candidates = len(results)
        successful = len([r for r in results if r.success and r.applied])
        failed = total_candidates - successful
        total_improvement = sum(r.improvement_ratio or 0 for r in results if r.improvement_ratio)

        report = {
            'session_info': {
                'session_id': session['session_id'],
                'file_path': session['file_path'],
                'start_time': session['start_time'].isoformat(),
                'duration': session.get('duration', 0),
                'status': session['status']
            },
            'optimization_summary': {
                'total_candidates': total_candidates,
                'successful_optimizations': successful,
                'failed_optimizations': failed,
                'success_rate': (successful / total_candidates * 100) if total_candidates > 0 else 0,
                'total_improvement': total_improvement,
                'average_improvement': (total_improvement / successful) if successful > 0 else 0
            },
            'detailed_results': [
                {
                    'candidate_name': r.candidate.function_name,
                    'success': r.success,
                    'applied': r.applied,
                    'improvement_ratio': r.improvement_ratio,
                    'error': r.error,
                    'priority': r.candidate.optimization_priority,
                    'estimated_impact': r.candidate.estimated_impact
                }
                for r in results
            ],
            'metrics': session.get('metrics', {}),
            'recommendations': self._generate_optimization_recommendations(results)
        }

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"[REPORT] Comprehensive report saved: {report_file}")

            # Log summary to console
            self.logger.info("[SUMMARY] === OPTIMIZATION SESSION COMPLETE ===")
            self.logger.info(f"[SUMMARY] Success rate: {report['optimization_summary']['success_rate']:.1f}% ({successful}/{total_candidates})")
            self.logger.info(f"[SUMMARY] Total improvement: {total_improvement:.2%}")
            self.logger.info(f"[SUMMARY] Session duration: {session.get('duration', 0):.2f}s")

            if successful > 0:
                self.logger.info(f"[SUMMARY] Average improvement per optimization: {(total_improvement / successful):.2%}")

        except Exception as e:
            self.logger.error(f"[REPORT] Failed to save report: {e}")
    
    def _generate_optimization_recommendations(self, results: List[OptimizationResult]) -> List[str]:
        """Generate optimization recommendations based on results"""
        recommendations = []

        successful_count = len([r for r in results if r.success and r.applied])
        total_count = len(results)

        if successful_count == 0:
            recommendations.extend(
                (
                    "Consider manual code review as no automatic optimizations were successful",
                    "Review error logs to identify specific optimization barriers",
                )
            )
        elif successful_count < total_count * 0.5:
            recommendations.extend(
                (
                    "Mixed optimization results - review failed optimizations for common patterns",
                    "Consider incremental optimization approach for complex cases",
                )
            )
        else:
            recommendations.append("Good optimization success rate - consider running additional optimization cycles")

        if improvements := [
            r.improvement_ratio
            for r in results
            if r.improvement_ratio and r.improvement_ratio > 0
        ]:
            avg_improvement = sum(improvements) / len(improvements)
            if avg_improvement > 0.2:
                recommendations.append("High improvement potential detected - consider similar optimizations in other files")
            elif avg_improvement < 0.05:
                recommendations.append("Low improvement ratios suggest code is already well-optimized")

        return recommendations
    
    # Helper methods for enhanced functionality
    def _create_comprehensive_backup(self, file_path: str) -> Dict[str, Any]:
        """Create comprehensive backup with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{file_path}.backup.{timestamp}"
        
        import shutil
        shutil.copy2(file_path, backup_path)
        
        return {
            'backup_path': backup_path,
            'original_path': file_path,
            'timestamp': timestamp,
            'size_bytes': os.path.getsize(file_path)
        }
    
    def _validate_optimized_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Enhanced validation of optimized code before application"""
        try:
            # Basic syntax validation
            compile(code, f"{file_path}_optimized", 'exec')
            
            # Enhanced validation for common AI generation errors
            validation_issues = self._check_code_structure_integrity(code)
            if validation_issues:
                self.logger.warning(f"[VALIDATION] Code structure issues detected: {validation_issues}")
                return {
                    'valid': False,
                    'error': f"Code structure validation failed: {', '.join(validation_issues)}"
                }
            
            # Context-aware validation - check if code can be integrated
            context_validation = self._validate_code_in_context(code, file_path)
            if not context_validation['valid']:
                return {
                    'valid': False,
                    'error': f"Context validation failed: {context_validation['error']}"
                }
            
            # Additional validation checks could go here
            # (e.g., AST analysis, import validation, etc.)
            
            return {
                'valid': True,
                'metrics': {
                    'syntax_valid': True,
                    'structure_valid': True,
                    'context_valid': True
                }
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error: {e}"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {e}"
            }
    
    def _check_code_structure_integrity(self, code: str) -> List[str]:
        """Check for common code structure issues that AI models tend to generate"""
        issues = []
        
        # Check for balanced quotes
        quote_issues = self._check_quote_balance(code)
        issues.extend(quote_issues)
        
        # Check for balanced braces/brackets
        brace_issues = self._check_brace_balance(code)
        issues.extend(brace_issues)
        
        # Check for incomplete string literals
        string_issues = self._check_string_literals(code)
        issues.extend(string_issues)
        
        return issues
    
    def _check_quote_balance(self, code: str) -> List[str]:
        """Check for balanced quotes in the code"""
        issues = []
        
        # Check for unterminated triple quotes
        triple_single = code.count("'''")
        triple_double = code.count('"""')
        
        if triple_single % 2 != 0:
            issues.append("unterminated triple single quotes (''')")
        if triple_double % 2 != 0:
            issues.append("unterminated triple double quotes (\"\"\")")
        
        # Check for unterminated single/double quotes (basic check)
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith('#'):
                continue
                
            # Simple check for unterminated quotes at line end
            if (stripped.count('"') % 2 != 0 or stripped.count("'") % 2 != 0) and not (
                stripped.endswith('\\') or stripped.endswith('"""') or stripped.endswith("'''")
            ):
                # Check if it's inside a larger structure
                in_multiline = False
                for prev_line in lines[max(0, i-3):i]:
                    if '"""' in prev_line or "'''" in prev_line:
                        in_multiline = True
                        break
                if not in_multiline:
                    issues.append(f"potential unterminated quote at line {i}")
        
        return issues
    
    def _check_brace_balance(self, code: str) -> List[str]:
        """Check for balanced braces, brackets, and parentheses"""
        issues = []
        
        # Check braces {}
        if code.count('{') != code.count('}'):
            issues.append(f"unbalanced braces: {code.count('{')} opening, {code.count('}')} closing")
        
        # Check brackets []
        if code.count('[') != code.count(']'):
            issues.append(f"unbalanced brackets: {code.count('[')} opening, {code.count(']')} closing")
        
        # Check parentheses ()
        if code.count('(') != code.count(')'):
            issues.append(f"unbalanced parentheses: {code.count('(')} opening, {code.count(')')} closing")
        
        return issues
    
    def _check_string_literals(self, code: str) -> List[str]:
        """Check for issues with string literals"""
        issues = []
        
        # Check for f-string issues
        f_string_pattern = re.compile(r'f["\'](?:[^"\'\\]|\\.)*')
        for match in f_string_pattern.finditer(code):
            f_string = match.group(0)
            # Check for unclosed braces in f-strings
            brace_count = f_string.count('{') - f_string.count('{{')
            if brace_count > 0:
                # Allow for escaped braces, but flag obvious issues
                unescaped_open = f_string.count('{') - f_string.count('{{')
                unescaped_close = f_string.count('}') - f_string.count('}}')
                if unescaped_open != unescaped_close:
                    issues.append(f"unbalanced braces in f-string: {f_string[:50]}...")
        
        return issues
    
    def _validate_code_in_context(self, code: str, file_path: str) -> Dict[str, Any]:
        """Validate that the code will work in the context of the target file"""
        try:
            # Read the target file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Try to find a good insertion point (this is a simplified check)
            # In practice, we'd want to check the specific location where the code will be inserted
            
            # For now, do a basic check: ensure the code doesn't break basic file structure
            lines = file_content.split('\n')
            if lines and lines[-1].strip():  # File doesn't end with newline
                # Check if adding code would create issues
                test_content = file_content + '\n' + code
                try:
                    compile(test_content, file_path, 'exec')
                    return {'valid': True}
                except SyntaxError as e:
                    return {
                        'valid': False,
                        'error': f"Code would create syntax error when added to file: {e}"
                    }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Context validation error: {e}"
            }
    
    def _correct_ai_generated_code(self, code: str) -> str:
        """Apply automatic corrections to common AI-generated code issues"""
        original_code = code
        
        # Fix unterminated triple quotes
        code = self._fix_unterminated_triple_quotes(code)
        
        # Fix unbalanced braces in f-strings
        code = self._fix_fstring_braces(code)
        
        # Fix other common issues
        code = self._fix_common_syntax_issues(code)
        
        if code != original_code:
            self.logger.info("[CORRECTION] Applied automatic corrections to AI-generated code")
        
        return code
    
    def _fix_unterminated_triple_quotes(self, code: str) -> str:
        """Fix unterminated triple quotes"""
        lines = code.split('\n')
        in_triple_single = False
        in_triple_double = False
        
        for i, line in enumerate(lines):
            # Check for triple quote starts/ends
            if '"""' in line and not in_triple_single:
                # Count triple doubles in this line
                triple_double_count = line.count('"""')
                if triple_double_count % 2 == 1:
                    in_triple_double = not in_triple_double
            
            if "'''" in line and not in_triple_double:
                # Count triple singles in this line
                triple_single_count = line.count("'''")
                if triple_single_count % 2 == 1:
                    in_triple_single = not in_triple_single
        
        # If we end in a triple quote block, close it
        if in_triple_double:
            lines.append('"""')
            self.logger.info("[CORRECTION] Added missing closing triple double quotes")
        elif in_triple_single:
            lines.append("'''")
            self.logger.info("[CORRECTION] Added missing closing triple single quotes")
        
        return '\n'.join(lines)
    
    def _fix_fstring_braces(self, code: str) -> str:
        """Fix unbalanced braces in f-strings"""
        # Find f-strings and check brace balance
        fstring_pattern = re.compile(r'f(["\'])(.*?)\1', re.DOTALL)
        
        def fix_fstring(match):
            quote_type = match.group(1)
            content = match.group(2)
            
            # Count braces
            open_braces = content.count('{') - content.count('{{')  # Ignore escaped braces
            close_braces = content.count('}') - content.count('}}')
            
            if open_braces > close_braces:
                # Add missing closing braces
                missing = open_braces - close_braces
                content += '}' * missing
                self.logger.info(f"[CORRECTION] Added {missing} missing closing braces to f-string")
            elif close_braces > open_braces:
                # Too many closing braces - this is harder to fix, just log
                self.logger.warning(f"[CORRECTION] F-string has {close_braces - open_braces} extra closing braces")
            
            return f'f{quote_type}{content}{quote_type}'
        
        return fstring_pattern.sub(fix_fstring, code)
    
    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix other common syntax issues"""
        # Fix trailing commas in function calls that might break things
        # This is a simple heuristic - in practice, this might need more context
        
        # Fix incomplete imports
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Fix incomplete import statements
            if stripped.startswith('import ') and not stripped.endswith('\\'):
                # Check if it looks incomplete
                if ',' in stripped and not (stripped.endswith(',') or '(' in stripped or ')' in stripped):
                    # Might be a multi-line import that's not properly formatted
                    pass  # For now, leave as-is
            
            # Fix incomplete function definitions
            if stripped.startswith('def ') and not stripped.endswith(':'):
                if not line.endswith('\\'):  # Not a line continuation
                    line += ':'
                    self.logger.info("[CORRECTION] Added missing colon to function definition")
            
            # Fix incomplete class definitions
            if stripped.startswith('class ') and not stripped.endswith(':'):
                if not line.endswith('\\'):  # Not a line continuation
                    line += ':'
                    self.logger.info("[CORRECTION] Added missing colon to class definition")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _aggressive_code_correction(self, code: str) -> str:
        """Apply more aggressive corrections when basic validation fails"""
        # This is a last resort - try to make the code syntactically valid
        
        # Try to parse and find the error location
        try:
            compile(code, '<string>', 'exec')
            return code  # If it compiles, return as-is
        except SyntaxError as e:
            self.logger.info(f"[AGGRESSIVE] Syntax error at line {e.lineno}: {e.msg}")
            
            # Try some aggressive fixes
            lines = code.split('\n')
            
            if e.lineno <= len(lines):
                error_line = lines[e.lineno - 1]
                
                # If it's an indentation error, try to fix it
                if 'indentation' in str(e.msg).lower():
                    # Add proper indentation
                    if not error_line.startswith(' ') and not error_line.startswith('\t'):
                        lines[e.lineno - 1] = '    ' + error_line
                        self.logger.info("[AGGRESSIVE] Fixed indentation issue")
                
                # If it's an unterminated string, try to close it
                elif 'unterminated' in str(e.msg).lower():
                    if error_line.count('"') % 2 == 1:
                        lines[e.lineno - 1] += '"'
                        self.logger.info("[AGGRESSIVE] Added missing quote")
                    elif error_line.count("'") % 2 == 1:
                        lines[e.lineno - 1] += "'"
                        self.logger.info("[AGGRESSIVE] Added missing quote")
            
            return '\n'.join(lines)
        
        except Exception:
            # If all else fails, return original code
            return code
    
    def _calculate_priority_from_target(self, target: Dict[str, Any]) -> int:
        """Calculate numeric priority (1-10) from target dictionary"""
        priority_str = target.get('priority', 'medium').lower()
        
        # Map string priorities to numeric values
        priority_map = {
            'critical': 10,
            'high': 8,
            'medium': 5,
            'low': 3,
            'trivial': 1
        }
        
        return priority_map.get(priority_str, 5)  # Default to medium (5)
