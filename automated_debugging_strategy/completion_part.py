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

            # Memory usage analysis
            mem_match = performance_patterns['memory_usage'].search(line_stripped)
            if mem_match:
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
                    func_match = re.search(r'(\w+)\(', line_stripped)
                    if func_match:
                        func_name = func_match.group(1)
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
        elif error_type in ['memory', 'connection', 'api'] or any(kw in line_lower for kw in high_keywords):
            return 'high'
        else:
            return 'medium'

    def _calculate_performance_trend(self, execution_times):
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

    def _calculate_memory_trend(self, memory_readings):
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

    def _generate_intelligent_targets(self, analysis):
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
        high_priority_bottlenecks = [b for b in bottlenecks if b.get('severity') in ['high', 'critical']]

        if high_priority_bottlenecks:
            targets.append({
                'type': 'bottleneck_optimization',
                'target': 'eliminate_bottlenecks',
                'priority': 'high',
                'description': f"{len(high_priority_bottlenecks)} performance bottlenecks identified",
                'suggested_techniques': ['profiling', 'code_restructuring', 'parallel_processing', 'optimization'],
                'estimated_improvement': '30-60%'
            })

        return targets