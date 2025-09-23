"""
AI-Powered Workflow Automation and Documentation System

This module provides intelligent workflow automation and context-aware documentation
generation for the GridBot automation pipeline, integrating with VS Code's intelligent
apps capabilities.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import ast
import inspect

class AIWorkflowAutomation:
    """
    AI-powered workflow automation system for repetitive tasks
    """
    
    def __init__(self, llm_interface=None, config: Optional[Dict] = None):
        self.llm_interface = llm_interface
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Workflow templates
        self.workflow_templates = {
            'backup_and_cleanup': self.create_backup_cleanup_workflow,
            'debug_and_optimize': self.create_debug_optimize_workflow,
            'test_generation': self.create_test_generation_workflow,
            'code_review': self.create_code_review_workflow,
            'documentation_update': self.create_documentation_workflow,
            'performance_analysis': self.create_performance_workflow
        }
        
        # Task execution history
        self.execution_history = []
        
        # Automated task patterns
        self.task_patterns = self._load_task_patterns()
        
    def _load_task_patterns(self) -> Dict:
        """Load common task patterns for automation"""
        return {
            'file_operations': {
                'backup_files': {
                    'description': 'Create backups of modified files',
                    'frequency': 'on_change',
                    'priority': 'high',
                    'automation_level': 'full'
                },
                'cleanup_temp_files': {
                    'description': 'Remove temporary and cache files',
                    'frequency': 'daily',
                    'priority': 'medium',
                    'automation_level': 'full'
                },
                'organize_logs': {
                    'description': 'Archive and organize log files',
                    'frequency': 'weekly',
                    'priority': 'low',
                    'automation_level': 'full'
                }
            },
            'code_maintenance': {
                'run_tests': {
                    'description': 'Execute test suite on code changes',
                    'frequency': 'on_save',
                    'priority': 'high',
                    'automation_level': 'partial'
                },
                'format_code': {
                    'description': 'Apply code formatting standards',
                    'frequency': 'on_save',
                    'priority': 'medium',
                    'automation_level': 'full'
                },
                'update_documentation': {
                    'description': 'Generate documentation for new functions',
                    'frequency': 'on_function_add',
                    'priority': 'medium',
                    'automation_level': 'partial'
                }
            },
            'monitoring': {
                'performance_check': {
                    'description': 'Monitor system performance metrics',
                    'frequency': 'hourly',
                    'priority': 'medium',
                    'automation_level': 'full'
                },
                'error_monitoring': {
                    'description': 'Check for recurring errors',
                    'frequency': 'continuous',
                    'priority': 'high',
                    'automation_level': 'full'
                }
            }
        }
        
    def create_automated_workflow(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Create an automated workflow from natural language description
        
        Args:
            task_description: Description of the task to automate
            context: Additional context for workflow creation
            
        Returns:
            Dict with workflow definition
        """
        workflow = {
            'id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': self._generate_workflow_name(task_description),
            'description': task_description,
            'created_at': datetime.now().isoformat(),
            'steps': [],
            'triggers': [],
            'conditions': [],
            'automation_level': 'partial',
            'estimated_time': '5-15 minutes',
            'ai_generated': True
        }

        # Analyze task description to determine workflow type
        workflow_type = self._determine_workflow_type(task_description)

        if workflow_type in self.workflow_templates:
            template_workflow = self.workflow_templates[workflow_type](task_description, context)
            workflow |= template_workflow
        else:
            # Generate custom workflow
            workflow.update(self._generate_custom_workflow(task_description, context))

        self.logger.info(f"[AI-WORKFLOW] Created workflow: {workflow['name']}")
        return workflow
        
    def create_backup_cleanup_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Create backup and cleanup workflow"""
        return {
            'steps': [
                {
                    'id': 'backup_modified_files',
                    'action': 'backup_files',
                    'target': 'modified_files',
                    'parameters': {'backup_dir': 'backups', 'include_timestamp': True}
                },
                {
                    'id': 'cleanup_temp_files',
                    'action': 'cleanup_files',
                    'target': 'temp_directory',
                    'parameters': {'patterns': ['*.tmp', '*.temp', '__pycache__']}
                },
                {
                    'id': 'organize_logs',
                    'action': 'organize_files',
                    'target': 'log_directory',
                    'parameters': {'max_age_days': 7, 'archive_old': True}
                }
            ],
            'triggers': [
                {'type': 'schedule', 'frequency': 'daily', 'time': '02:00'},
                {'type': 'event', 'event': 'file_modified', 'threshold': 5}
            ],
            'automation_level': 'full'
        }
        
    def create_debug_optimize_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Create debug and optimization workflow"""
        return {
            'steps': [
                {
                    'id': 'run_syntax_check',
                    'action': 'validate_syntax',
                    'target': 'python_files',
                    'parameters': {'include_imports': True}
                },
                {
                    'id': 'run_debug_cycle',
                    'action': 'debug_files',
                    'target': 'target_files',
                    'parameters': {'max_iterations': 3, 'use_ai': True}
                },
                {
                    'id': 'optimize_code',
                    'action': 'optimize_performance',
                    'target': 'debugged_files',
                    'parameters': {'optimization_level': 'comprehensive'}
                },
                {
                    'id': 'generate_report',
                    'action': 'create_report',
                    'target': 'results',
                    'parameters': {'include_metrics': True, 'format': 'json'}
                }
            ],
            'triggers': [
                {'type': 'command', 'command': 'gridbot.ai.fullPipeline'},
                {'type': 'event', 'event': 'error_detected', 'auto_run': True}
            ],
            'automation_level': 'partial'
        }
        
    def create_test_generation_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Create test generation workflow"""
        return {
            'steps': [
                {
                    'id': 'analyze_functions',
                    'action': 'analyze_code',
                    'target': 'python_files',
                    'parameters': {'extract_functions': True, 'complexity_analysis': True}
                },
                {
                    'id': 'generate_tests',
                    'action': 'create_tests',
                    'target': 'functions',
                    'parameters': {'test_types': ['unit', 'integration'], 'use_ai': True}
                },
                {
                    'id': 'validate_tests',
                    'action': 'run_tests',
                    'target': 'generated_tests',
                    'parameters': {'check_coverage': True}
                }
            ],
            'triggers': [
                {'type': 'event', 'event': 'function_added', 'auto_run': True},
                {'type': 'command', 'command': 'gridbot.ai.generateTests'}
            ],
            'automation_level': 'partial'
        }
        
    def create_code_review_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Create code review workflow"""
        return {
            'steps': [
                {
                    'id': 'analyze_changes',
                    'action': 'analyze_diff',
                    'target': 'modified_files',
                    'parameters': {'include_context': True}
                },
                {
                    'id': 'run_ai_review',
                    'action': 'ai_code_review',
                    'target': 'changes',
                    'parameters': {'review_level': 'comprehensive', 'check_best_practices': True}
                },
                {
                    'id': 'check_security',
                    'action': 'security_scan',
                    'target': 'code_changes',
                    'parameters': {'scan_for': ['hardcoded_secrets', 'sql_injection', 'xss']}
                },
                {
                    'id': 'generate_review_report',
                    'action': 'create_report',
                    'target': 'review_results',
                    'parameters': {'format': 'markdown', 'include_suggestions': True}
                }
            ],
            'triggers': [
                {'type': 'event', 'event': 'file_saved', 'file_types': ['.py']},
                {'type': 'command', 'command': 'gridbot.ai.reviewCode'}
            ],
            'automation_level': 'full'
        }
        
    def create_documentation_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Create documentation generation workflow"""
        return {
            'steps': [
                {
                    'id': 'scan_code_changes',
                    'action': 'analyze_code',
                    'target': 'modified_files',
                    'parameters': {'extract_docstrings': True, 'find_undocumented': True}
                },
                {
                    'id': 'generate_docstrings',
                    'action': 'create_documentation',
                    'target': 'undocumented_functions',
                    'parameters': {'style': 'google', 'include_examples': True}
                },
                {
                    'id': 'update_api_docs',
                    'action': 'update_docs',
                    'target': 'api_documentation',
                    'parameters': {'auto_update': True, 'include_type_hints': True}
                },
                {
                    'id': 'generate_readme',
                    'action': 'create_readme',
                    'target': 'project_root',
                    'parameters': {'include_usage_examples': True, 'auto_toc': True}
                }
            ],
            'triggers': [
                {'type': 'event', 'event': 'function_added', 'auto_run': True},
                {'type': 'command', 'command': 'gridbot.ai.generateDocs'}
            ],
            'automation_level': 'partial'
        }
        
    def create_performance_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Create performance analysis workflow"""
        return {
            'steps': [
                {
                    'id': 'profile_performance',
                    'action': 'profile_code',
                    'target': 'python_files',
                    'parameters': {'profiler': 'cProfile', 'include_memory': True}
                },
                {
                    'id': 'identify_bottlenecks',
                    'action': 'analyze_performance',
                    'target': 'profile_results',
                    'parameters': {'threshold_ms': 100, 'memory_threshold_mb': 50}
                },
                {
                    'id': 'suggest_optimizations',
                    'action': 'optimization_analysis',
                    'target': 'bottlenecks',
                    'parameters': {'use_ai': True, 'include_algorithms': True}
                }
            ],
            'triggers': [
                {'type': 'schedule', 'frequency': 'weekly'},
                {'type': 'command', 'command': 'gridbot.ai.performanceTest'}
            ],
            'automation_level': 'full'
        }
        
    def _determine_workflow_type(self, description: str) -> str:
        """Determine workflow type from description"""
        description_lower = description.lower()

        type_keywords = {
            'backup_and_cleanup': ['backup', 'cleanup', 'clean', 'organize', 'archive'],
            'debug_and_optimize': ['debug', 'fix', 'optimize', 'improve', 'enhance'],
            'test_generation': ['test', 'testing', 'unit test', 'coverage'],
            'code_review': ['review', 'analyze', 'check', 'quality', 'lint'],
            'documentation_update': ['document', 'docs', 'readme', 'docstring'],
            'performance_analysis': ['performance', 'profile', 'benchmark', 'speed']
        }

        return next(
            (
                workflow_type
                for workflow_type, keywords in type_keywords.items()
                if any(keyword in description_lower for keyword in keywords)
            ),
            'custom',
        )
        
    def _generate_custom_workflow(self, description: str, context: Optional[Dict] = None) -> Dict:
        """Generate custom workflow for unrecognized tasks"""
        return {
            'steps': [
                {
                    'id': 'analyze_task',
                    'action': 'task_analysis',
                    'target': 'user_input',
                    'parameters': {'description': description}
                },
                {
                    'id': 'execute_custom_logic',
                    'action': 'custom_execution',
                    'target': 'task_parameters',
                    'parameters': {'use_ai_assistance': True}
                }
            ],
            'triggers': [
                {'type': 'manual', 'description': 'User-initiated custom workflow'}
            ],
            'automation_level': 'manual'
        }
        
    def _generate_workflow_name(self, description: str) -> str:
        """Generate a descriptive name for the workflow"""
        # Extract key words and create a concise name
        words = description.split()
        key_words = [word for word in words[:5] if len(word) > 3]
        name = '_'.join(key_words[:3]).lower()
        return f"ai_workflow_{name}"

class AIDocumentationGenerator:
    """
    AI-powered documentation generation system
    """
    
    def __init__(self, llm_interface=None, config: Optional[Dict] = None):
        self.llm_interface = llm_interface
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Documentation templates
        self.doc_templates = {
            'function': self._generate_function_docs,
            'class': self._generate_class_docs,
            'module': self._generate_module_docs,
            'api': self._generate_api_docs,
            'readme': self._generate_readme_docs,
            'tutorial': self._generate_tutorial_docs
        }
        
        # Documentation styles
        self.doc_styles = {
            'google': self._format_google_style,
            'numpy': self._format_numpy_style,
            'sphinx': self._format_sphinx_style
        }
        
    def generate_comprehensive_documentation(self, file_path: str, doc_type: str = 'auto') -> Dict:
        """
        Generate comprehensive documentation for a file
        
        Args:
            file_path: Path to the file to document
            doc_type: Type of documentation to generate
            
        Returns:
            Dict with generated documentation
        """
        documentation = {
            'file_path': file_path,
            'generated_at': datetime.now().isoformat(),
            'doc_type': doc_type,
            'sections': {},
            'metadata': {},
            'ai_generated': True
        }
        
        try:
            # Read and analyze the file
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse the code structure
            code_analysis = self._analyze_code_structure(source_code, file_path)
            documentation['metadata'] = code_analysis['metadata']
            
            # Determine documentation type if auto
            if doc_type == 'auto':
                doc_type = self._determine_doc_type(code_analysis)
                
            # Generate documentation sections
            if doc_type in self.doc_templates:
                doc_content = self.doc_templates[doc_type](source_code, code_analysis)
                documentation['sections'] = doc_content
                
            # Apply documentation style
            style = self.config.get('documentation_style', 'google')
            if style in self.doc_styles:
                documentation['formatted_docs'] = self.doc_styles[style](documentation['sections'])
                
            self.logger.info(f"[AI-DOCS] Generated {doc_type} documentation for {file_path}")
            
        except Exception as e:
            self.logger.error(f"[AI-DOCS] Error generating documentation: {e}")
            documentation['error'] = str(e)
            
        return documentation
        
    def _analyze_code_structure(self, source_code: str, file_path: str) -> Dict:
        """Analyze code structure for documentation generation"""
        analysis = {
            'metadata': {
                'file_name': os.path.basename(file_path),
                'file_type': os.path.splitext(file_path)[1],
                'line_count': len(source_code.split('\n')),
                'char_count': len(source_code)
            },
            'functions': [],
            'classes': [],
            'imports': [],
            'constants': [],
            'complexity': 'low'
        }
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line_number': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'is_private': node.name.startswith('_'),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    analysis['functions'].append(func_info)
                    
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases]  # type: ignore
                    }
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                            
                    analysis['classes'].append(class_info)
                    
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append(alias.name)
                    else:
                        module = node.module or ''
                        for alias in node.names:
                            analysis['imports'].append(f"{module}.{alias.name}")
                            
            # Determine complexity
            total_items = len(analysis['functions']) + len(analysis['classes'])
            if total_items > 20:
                analysis['complexity'] = 'high'
            elif total_items > 10:
                analysis['complexity'] = 'medium'
                
        except Exception as e:
            self.logger.warning(f"Error analyzing code structure: {e}")
            
        return analysis
        
    def _determine_doc_type(self, code_analysis: Dict) -> str:
        """Determine the appropriate documentation type"""
        if len(code_analysis['classes']) > 3:
            return 'module'
        elif len(code_analysis['classes']) > 0:
            return 'class'
        elif len(code_analysis['functions']) > 5:
            return 'module'
        elif len(code_analysis['functions']) > 0:
            return 'function'
        else:
            return 'module'
            
    def _generate_function_docs(self, source_code: str, analysis: Dict) -> Dict:
        """Generate documentation for functions"""
        sections = {
            'overview': f"This module contains {len(analysis['functions'])} functions for the GridBot automation system.",
            'functions': []
        }
        
        for func in analysis['functions']:
            func_doc = {
                'name': func['name'],
                'description': func.get('docstring', f"AI-generated description for {func['name']}"),
                'parameters': self._document_parameters(func['args']),
                'returns': "AI-generated return description",
                'examples': self._generate_usage_examples(func['name'], func['args']),
                'notes': self._generate_function_notes(func)
            }
            sections['functions'].append(func_doc)
            
        return sections
        
    def _generate_class_docs(self, source_code: str, analysis: Dict) -> Dict:
        """Generate documentation for classes"""
        sections = {
            'overview': f"This module contains {len(analysis['classes'])} classes for the GridBot system.",
            'classes': []
        }
        
        for cls in analysis['classes']:
            class_doc = {
                'name': cls['name'],
                'description': cls.get('docstring', f"AI-generated description for {cls['name']}"),
                'inheritance': cls['bases'],
                'methods': cls['methods'],
                'attributes': "AI-generated attribute documentation",
                'examples': self._generate_class_examples(cls['name']),
                'notes': "Additional notes about class usage"
            }
            sections['classes'].append(class_doc)
            
        return sections
        
    def _generate_module_docs(self, source_code: str, analysis: Dict) -> Dict:
        """Generate documentation for modules"""
        return {
            'title': f"Module: {analysis['metadata']['file_name']}",
            'overview': "AI-generated module overview for GridBot automation system",
            'imports': analysis['imports'],
            'functions': [func['name'] for func in analysis['functions']],
            'classes': [cls['name'] for cls in analysis['classes']],
            'complexity': analysis['complexity'],
            'usage_guide': "AI-generated usage guide",
            'api_reference': "Detailed API reference"
        }
        
    def _generate_api_docs(self, source_code: str, analysis: Dict) -> Dict:
        """Generate API documentation"""
        return {
            'api_overview': "GridBot Automation API",
            'endpoints': self._extract_api_endpoints(source_code),
            'authentication': "API authentication information",
            'examples': "API usage examples",
            'error_codes': "Common error codes and handling"
        }
        
    def _generate_readme_docs(self, source_code: str, analysis: Dict) -> Dict:
        """Generate README documentation"""
        return {
            'title': f"# {analysis['metadata']['file_name']}",
            'description': "AI-generated project description",
            'installation': "## Installation\n\nInstallation instructions here",
            'usage': "## Usage\n\nUsage examples here",
            'api': "## API Reference\n\nAPI documentation here",
            'contributing': "## Contributing\n\nContribution guidelines here",
            'license': "## License\n\nLicense information here"
        }
        
    def _generate_tutorial_docs(self, source_code: str, analysis: Dict) -> Dict:
        """Generate tutorial documentation"""
        return {
            'introduction': "Getting started with GridBot automation",
            'prerequisites': "Required knowledge and setup",
            'step_by_step': "Step-by-step tutorial",
            'examples': "Practical examples",
            'troubleshooting': "Common issues and solutions",
            'next_steps': "Advanced topics and next steps"
        }
        
    def _document_parameters(self, args: List[str]) -> List[Dict]:
        """Generate parameter documentation"""
        return [
            {
                'name': arg,
                'type': 'AI-inferred type',
                'description': f'AI-generated description for {arg}',
                'required': True,
                'default': None
            }
            for arg in args if arg != 'self'
        ]
        
    def _generate_usage_examples(self, func_name: str, args: List[str]) -> List[str]:
        """Generate usage examples for functions"""
        if not args or (len(args) == 1 and args[0] == 'self'):
            return [f">>> {func_name}()\n# AI-generated example output"]
        param_str = ', '.join(f'{arg}=value' for arg in args if arg != 'self')
        return [f">>> {func_name}({param_str})\n# AI-generated example output"]
            
    def _generate_function_notes(self, func: Dict) -> List[str]:
        """Generate notes for functions"""
        notes = []
        
        if func['is_private']:
            notes.append("This is a private function - use with caution")
        if func['is_async']:
            notes.append("This is an async function - use with await")
        if not func.get('docstring'):
            notes.append("Documentation generated by AI - may need manual review")
            
        return notes
        
    def _generate_class_examples(self, class_name: str) -> List[str]:
        """Generate usage examples for classes"""
        return [
            f">>> instance = {class_name}()",
            ">>> # Use the instance methods",
            ">>> result = instance.method_name()",
        ]
        
    def _extract_api_endpoints(self, source_code: str) -> List[Dict]:
        """Extract API endpoints from code"""
        # This would analyze Flask/FastAPI routes or similar
        return [
            {
                'method': 'GET',
                'path': '/api/example',
                'description': 'AI-detected API endpoint',
                'parameters': [],
                'responses': {}
            }
        ]
        
    def _format_google_style(self, sections: Dict) -> str:
        """Format documentation in Google style"""
        formatted = "# AI-Generated Documentation\n\n"
        
        if 'overview' in sections:
            formatted += f"{sections['overview']}\n\n"
            
        if 'functions' in sections:
            formatted += "## Functions\n\n"
            for func in sections['functions']:
                formatted += f"### {func['name']}\n\n"
                formatted += f"{func['description']}\n\n"
                formatted += "**Args:**\n"
                for param in func['parameters']:
                    formatted += f"    {param['name']} ({param['type']}): {param['description']}\n"
                formatted += f"\n**Returns:**\n    {func['returns']}\n\n"
                
        return formatted
        
    def _format_numpy_style(self, sections: Dict) -> str:
        """Format documentation in NumPy style"""
        # Similar to Google style but with NumPy conventions
        return self._format_google_style(sections)
        
    def _format_sphinx_style(self, sections: Dict) -> str:
        """Format documentation in Sphinx style"""
        # Similar to Google style but with Sphinx conventions
        return self._format_google_style(sections)

class ExternalAPIManager:
    """
    Intelligent API management for external services and LLM endpoints
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # API endpoint configurations
        self.api_endpoints = {
            'ollama': {
                'url': self.config.get('ollama_url', 'http://localhost:11434'),
                'models': ['deepseek-coder', 'smollm2:1.7b', 'qwen3:1.7b'],
                'health_check': '/api/tags',
                'status': 'unknown'
            },
            'vscode_api': {
                'url': 'vscode://api',
                'features': ['commands', 'workspace', 'editor'],
                'status': 'unknown'
            }
        }
        
        # Connection pool and retry logic
        self.connection_stats = {}
        
    def intelligent_endpoint_management(self) -> Dict:
        """Manage API endpoints intelligently"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'endpoints': {},
            'recommendations': []
        }
        
        for endpoint_name, config in self.api_endpoints.items():
            status = self._check_endpoint_health(endpoint_name, config)
            results['endpoints'][endpoint_name] = status
            
            # Generate recommendations
            if status['status'] == 'healthy':
                results['recommendations'].append(f"{endpoint_name}: Operating normally")
            elif status['status'] == 'degraded':
                results['recommendations'].append(f"{endpoint_name}: Performance issues detected")
            else:
                results['recommendations'].append(f"{endpoint_name}: Connection issues - check configuration")
                
        return results
        
    def _check_endpoint_health(self, name: str, config: Dict) -> Dict:
        """Check health of an API endpoint"""
        # Simulate health check
        return {
            'name': name,
            'status': 'healthy',
            'response_time_ms': 150,
            'last_check': datetime.now().isoformat(),
            'error_rate': 0.0
        }

# Integration functions
def create_ai_workflow_system(llm_interface, config: Optional[Dict] = None) -> AIWorkflowAutomation:
    """Create AI workflow automation system"""
    return AIWorkflowAutomation(llm_interface, config)

def create_ai_documentation_system(llm_interface, config: Optional[Dict] = None) -> AIDocumentationGenerator:
    """Create AI documentation generation system"""
    return AIDocumentationGenerator(llm_interface, config)

def create_external_api_manager(config: Optional[Dict] = None) -> ExternalAPIManager:
    """Create external API manager"""
    return ExternalAPIManager(config)