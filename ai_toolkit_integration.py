"""
AI Toolkit for Visual Studio Code Integration

This module provides seamless integration between Microsoft's AI Toolkit for Visual Studio Code
and the GridBot automation pipeline, enabling enhanced model management, prompt engineering,
and fine-tuning capabilities.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

class AIToolkitIntegration:
    """
    Integration bridge between Microsoft AI Toolkit and GridBot AI pipeline
    """
    
    def __init__(self, workspace_path: str, config: Optional[Dict] = None):
        self.workspace_path = workspace_path
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # AI Toolkit paths
        self.ai_models_path = os.path.join(workspace_path, '.ai-models')
        self.prompts_path = os.path.join(workspace_path, '.vscode', 'ai-prompts')
        self.metrics_path = os.path.join(workspace_path, '.vscode', 'ai-metrics')
        self.training_data_path = os.path.join(workspace_path, 'training-data')
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Model registry
        self.model_registry = {
            'debugger': {
                'model': 'deepseek-coder',
                'provider': 'ollama',
                'endpoint': 'http://localhost:11434',
                'purpose': 'Code debugging and error analysis',
                'status': 'unknown',
                'performance_metrics': {}
            },
            'optimizer': {
                'model': 'smollm2:1.7b',
                'provider': 'ollama',
                'endpoint': 'http://localhost:11434',
                'purpose': 'Code optimization and performance enhancement',
                'status': 'unknown',
                'performance_metrics': {}
            },
            'orchestrator': {
                'model': 'qwen3:1.7b',
                'provider': 'ollama',
                'endpoint': 'http://localhost:11434',
                'purpose': 'Workflow orchestration and natural language processing',
                'status': 'unknown',
                'performance_metrics': {}
            }
        }
        
        # Prompt templates
        self.prompt_templates = {}
        self._load_prompt_templates()
        
        # Performance metrics
        self.performance_metrics = {
            'model_usage': {},
            'response_times': {},
            'success_rates': {},
            'error_rates': {}
        }
        
    def _ensure_directories(self):
        """Create necessary directories for AI Toolkit integration"""
        directories = [
            self.ai_models_path,
            self.prompts_path,
            self.metrics_path,
            self.training_data_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def _load_prompt_templates(self):
        """Load existing prompt templates from the prompts directory"""
        if not os.path.exists(self.prompts_path):
            return
            
        for file_name in os.listdir(self.prompts_path):
            if file_name.endswith('.json'):
                template_path = os.path.join(self.prompts_path, file_name)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                        template_name = file_name.replace('.json', '')
                        self.prompt_templates[template_name] = template_data
                except Exception as e:
                    self.logger.warning(f"Failed to load prompt template {file_name}: {e}")
                    
    def check_model_health(self) -> Dict[str, Any]:
        """
        Check the health and availability of all registered AI models
        
        Returns:
            Dict with model health status and performance metrics
        """
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'unknown',
            'models': {},
            'recommendations': []
        }
        
        healthy_models = 0
        total_models = len(self.model_registry)
        
        for model_name, model_info in self.model_registry.items():
            model_health = self._check_individual_model_health(model_name, model_info)
            health_report['models'][model_name] = model_health
            
            if model_health['status'] == 'healthy':
                healthy_models += 1
                
        # Determine overall health
        health_percentage = healthy_models / total_models
        if health_percentage >= 1.0:
            health_report['overall_health'] = 'excellent'
        elif health_percentage >= 0.7:
            health_report['overall_health'] = 'good'
        elif health_percentage >= 0.5:
            health_report['overall_health'] = 'fair'
        else:
            health_report['overall_health'] = 'poor'
            
        # Generate recommendations
        health_report['recommendations'] = self._generate_health_recommendations(health_report['models'])
        
        self.logger.info(f"[AI-TOOLKIT] Model health check completed: {health_report['overall_health']}")
        return health_report
        
    def _check_individual_model_health(self, model_name: str, model_info: Dict) -> Dict:
        """Check health of an individual model"""
        health_status = {
            'model': model_info['model'],
            'provider': model_info['provider'],
            'status': 'unknown',
            'response_time_ms': None,
            'last_check': datetime.now().isoformat(),
            'error_message': None
        }
        
        try:
            if model_info['provider'] == 'ollama':
                # Check Ollama model availability
                response = requests.post(
                    f"{model_info['endpoint']}/api/show",
                    json={'name': model_info['model']},
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Test with a simple prompt
                    start_time = datetime.now()
                    test_response = requests.post(
                        f"{model_info['endpoint']}/api/generate",
                        json={
                            'model': model_info['model'],
                            'prompt': 'Test connection. Respond with "OK".',
                            'stream': False
                        },
                        timeout=30
                    )
                    end_time = datetime.now()
                    
                    response_time = (end_time - start_time).total_seconds() * 1000
                    health_status['response_time_ms'] = response_time
                    
                    if test_response.status_code == 200:
                        health_status['status'] = 'healthy'
                    else:
                        health_status['status'] = 'degraded'
                        health_status['error_message'] = f"Response error: {test_response.status_code}"
                else:
                    health_status['status'] = 'unavailable'
                    health_status['error_message'] = f"Model not found: {response.status_code}"
                    
        except requests.exceptions.RequestException as e:
            health_status['status'] = 'connection_failed'
            health_status['error_message'] = str(e)
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error_message'] = str(e)
            
        return health_status
        
    def _generate_health_recommendations(self, model_status: Dict) -> List[str]:
        """Generate recommendations based on model health status"""
        recommendations = []
        
        for model_name, status in model_status.items():
            if status['status'] == 'unavailable':
                recommendations.append(f"Pull {status['model']} model: ollama pull {status['model']}")
            elif status['status'] == 'connection_failed':
                recommendations.append(f"Check Ollama server connectivity for {model_name}")
            elif status['status'] == 'degraded':
                recommendations.append(f"Consider restarting {status['model']} model for better performance")
            elif status.get('response_time_ms', 0) > 5000:
                recommendations.append(f"High response time for {model_name} - consider optimizing model parameters")
                
        if not recommendations:
            recommendations.append("All models are healthy and performing well")
            
        return recommendations
        
    def create_prompt_template(self, name: str, template: str, category: str = 'gridbot', 
                             metadata: Optional[Dict] = None) -> bool:
        """
        Create a new prompt template for AI Toolkit
        
        Args:
            name: Template name
            template: Prompt template content
            category: Template category
            metadata: Additional metadata
            
        Returns:
            bool indicating success
        """
        try:
            template_data = {
                'name': name,
                'template': template,
                'category': category,
                'created': datetime.now().isoformat(),
                'metadata': metadata or {},
                'usage_count': 0,
                'last_used': None
            }
            
            template_path = os.path.join(self.prompts_path, f"{name}.json")
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2)
                
            self.prompt_templates[name] = template_data
            self.logger.info(f"[AI-TOOLKIT] Created prompt template: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"[AI-TOOLKIT] Failed to create prompt template {name}: {e}")
            return False
            
    def get_prompt_template(self, name: str) -> Optional[str]:
        """
        Get a prompt template by name
        
        Args:
            name: Template name
            
        Returns:
            Template content or None if not found
        """
        if name in self.prompt_templates:
            template_data = self.prompt_templates[name]
            # Update usage statistics
            template_data['usage_count'] = template_data.get('usage_count', 0) + 1
            template_data['last_used'] = datetime.now().isoformat()
            
            # Save updated statistics
            template_path = os.path.join(self.prompts_path, f"{name}.json")
            try:
                with open(template_path, 'w', encoding='utf-8') as f:
                    json.dump(template_data, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to update template usage stats: {e}")
                
            return template_data['template']
        return None
        
    def list_prompt_templates(self, category: Optional[str] = None) -> List[Dict]:
        """
        List available prompt templates
        
        Args:
            category: Filter by category
            
        Returns:
            List of template metadata
        """
        return [
            {
                'name': name,
                'category': data.get('category', 'unknown'),
                'created': data.get('created'),
                'usage_count': data.get('usage_count', 0),
                'last_used': data.get('last_used')
            }
            for name, data in self.prompt_templates.items()
            if category is None or data.get('category') == category
        ]
        
    def record_performance_metrics(self, model_name: str, operation: str, 
                                 response_time: float, success: bool, 
                                 metadata: Optional[Dict] = None):
        """
        Record performance metrics for AI Toolkit analysis
        
        Args:
            model_name: Name of the model used
            operation: Type of operation performed
            response_time: Response time in seconds
            success: Whether the operation was successful
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize model metrics if not exists
        if model_name not in self.performance_metrics['model_usage']:
            self.performance_metrics['model_usage'][model_name] = []
            self.performance_metrics['response_times'][model_name] = []
            self.performance_metrics['success_rates'][model_name] = []
            self.performance_metrics['error_rates'][model_name] = []
            
        # Record metrics
        metric_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'response_time': response_time,
            'success': success,
            'metadata': metadata or {}
        }
        
        self.performance_metrics['model_usage'][model_name].append(metric_entry)
        self.performance_metrics['response_times'][model_name].append(response_time)
        self.performance_metrics['success_rates'][model_name].append(1 if success else 0)
        
        # Save metrics to file
        self._save_performance_metrics()
        
    def _save_performance_metrics(self):
        """Save performance metrics to file"""
        try:
            metrics_file = os.path.join(self.metrics_path, 'performance_metrics.json')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save performance metrics: {e}")
            
    def generate_model_evaluation_report(self) -> Dict:
        """
        Generate comprehensive model evaluation report
        
        Returns:
            Dict with evaluation results and recommendations
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_summary': {},
            'model_performance': {},
            'recommendations': [],
            'optimization_opportunities': []
        }
        
        for model_name in self.performance_metrics['model_usage']:
            usage_data = self.performance_metrics['model_usage'][model_name]
            response_times = self.performance_metrics['response_times'][model_name]
            success_rates = self.performance_metrics['success_rates'][model_name]
            
            if not usage_data:
                continue
                
            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
            total_operations = len(usage_data)
            
            report['model_performance'][model_name] = {
                'total_operations': total_operations,
                'average_response_time': avg_response_time,
                'success_rate': success_rate,
                'performance_grade': self._calculate_performance_grade(avg_response_time, success_rate)
            }
            
            # Generate recommendations
            if avg_response_time > 5.0:
                report['optimization_opportunities'].append(
                    f"Consider optimizing {model_name} - high average response time: {avg_response_time:.2f}s"
                )
            if success_rate < 0.9:
                report['optimization_opportunities'].append(
                    f"Investigate {model_name} reliability - success rate: {success_rate:.2%}"
                )
                
        # Overall recommendations
        if not report['optimization_opportunities']:
            report['recommendations'].append("All models are performing within acceptable parameters")
        else:
            report['recommendations'].extend(report['optimization_opportunities'])
            
        return report
        
    def _calculate_performance_grade(self, response_time: float, success_rate: float) -> str:
        """Calculate performance grade based on metrics"""
        if success_rate >= 0.95 and response_time <= 2.0:
            return 'A'
        elif success_rate >= 0.90 and response_time <= 5.0:
            return 'B'
        elif success_rate >= 0.80 and response_time <= 10.0:
            return 'C'
        elif success_rate >= 0.70:
            return 'D'
        else:
            return 'F'
            
    def create_training_data_sample(self, purpose: str = 'gridbot_fine_tuning') -> str:
        """
        Create a sample training data file for fine-tuning
        
        Args:
            purpose: Purpose of the training data
            
        Returns:
            Path to created training data file
        """
        training_examples = [
            {
                'input': 'Debug this Python syntax error: SyntaxError: invalid syntax',
                'output': 'This is a Python syntax error. Check for missing colons, unmatched parentheses, or incorrect indentation. Common fixes include adding missing colons after if/for/while statements and ensuring proper indentation.',
                'metadata': {'error_type': 'SyntaxError', 'domain': 'debugging'}
            },
            {
                'input': 'Optimize this Python function for better performance',
                'output': 'Consider using list comprehensions instead of loops, utilize built-in functions, avoid repeated calculations, and use appropriate data structures. Profile the code to identify bottlenecks.',
                'metadata': {'task_type': 'optimization', 'domain': 'performance'}
            },
            {
                'input': 'Generate test cases for a trading algorithm function',
                'output': 'Create unit tests covering normal cases, edge cases, error conditions, and boundary values. Include tests for different market conditions and parameter combinations.',
                'metadata': {'task_type': 'testing', 'domain': 'trading'}
            }
        ]
        
        training_data = {
            'purpose': purpose,
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'examples': training_examples,
            'metadata': {
                'domain': 'gridbot_automation',
                'tasks': ['debugging', 'optimization', 'testing'],
                'model_targets': list(self.model_registry.keys())
            }
        }
        
        file_path = os.path.join(self.training_data_path, f'{purpose}_sample.json')
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
            
            self.logger.info(f"[AI-TOOLKIT] Created training data sample: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"[AI-TOOLKIT] Failed to create training data: {e}")
            return ""
            
    def integrate_with_gridbot_pipeline(self, pipeline_instance) -> bool:
        """
        Integrate AI Toolkit capabilities with the GridBot automation pipeline
        
        Args:
            pipeline_instance: Instance of MasterAutomationPipeline
            
        Returns:
            bool indicating successful integration
        """
        try:
            # Add AI Toolkit methods to pipeline
            pipeline_instance.ai_toolkit = self
            pipeline_instance.check_model_health = self.check_model_health
            pipeline_instance.create_prompt_template = self.create_prompt_template
            pipeline_instance.get_prompt_template = self.get_prompt_template
            pipeline_instance.generate_model_evaluation = self.generate_model_evaluation_report
            pipeline_instance.record_ai_metrics = self.record_performance_metrics
            
            # Create default prompt templates if they don't exist
            self._create_default_prompt_templates()
            
            # Initialize performance monitoring
            self._initialize_performance_monitoring(pipeline_instance)
            
            self.logger.info("[AI-TOOLKIT] Successfully integrated with GridBot automation pipeline")
            return True
            
        except Exception as e:
            self.logger.error(f"[AI-TOOLKIT] Failed to integrate with pipeline: {e}")
            return False
            
    def _create_default_prompt_templates(self):
        """Create default prompt templates for common GridBot operations"""
        default_templates = {
            'debug_error': {
                'template': 'Analyze this Python error and provide a detailed explanation with fix suggestions:\n\nError: {error_message}\nCode Context: {code_context}\n\nProvide:\n1. Error explanation\n2. Root cause analysis\n3. Specific fix suggestions\n4. Prevention strategies',
                'category': 'debugging'
            },
            'optimize_function': {
                'template': 'Analyze this Python function for performance optimization opportunities:\n\n{function_code}\n\nProvide:\n1. Performance bottlenecks\n2. Optimization suggestions\n3. Improved code examples\n4. Expected performance gains',
                'category': 'optimization'
            },
            'generate_tests': {
                'template': 'Generate comprehensive test cases for this Python function:\n\n{function_code}\n\nInclude:\n1. Unit tests for normal cases\n2. Edge case tests\n3. Error handling tests\n4. Performance tests if applicable',
                'category': 'testing'
            },
            'code_review': {
                'template': 'Perform a comprehensive code review for this Python code:\n\n{code}\n\nAnalyze:\n1. Code quality and style\n2. Security vulnerabilities\n3. Performance issues\n4. Best practices compliance\n5. Maintainability concerns',
                'category': 'review'
            }
        }
        
        for name, template_info in default_templates.items():
            if name not in self.prompt_templates:
                self.create_prompt_template(
                    name=name,
                    template=template_info['template'],
                    category=template_info['category'],
                    metadata={'default': True, 'gridbot_standard': True}
                )
                
    def _initialize_performance_monitoring(self, pipeline_instance):
        """Initialize performance monitoring for the pipeline"""
        # Store reference for performance tracking
        self.pipeline_instance = pipeline_instance
        
        # Create performance monitoring wrapper
        def monitor_llm_call(original_method):
            def wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = original_method(*args, **kwargs)
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds()
                    
                    # Record successful operation
                    self.record_performance_metrics(
                        model_name='llm_interface',
                        operation=original_method.__name__,
                        response_time=response_time,
                        success=True,
                        metadata={'args_count': len(args), 'kwargs_count': len(kwargs)}
                    )
                    return result
                except Exception as e:
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds()
                    
                    # Record failed operation
                    self.record_performance_metrics(
                        model_name='llm_interface',
                        operation=original_method.__name__,
                        response_time=response_time,
                        success=False,
                        metadata={'error': str(e)}
                    )
                    raise
            return wrapper
        
        # Apply monitoring to key pipeline methods
        if hasattr(pipeline_instance, 'llm_interface') and hasattr(pipeline_instance.llm_interface, '_call_deepseek_debugger'):
            pipeline_instance.llm_interface._call_deepseek_debugger = monitor_llm_call(
                pipeline_instance.llm_interface._call_deepseek_debugger
            )

def create_ai_toolkit_integration(workspace_path: str, config: Optional[Dict] = None) -> AIToolkitIntegration:
    """
    Factory function to create AI Toolkit integration
    
    Args:
        workspace_path: Path to workspace
        config: Configuration dictionary
        
    Returns:
        AIToolkitIntegration instance
    """
    return AIToolkitIntegration(workspace_path, config)

def enhance_pipeline_with_ai_toolkit(pipeline_instance, workspace_path: str) -> bool:
    """
    Helper function to enhance pipeline with AI Toolkit capabilities
    
    Args:
        pipeline_instance: MasterAutomationPipeline instance
        workspace_path: Workspace path
        
    Returns:
        bool indicating success
    """
    ai_toolkit = create_ai_toolkit_integration(workspace_path)
    return ai_toolkit.integrate_with_gridbot_pipeline(pipeline_instance)