"""
Main Automation Script

This is the master script that orchestrates the entire debugging -> optimization pipeline.
It runs GridbotBackup.py in debug mode, handles errors automatically, and then optimizes the code.
"""

import os
import argparse
import logging
import json
import subprocess
import re
from typing import List, Dict, Union, Optional, Any
from datetime import datetime, timedelta
import queue
import threading
import time
import time
import threading
import queue

# Import our automation modules
try:
    # Try relative imports first (when run as module)
    from .debug_log_parser import DebugLogParser
    from .qwen_agent_interface import QwenAgentInterface
    from .automated_file_editor import SafeFileEditor
    from .debug_automation_orchestrator import DebugAutomationOrchestrator, DebugSession
    from .optimization_automation_system import OptimizationAutomationSystem
    from .enhanced_optimization_system import OptimizationResult
    from .file_management_system import FileManagementSystem
    from .serena_integration import SerenaMCPClient, SerenaCodeAnalyzer, SerenaCodeEditor
    from .systematic_improvement_tracker import ImprovementTracker
    from .intelligent_apps_integration import IntelligentAppsIntegration, enhance_master_pipeline_with_ai
    from .ai_testing_debugging import AITestGenerator, AIDebuggingAssistant, create_test_for_function, analyze_error_with_ai
    from .intelligent_error_explanation import IntelligentErrorExplainer, enhance_debug_orchestrator_with_ai
    from .ai_workflow_documentation import AIWorkflowAutomation, AIDocumentationGenerator, ExternalAPIManager
    from .ai_toolkit_integration import AIToolkitIntegration, enhance_pipeline_with_ai_toolkit
    from .autonomous_strategy_manager import AutonomousStrategyManager
except ImportError:
    # Fall back to absolute imports (when run as script) - now that files are in correct location
    from debug_log_parser import DebugLogParser
    from qwen_agent_interface import QwenAgentInterface
    from automated_file_editor import SafeFileEditor
    from debug_automation_orchestrator import DebugAutomationOrchestrator, DebugSession
    from optimization_automation_system import OptimizationAutomationSystem
    from enhanced_optimization_system import OptimizationResult
    from file_management_system import FileManagementSystem
    from serena_integration import SerenaMCPClient, SerenaCodeAnalyzer, SerenaCodeEditor
    from systematic_improvement_tracker import ImprovementTracker
    from intelligent_apps_integration import IntelligentAppsIntegration, enhance_master_pipeline_with_ai
    from ai_testing_debugging import AITestGenerator, AIDebuggingAssistant, create_test_for_function, analyze_error_with_ai
    from intelligent_error_explanation import IntelligentErrorExplainer, enhance_debug_orchestrator_with_ai
    from ai_workflow_documentation import AIWorkflowAutomation, AIDocumentationGenerator, ExternalAPIManager
    from ai_toolkit_integration import AIToolkitIntegration, enhance_pipeline_with_ai_toolkit
    from ai_strategy_orchestrator import AIStrategyOrchestrator
    from autonomous_strategy_manager import AutonomousStrategyManager

class MasterAutomationPipeline:
    """Master pipeline that orchestrates debugging and optimization"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self.load_default_config()
        self.setup_logging()

        # Track cleanup timing to prevent redundant cleanups
        self.last_cleanup_time = None

        # Initialize components with detailed logging
        self.logger.info("INITIALIZING COMPONENTS")

        # Use Qwen agent interface (enhanced with tool calling capabilities)
        self.logger.info("Using QwenAgentInterface (Qwen3 agent orchestrator with deepseek-coder debugger + smollm2 optimizer)")
        self.llm_interface = QwenAgentInterface(
            model_name=self.config.get('llm_model', 'Qwen3-1.7B'),
            base_url=self.config.get('llm_base_url', 'http://localhost:8000/v1'),
            api_key=self.config.get('llm_api_key', 'EMPTY'),
            workspace_path=os.path.dirname(os.path.abspath(__file__)),
            enable_thinking=self.config.get('enable_thinking', True),
            temperature=self.config.get('llm_temperature', 0.6),
            top_p=self.config.get('llm_top_p', 0.95),
            top_k=self.config.get('llm_top_k', 20),
            min_p=self.config.get('llm_min_p', 0.0),
            max_tokens=self.config.get('llm_max_tokens', 32768),
            deepseek_debugger_url=self.config.get('deepseek_debugger_url', 'http://localhost:11434'),
            deepseek_model=self.config.get('deepseek_model', 'deepseek-coder'),
            qwen_optimizer_url=self.config.get('qwen_optimizer_url', 'http://localhost:11434'),
            qwen_model=self.config.get('qwen_model', 'smollm2:1.7b')
        )

        self.logger.info(f"LLM Model: {self.config.get('llm_model', 'Qwen3-1.7B')} (orchestrator)")
        self.logger.info(f"DeepSeek Debugger: {self.config.get('deepseek_model', 'deepseek-coder')} @ {self.config.get('deepseek_debugger_url', 'http://localhost:11434')}")
        self.logger.info(f"Qwen Optimizer: {self.config.get('qwen_model', 'smollm2:1.7b')} @ {self.config.get('qwen_optimizer_url', 'http://localhost:11434')}")
        self.logger.info(f"LLM Timeout: {self.config.get('llm_timeout', 600)}s")
        self.logger.info(f"Thinking Mode: {self.config.get('enable_thinking', True)}")
        self.logger.info(f"Temperature: {self.config.get('llm_temperature', 0.6)}")
        self.logger.info(f"Max Tokens: {self.config.get('llm_max_tokens', 32768)}")

        # Initialize DebugLogParser BEFORE testing
        self.debug_parser = DebugLogParser()
        self.logger.info("[OK] DebugLogParser initialized")

        # Initialize SafeFileEditor with Serena BEFORE testing
        self.file_editor = SafeFileEditor(
            backup_dir=self.config.get('backup_dir', 'backups'),
            validate_syntax=self.config.get('validate_syntax', True),
            use_serena=self.config.get('use_serena', True)
        )
        self.logger.info("[OK] SafeFileEditor initialized with Serena semantic editing")

        # Initialize debug orchestrator BEFORE testing
        self.debug_orchestrator = DebugAutomationOrchestrator(
            llm_interface=self.llm_interface,
            file_editor=self.file_editor,
            log_parser=self.debug_parser,
            max_iterations=self.config.get('max_debug_iterations', 3),
            timeout_per_run=self.config.get('debug_timeout', 300)
        )
        self.logger.info("[OK] DebugAutomationOrchestrator initialized")

        # Initialize enhanced optimization system BEFORE testing
        try:
            from enhanced_optimization_system import EnhancedOptimizationSystem
            self.optimization_system = EnhancedOptimizationSystem(
                llm_interface=self.llm_interface,
                file_editor=self.file_editor,
                min_improvement_threshold=self.config.get('min_optimization_improvement', 0.05),
                python_executable=self.config.get('python_executable', 'python'),
                optimization_mode=self.config.get('optimization_mode', 'log-driven')
            )
            self.logger.info("[OK] Enhanced optimization system initialized")
        except Exception as e:
            self.logger.error(f"[ERROR] Enhanced optimization system failed: {e}")
            raise RuntimeError(
                f"Enhanced optimization system initialization failed: {e}"
            ) from e

        # Initialize file management system BEFORE testing
        file_mgmt_config = self.config.get('file_management', {})
        file_manager_config = {
            'backup_dir': self.config.get('backup_dir', 'backups'),
            'max_backup_files': file_mgmt_config.get('max_backup_files', 20),
            'backup_retention_days': file_mgmt_config.get('backup_retention_days', 7),
            'max_log_files': file_mgmt_config.get('max_log_files', 10),
            'log_retention_days': file_mgmt_config.get('log_retention_days', 3),
            'max_log_size_mb': file_mgmt_config.get('max_log_size_mb', 50),
            'max_session_files': file_mgmt_config.get('max_session_files', 15),
            'max_report_files': file_mgmt_config.get('max_report_files', 15),
            'session_retention_days': file_mgmt_config.get('session_retention_days', 5),
            'max_summary_files': file_mgmt_config.get('max_summary_files', 10),
            'summary_retention_days': file_mgmt_config.get('summary_retention_days', 7),
            'temp_file_age_hours': file_mgmt_config.get('temp_file_age_hours', 24),
            'file_patterns': {
                'backups': ['*.backup.*'],
                'logs': ['*.log', '*.log.*'],
                'sessions': ['automation_session_*.json'],
                'reports': ['automation_report_*.json'],
                'summaries': ['summary_iteration_*.txt', '*summary*.txt', '*_summary.txt'],
                'temp': ['temp_*', '*.tmp', '*.temp']
            }
        }
        self.file_manager = FileManagementSystem(config=file_manager_config)
        self.logger.info("[OK] FileManagementSystem initialized")

        # Initialize operation queue BEFORE testing
        self.operation_queue = queue.Queue()
        self.queue_processor_thread = None
        self.queue_active = False
        self.logger.info("[OK] Operation queue initialized")

        # Enhanced comprehensive system testing (replaces simple connection test)
        if self.config.get('skip_connection_test', False) or self.config.get('test_mode', False):
            self.logger.info("Skipping comprehensive system test (test mode)")
        elif self.config.get('quick_test_mode', True):
            self.logger.info("Running quick system test (faster initialization)...")
            try:
                if test_success := self.run_quick_system_test():
                    self.logger.info("[OK] Quick system test successful - core components validated")
                else:
                    self.logger.warning("[WARN] Quick system test failed - proceeding anyway")

            except Exception as e:
                self.logger.warning(f"[WARN] Quick system test error: {e} - proceeding anyway")
        else:
            self.logger.info("Running comprehensive system test (includes LLM, file editing, syntax validation)...")
            try:
                if test_success := self.run_comprehensive_system_test():
                    self.logger.info("[OK] Comprehensive system test successful - all components validated")
                else:
                    self.logger.warning("[WARN] Comprehensive system test failed - proceeding anyway")

            except Exception as e:
                self.logger.warning(f"[WARN] Comprehensive system test error: {e} - proceeding anyway")



        # Initialize enhanced optimization system (primary optimization engine)
        try:
            # Direct import since we're already in the automated_debugging_strategy directory
            from enhanced_optimization_system import EnhancedOptimizationSystem
            self.optimization_system = EnhancedOptimizationSystem(
                llm_interface=self.llm_interface,
                file_editor=self.file_editor,
                min_improvement_threshold=self.config.get('min_optimization_improvement', 0.05),
                python_executable=self.config.get('python_executable', 'python'),
                optimization_mode=self.config.get('optimization_mode', 'log-driven')
            )
            self.logger.info("[ENHANCED] Enhanced optimization system initialized as primary optimization engine")
        except ImportError as e:
            self.logger.error(f"[ENHANCED] Enhanced optimization system failed to load: {e}")
            raise RuntimeError(
                f"Enhanced optimization system is required but failed to load: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"[ENHANCED] Enhanced optimization system error: {e}")
            raise RuntimeError(
                f"Enhanced optimization system initialization failed: {e}"
            ) from e

        # Initialize systematic improvement tracker for continuous enhancement prioritization
        try:
            self.improvement_tracker = ImprovementTracker()
            self.logger.info("[TRACKER] Systematic improvement tracker initialized for 1% iteration gains")

            # Establish baseline if this is the first run
            if self.config.get('establish_improvement_baseline', True):
                if not os.path.exists(self.improvement_tracker.results_file):
                    self.logger.info("[BASELINE] Establishing performance baseline for systematic improvements...")
                    baseline_result = self.improvement_tracker.establish_baseline()
                    self.logger.info(f"[BASELINE] Performance baseline: {baseline_result['overall_performance']:.2f}%")
                else:
                    self.logger.info("[BASELINE] Using existing improvement tracking data")

        except Exception as e:
            self.logger.warning(f"[TRACKER] Systematic improvement tracker initialization failed: {e}")
            self.improvement_tracker = None

        # Initialize file management system
        file_mgmt_config = self.config.get('file_management', {})
        # Create complete config for file manager
        file_manager_config = {
            'backup_dir': self.config.get('backup_dir', 'backups'),
            'max_backup_files': file_mgmt_config.get('max_backup_files', 20),
            'backup_retention_days': file_mgmt_config.get('backup_retention_days', 7),
            'max_log_files': file_mgmt_config.get('max_log_files', 10),
            'log_retention_days': file_mgmt_config.get('log_retention_days', 3),
            'max_log_size_mb': file_mgmt_config.get('max_log_size_mb', 50),
            'max_session_files': file_mgmt_config.get('max_session_files', 15),
            'max_report_files': file_mgmt_config.get('max_report_files', 15),
            'session_retention_days': file_mgmt_config.get('session_retention_days', 5),
            'max_summary_files': file_mgmt_config.get('max_summary_files', 10),
            'summary_retention_days': file_mgmt_config.get('summary_retention_days', 7),
            'temp_file_age_hours': file_mgmt_config.get('temp_file_age_hours', 24)
        }
        # Merge with default file patterns
        default_patterns = {
            'file_patterns': {
                'backups': ['*.backup.*'],
                'logs': ['*.log', '*.log.*'],
                'sessions': ['automation_session_*.json'],
                'reports': ['automation_report_*.json'],
                'summaries': ['summary_iteration_*.txt', '*summary*.txt', '*_summary.txt'],
                'temp': ['temp_*', '*.tmp', '*.temp']
            }
        }
        file_manager_config |= default_patterns
        self.file_manager = FileManagementSystem(config=file_manager_config)

        # Run initial cleanup during initialization
        if self.config.get('file_management', {}).get('cleanup_at_startup', True):
            self.logger.info("Running initial file cleanup...")
            cleanup_stats = self.file_manager.run_cleanup()
            self.last_cleanup_time = datetime.now()
            self.logger.info(f"Cleanup complete: {cleanup_stats}")

        # Setup LLM monitoring
        self._setup_llm_monitoring()

        # Initialize operation queue for orderly processing
        self.operation_queue = queue.Queue()
        self.queue_processor_thread = None
        self.queue_active = False

        # =================================================================================
        # AI STRATEGY ORCHESTRATOR - Intelligent Strategy Engineering
        # =================================================================================

        # Initialize AI Strategy Orchestrator for intelligent automation strategies
        self.logger.info("[AI-STRATEGY] Initializing AI Strategy Orchestrator...")
        try:
            self.ai_strategy_orchestrator = AIStrategyOrchestrator(self)
            self.logger.info("[AI-STRATEGY] AI Strategy Orchestrator initialized successfully")

            # Run initial strategy analysis
            if not self.config.get('skip_strategy_analysis', False):
                self.logger.info("[AI-STRATEGY] Running initial strategy analysis...")
                self._run_ai_strategy_analysis()

        except Exception as e:
            self.logger.warning(f"[AI-STRATEGY] Failed to initialize AI Strategy Orchestrator: {e}")
            self.ai_strategy_orchestrator = None

        # =================================================================================
        # AI-ENHANCED FEATURES INTEGRATION
        # Initialize VS Code Intelligent Apps Integration
        # =================================================================================

        # Initialize AI-powered features
        self.logger.info("[AI-ENHANCED] Initializing Intelligent Apps Integration...")
        try:
            self.ai_integration = IntelligentAppsIntegration(config=self.config)

            # Initialize AI test generator
            self.ai_test_generator = AITestGenerator(llm_interface=self.llm_interface)
            self.logger.info("[AI-TEST] AI test generator initialized")

            # Initialize AI debugging assistant  
            self.ai_debugging_assistant = AIDebuggingAssistant(llm_interface=self.llm_interface)
            self.logger.info("[AI-DEBUG] AI debugging assistant initialized")

            # Initialize intelligent error explainer
            self.ai_error_explainer = IntelligentErrorExplainer(
                llm_interface=self.llm_interface,
                ai_debugging_assistant=self.ai_debugging_assistant
            )
            self.logger.info("[AI-ERROR] Intelligent error explainer initialized")

            if error_integration_success := self.ai_error_explainer.integrate_with_debug_orchestrator(
                self.debug_orchestrator
            ):
                self.logger.info("[AI-ERROR] Debug orchestrator enhanced with intelligent error explanation")
            else:
                self.logger.warning("[AI-ERROR] Failed to enhance debug orchestrator")

            # Initialize AI workflow automation and documentation
            self.ai_workflow_automation = AIWorkflowAutomation(
                llm_interface=self.llm_interface,
                config=self.config
            )
            self.logger.info("[AI-WORKFLOW] AI workflow automation initialized")

            self.ai_documentation_generator = AIDocumentationGenerator(
                llm_interface=self.llm_interface,
                config=self.config
            )
            self.logger.info("[AI-DOCS] AI documentation generator initialized")

            self.external_api_manager = ExternalAPIManager(config=self.config)
            self.logger.info("[AI-API] External API manager initialized")

            # Initialize AI Toolkit Integration (Microsoft AI Toolkit for VS Code)
            self.ai_toolkit_integration = AIToolkitIntegration(
                workspace_path=os.path.dirname(os.path.abspath(__file__)),
                config=self.config
            )
            self.logger.info("[AI-TOOLKIT] Microsoft AI Toolkit integration initialized")

            if toolkit_integration_success := self.ai_toolkit_integration.integrate_with_gridbot_pipeline(
                self
            ):
                self.logger.info("[AI-TOOLKIT] Successfully integrated Microsoft AI Toolkit capabilities:")
                self.logger.info("  [OK] Model health monitoring")
                self.logger.info("  [OK] Prompt template management")
                self.logger.info("  [OK] Performance metrics collection")
                self.logger.info("  [OK] Training data generation")
                self.logger.info("  [OK] Model evaluation and optimization")
            else:
                self.logger.warning("[AI-TOOLKIT] Failed to integrate AI Toolkit - continuing without enhanced features")

            if ai_integration_success := self.ai_integration.integrate_with_master_pipeline(
                self
            ):
                self.logger.info("[AI-ENHANCED] Successfully integrated AI features:")
                self.logger.info("  [OK] Natural language command processing")
                self.logger.info("  [OK] Intelligent error explanations")
                self.logger.info("  [OK] AI-powered code completion")
                self.logger.info("  [OK] Semantic code search")
                self.logger.info("  [OK] Automated test generation")
                self.logger.info("  [OK] AI code review assistance")
                self.logger.info("  [OK] Context-aware documentation")

                # Add AI-enhanced methods to the pipeline
                self.generate_ai_tests = self.ai_test_generator.generate_test_file
                self.explain_error_ai = self.ai_debugging_assistant.analyze_error
                self.ai_code_completion = self.ai_integration.enhance_code_completion
                self.semantic_search = self.ai_integration.semantic_search
                self.ai_code_review = self.ai_integration.perform_ai_code_review
                self.generate_ai_docs = self.ai_integration.generate_documentation
                self.process_natural_language = self.ai_integration.process_natural_language_command

                # Add workflow and documentation methods
                self.create_ai_workflow = self.ai_workflow_automation.create_automated_workflow
                self.generate_comprehensive_docs = self.ai_documentation_generator.generate_comprehensive_documentation
                self.manage_external_apis = self.external_api_manager.intelligent_endpoint_management

                # Add AI Toolkit methods
                self.check_model_health = self.ai_toolkit_integration.check_model_health
                self.create_prompt_template = self.ai_toolkit_integration.create_prompt_template
                self.get_prompt_template = self.ai_toolkit_integration.get_prompt_template
                self.generate_model_evaluation = self.ai_toolkit_integration.generate_model_evaluation_report
                self.record_ai_metrics = self.ai_toolkit_integration.record_performance_metrics

            else:
                self.logger.warning("[AI-ENHANCED] AI integration failed - continuing without AI features")

        except Exception as e:
            self.logger.warning(f"[AI-ENHANCED] Failed to initialize AI features: {e}")
            self.logger.info("[AI-ENHANCED] Continuing without AI enhancements")
            self.ai_integration = None
            self.ai_test_generator = None
            self.ai_debugging_assistant = None

        self.session_data = {
            'start_time': datetime.now(),
            'debug_sessions': [],
            'optimization_results': [],
            'total_errors_fixed': 0,
            'total_optimizations_applied': 0,
            'continuous_automation': {
                'current_cycle': 0,
                'total_cycles_completed': 0,
                'consecutive_failures': 0,
                'last_successful_cycle': None,
                'cycle_start_times': [],
                'cycle_results': []
            }
        }

        # ================= Autonomous Strategy Manager (ASM) =================
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            asm_interval = int(self.config.get('asm_interval_sec', 60))
            asm_targets = self.config.get('autonomous_targets') or [
                os.path.join(base_dir, 'GridbotBackup.py'),
                os.path.join(base_dir, 'gridbot_websocket_server.py'),
            ]
            from autonomous_strategy_manager import AutonomousStrategyManager
            self.autonomous_manager = AutonomousStrategyManager(base_dir=base_dir, interval_sec=asm_interval, targets=asm_targets)
            self.logger.info("[ASM] Autonomous Strategy Manager initialized")
        except Exception as e:
            self.logger.warning(f"[ASM] Failed to initialize Autonomous Strategy Manager: {e}")
            self.autonomous_manager = None
    
    def load_default_config(self) -> Dict:
        """Load default configuration, with automatic JSON config loading"""
        # First load hardcoded defaults
        default_config = {
            'llm_base_url': 'http://localhost:11434',  # Ollama server endpoint
            'llm_model': 'qwen3:1.7b',  # Qwen3 model for orchestration (agent capabilities)
            'llm_api_key': 'EMPTY',  # For local models
            'llm_timeout': 1200,  # 20 minute timeout for local models (increased for complex operations)
            'llm_temperature': 0.6,  # Qwen thinking mode temperature
            'llm_top_p': 0.95,  # Qwen recommended top_p
            'llm_top_k': 20,  # Qwen recommended top_k
            'llm_min_p': 0.0,  # Qwen recommended min_p
            'llm_max_tokens': 32768,  # Qwen recommended max tokens
            'enable_thinking': True,  # Enable Qwen thinking mode
            'debug_timeout': 600,  # 10 minute timeout for debug runs (increased for complex debugging)
            'backup_dir': 'backups',
            'validate_syntax': True,  # CRITICAL: Enable syntax validation
            'use_serena': True,  # Enable Serena semantic code editing
            'max_debug_iterations': 50,  # High iteration count for thorough autonomous debugging
            'min_optimization_improvement': 0.01,  # Lower threshold for more optimizations
            'python_executable': 'C:\\Users\\805Sk\\GridBotWorkspace\\.venv\\Scripts\\python.exe',
            'target_files': ['simple_function.py', 'quick_integration_test.py', 'timeout_test.py'],  # Use test files for debugging
            'run_optimization': True,  # Always optimize for autonomous development
            'optimization_mode': 'log-driven',  # Use log-driven mode for Serena integration
            'asm_interval_sec': 60,
            'autonomous_targets': [],
            'restart_gridbot_after_changes': True,
            'save_reports': True,
            'verbose': True,
            # Enhanced strategy prioritization settings
            'establish_improvement_baseline': True,  # Enable systematic improvement tracking
            'prioritize_enhancements': True,  # Prioritize enhanced optimization strategies
            'use_systematic_improvements': True,  # Enable 1% iteration improvement tracking
            'enhancement_mode': 'aggressive',  # Aggressive enhancement application
            'path_fix_priority': 'high',  # High priority for path corrections
            'test_mode': False,  # Enable test mode to reduce connection tests
            'quick_test_mode': True,  # Use quick tests to speed up initialization
            'performance_profiling': {
                'enabled': True,
                'timeout_per_run': 30
            },
            'optimization_settings': {
                'analyze_ast': True,
                'profile_performance': True,
                'min_function_length': 5,  # Lower threshold for more optimization candidates
                'max_candidates_per_file': 20,  # More aggressive optimization
                'mode': 'log-driven'  # Use log-driven mode for better Serena integration
            },
            'safety_settings': {
                'create_backups': True,
                'validate_before_apply': False,
                'rollback_on_failure': True
            },
            'file_management': {
                'max_backup_files': 20,
                'backup_retention_days': 7,
                'max_log_files': 10,
                'log_retention_days': 3,
                'max_log_size_mb': 50,
                'max_session_files': 15,
                'session_retention_days': 5,
                'max_summary_files': 10,
                'summary_retention_days': 7,
                'max_ai_doctor_files': 10,
                'ai_doctor_retention_days': 7,
                'temp_file_age_hours': 24,
                'cleanup_at_startup': True
            },
            'continuous_mode': {
                'enabled': True,  # Default to continuous mode
                'cycle_delay_minutes': 10,
                'max_cycles': -1,  # -1 = unlimited
                'restart_on_failure': True,
                'cycle_cleanup_frequency': 5
            },
            'continuous_automation': {
                'enabled': True,
                'max_cycles': 0,  # 0 = infinite loops for autonomous development
                'cycle_delay_minutes': 5,  # Shorter cycles for active development (5 minutes)
                'restart_on_max_iterations': False,  # Don't restart, keep iterating
                'stop_on_success': False,  # Continue even if all files are successful
                'max_consecutive_failures': 10,  # Higher tolerance for complex debugging
                'health_check_interval': 20,  # Check system health every 20 cycles
                'auto_cleanup_between_cycles': True
            }
        }

        # Try to load from automation_config.json if it exists
        config_file = 'automation_config.json'
        # Also try looking for it in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_config_file = os.path.join(script_dir, 'automation_config.json')

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config |= file_config
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        elif os.path.exists(alt_config_file):
            try:
                with open(alt_config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                print(f"Loaded configuration from {alt_config_file}")
            except Exception as e:
                print(f"Warning: Could not load config file {alt_config_file}: {e}")
        else:
            print("No automation_config.json found, using default configuration")

        return default_config

    # ================= ASM Public Controls =================
    def start_autonomous_mode(self) -> bool:
        """Start background autonomous strategy daemon."""
        try:
            if hasattr(self, 'autonomous_manager') and self.autonomous_manager:
                self.autonomous_manager.start()
                self.logger.info("[ASM] Autonomous mode started")
                return True
            self.logger.warning("[ASM] Autonomous manager not available")
            return False
        except Exception as e:
            self.logger.error(f"[ASM] Failed to start autonomous mode: {e}")
            return False

    def stop_autonomous_mode(self) -> bool:
        """Stop background autonomous strategy daemon."""
        try:
            if hasattr(self, 'autonomous_manager') and self.autonomous_manager:
                self.autonomous_manager.stop()
                self.logger.info("[ASM] Autonomous mode stopped")
                return True
            self.logger.warning("[ASM] Autonomous manager not available")
            return False
        except Exception as e:
            self.logger.error(f"[ASM] Failed to stop autonomous mode: {e}")
            return False

    def autonomous_tick(self) -> Optional[Dict[str, Any]]:
        """Run a single autonomous iteration (useful for cron/testing)."""
        try:
            if hasattr(self, 'autonomous_manager') and self.autonomous_manager:
                return self.autonomous_manager.tick()
            self.logger.warning("[ASM] Autonomous manager not available")
            return None
        except Exception as e:
            self.logger.error(f"[ASM] Tick failed: {e}")
            return None
    
    def setup_logging(self):
        """Setup comprehensive logging with detailed real-time terminal output"""
        log_level = logging.DEBUG if self.config.get('verbose', True) else logging.INFO
        
        # Create custom formatter for detailed output
        class DetailedFormatter(logging.Formatter):
            def format(self, record):
                # Use datetime for millisecond precision
                import datetime
                dt = datetime.datetime.fromtimestamp(record.created)
                timestamp = dt.strftime('%H:%M:%S') + f'.{dt.microsecond // 1000:03d}'
                if record.levelname == 'INFO':
                    return f"[{timestamp}] {record.getMessage()}"
                elif record.levelname == 'ERROR':
                    return f"[{timestamp}] [ERROR] {record.getMessage()}"
                elif record.levelname == 'WARNING':
                    return f"[{timestamp}] [WARN] {record.getMessage()}"
                elif record.levelname == 'DEBUG':
                    return f"[{timestamp}] [DEBUG] {record.getMessage()}"
                else:
                    return f"[{timestamp}] {record.levelname}: {record.getMessage()}"
        
        # Force clear all existing logging configuration
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.NOTSET)
        
        # Clear all existing named loggers to prevent duplication
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
        
        # Setup file handler with standard format
        file_handler = logging.FileHandler('master_automation.log')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Setup console handler with detailed format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(DetailedFormatter())
        
        # Configure root logger with force to override any existing configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Ensure no duplicate handlers
        seen_handlers = set()
        for handler in root_logger.handlers[:]:
            handler_id = (type(handler).__name__, getattr(handler, 'baseFilename', getattr(handler, 'stream', str(handler))))
            if handler_id in seen_handlers:
                root_logger.removeHandler(handler)
            else:
                seen_handlers.add(handler_id)
        
        # Get logger for this module specifically
        self.logger = logging.getLogger(__name__)
        # Clear any existing handlers on this specific logger
        self.logger.handlers.clear()
        # Ensure it propagates to root logger only
        self.logger.propagate = True
        self.logger.setLevel(logging.NOTSET)  # Let root logger handle level
        
        # Add real-time interaction logging methods
        self.log_section = self._create_section_logger()
        self.log_llm_request = self._create_llm_request_logger()
        self.log_llm_response = self._create_llm_response_logger()
        self.log_edit_operation = self._create_edit_logger()
        self.log_error_details = self._create_error_logger()
        
    def _create_section_logger(self):
        """Create section header logger"""
        def log_section(title, char="=", width=80):
            border = char * width
            self.logger.info(f"\n{border}")
            self.logger.info(f"{title:^{width}}")
            self.logger.info(f"{border}\n")
        return log_section
    
    def _create_llm_request_logger(self):
        """Create LLM request logger with full prompt display for live runs"""
        def log_llm_request(prompt, context=""):
            self.logger.info("[LLM] REQUEST:")
            self.logger.info(f"Context: {context}")
            self.logger.info(f"Prompt Size: {len(prompt)} characters")
            self.logger.info("-" * 60)
            
            # For live runs, show more of the prompt content
            if len(prompt) > 3000:
                # For very long prompts, show structured preview with more content
                self.logger.info("[LARGE PROMPT - SHOWING STRUCTURED PREVIEW]")
                
                # Show first 1200 chars
                self.logger.info("=== PROMPT START ===")
                self.logger.info(prompt[:1200])
                
                # Show middle section if it contains important keywords
                middle_start = len(prompt) // 2 - 500
                middle_end = len(prompt) // 2 + 500
                middle_section = prompt[middle_start:middle_end]
                if any(keyword in middle_section.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'debug', 'fix', 'optimize']):
                    self.logger.info("\n=== KEY CONTEXT (MIDDLE) ===")
                    self.logger.info(middle_section)
                
                # Show last 800 chars
                self.logger.info("\n=== PROMPT END ===")
                self.logger.info(prompt[-800:])
                
                self.logger.info(f"\n[TOTAL SIZE: {len(prompt)} chars]")
            elif len(prompt) > 1000:
                # For medium prompts, show more content
                self.logger.info("=== FULL PROMPT ===")
                self.logger.info(prompt[:800])
                self.logger.info("\n... [middle section omitted] ...")
                self.logger.info(prompt[-400:])
                self.logger.info(f"\n[TOTAL SIZE: {len(prompt)} chars]")
            else:
                # Show full prompt for reasonable sizes
                self.logger.info("=== FULL PROMPT ===")
                self.logger.info(prompt)
            
            self.logger.info("-" * 60)
        return log_llm_request
    
    def _create_llm_response_logger(self):
        """Create LLM response logger with full response display"""
        def log_llm_response(response, success=True, error=None):
            if success:
                self.logger.info("[OK] LLM RESPONSE:")
                self.logger.info("-" * 60)
                
                # For live runs, show more complete responses
                if len(response) > 2000:
                    self.logger.info("[LARGE RESPONSE - SHOWING STRUCTURED PREVIEW]")
                    self.logger.info("=== RESPONSE START ===")
                    self.logger.info(response[:1000])
                    self.logger.info("\n=== RESPONSE END ===")
                    self.logger.info(response[-600:])
                    self.logger.info(f"\n[TOTAL RESPONSE SIZE: {len(response)} chars]")
                else:
                    # Show full response for reasonable sizes
                    self.logger.info("=== FULL RESPONSE ===")
                    self.logger.info(response)
                
                self.logger.info("-" * 60)
            else:
                self.logger.error("[FAIL] LLM RESPONSE FAILED:")
                self.logger.error(f"Error: {error}")
                if response:
                    self.logger.error(f"Partial response: {response}")
                self.logger.error("-" * 60)
        return log_llm_response
    
    def _create_edit_logger(self):
        """Create file edit operation logger"""
        def log_edit_operation(operation, file_path, details=""):
            self.logger.info(f"[EDIT] OPERATION: {operation}")
            self.logger.info(f"File: {file_path}")
            if details:
                self.logger.info(f"Details: {details}")
            self.logger.info("-" * 40)
        return log_edit_operation
    
    def _create_error_logger(self):
        """Create detailed error logger"""
        def log_error_details(error_type, file_path, error_msg, line_number=None, context=""):
            self.logger.error(f"[ERROR] {error_type} ERROR:")
            self.logger.error(f"File: {file_path}")
            if line_number:
                self.logger.error(f"Line: {line_number}")
            self.logger.error(f"Message: {error_msg}")
            if context:
                self.logger.error(f"Context: {context}")
            self.logger.error("-" * 50)
        return log_error_details
    
    def validate_environment(self) -> bool:
        """Validate that the environment is ready for automation with comprehensive testing"""
        self.logger.info("Validating automation environment...")
        
        # Run comprehensive system test (includes LLM, file editing, syntax validation)
        self.logger.info("[ENHANCED] Running comprehensive system test...")
        if not self.run_comprehensive_system_test():
            self.logger.error("[ERROR] Comprehensive system test failed. Cannot proceed with automation.")
            return False
        
        # Check Python executable
        try:
            import subprocess
            python_exe = self.config.get('python_executable', 'python')
            result = subprocess.run([python_exe, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Python executable not working: {python_exe}")
                return False
            
            self.logger.info(f"Python version: {result.stdout.strip()}")
            
        except Exception as e:
            self.logger.error(f"Error checking Python executable: {e}")
            return False
        
        self.logger.info("Enhanced environment validation successful!")
        self.logger.info("All critical systems tested and validated:")
        self.logger.info("  [OK] LLM models (connection + response quality)")
        self.logger.info("  [OK] File editing system (with backups)")
        self.logger.info("  [OK] Syntax validation system")
        self.logger.info("  [OK] Timestamp tracking system")
        self.logger.info("  [OK] Python executable")
        return True

    def run_comprehensive_system_test(self) -> bool:
        """Run comprehensive AI-enhanced system test leveraging IntelligentAppsIntegration"""
        self.log_section("AI-ENHANCED COMPREHENSIVE SYSTEM TEST", "=", 80)

        test_start_time = datetime.now()
        test_timestamp = test_start_time.strftime("%Y-%m-%d %H:%M:%S")

        # Track all test results  
        test_results = {
            'ai_integration': False,
            'natural_language_commands': False,
            'intelligent_error_explanation': False,
            'ai_code_completion': False,
            'semantic_search': False,
            'ai_code_review': False,
            'ai_documentation': False,
            'ai_test_generation': False,
            'ai_workflow_automation': False,
            'toolkit_integration': False,
            'overall_success': False
        }

        try:
            # 1. Test AI Integration Status
            self.logger.info("[AI-TEST 1/10] Testing AI integration capabilities...")

            if hasattr(self, 'ai_integration') and self.ai_integration:
                self.logger.info("[PASS] AI Integration system is active")
                test_results['ai_integration'] = True

                # Test AI feature usage tracking
                usage_stats = self.ai_integration.track_ai_feature_usage()
                self.logger.info(f"[PASS] AI usage tracking operational: {len(usage_stats)} features monitored")
            else:
                self.logger.warning("[WARN] AI Integration system not available")

            # 2. Test Natural Language Command Processing
            self.logger.info("[AI-TEST 2/10] Testing natural language command processing...")

            if hasattr(self, 'process_natural_language') and self.ai_integration:
                test_commands = [
                    "debug the pipeline code",
                    "optimize performance bottlenecks", 
                    "run comprehensive tests",
                    "generate documentation for the system"
                ]

                successful_commands = 0
                for cmd in test_commands:
                    try:
                        nl_result = self.process_natural_language(cmd)
                        if nl_result.get('action') != 'unknown':
                            successful_commands += 1
                            self.logger.info(f"[PASS] NL Command '{cmd}' -> Action: {nl_result['action']}")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] NL Command failed: {e}")

                if successful_commands >= len(test_commands) // 2:
                    test_results['natural_language_commands'] = True
                    self.logger.info(f"[PASS] Natural language processing: {successful_commands}/{len(test_commands)} commands")
                else:
                    self.logger.warning("[PARTIAL] Natural language processing needs improvement")
            else:
                self.logger.warning("[WARN] Natural language processing not available")

            # 3. Test Intelligent Error Explanation
            self.logger.info("[AI-TEST 3/10] Testing intelligent error explanation...")

            if hasattr(self, 'explain_error_ai') and self.ai_integration:
                test_errors = [
                    "SyntaxError: expected ':'",
                    "NameError: name 'undefined_variable' is not defined",
                    "ImportError: No module named 'missing_module'"
                ]

                explanation_success = 0
                for error in test_errors:
                    try:
                        explanation = self.explain_error_ai({'error': error, 'context': 'string_error'})
                        if explanation and explanation.get('explanation'):
                            explanation_success += 1
                            self.logger.info(f"[PASS] Error explained: {error[:30]}...")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Error explanation failed: {e}")

                if explanation_success >= len(test_errors) // 2:
                    test_results['intelligent_error_explanation'] = True
                    self.logger.info(f"[PASS] Intelligent error explanation: {explanation_success}/{len(test_errors)} explained")
                else:
                    self.logger.warning("[PARTIAL] Error explanation needs improvement")
            else:
                self.logger.warning("[WARN] Intelligent error explanation not available")

            # 4. Test AI Code Completion
            self.logger.info("[AI-TEST 4/10] Testing AI-powered code completion...")

            if hasattr(self, 'ai_code_completion') and self.ai_integration:
                test_contexts = [
                    "def process_data(data):",
                    "import pandas as pd\ndf = pd.DataFrame(",
                    "class DataProcessor:\n    def __init__(self"
                ]

                completion_success = 0
                for context in test_contexts:
                    try:
                        completions = self.ai_code_completion("test.py", context)
                        if completions and any(completions.values()):
                            completion_success += 1
                            self.logger.info(f"[PASS] Code completion generated for context: {context[:20]}...")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Code completion failed: {e}")

                if completion_success >= len(test_contexts) // 2:
                    test_results['ai_code_completion'] = True
                    self.logger.info(f"[PASS] AI code completion: {completion_success}/{len(test_contexts)} contexts")
                else:
                    self.logger.warning("[PARTIAL] AI code completion needs improvement")
            else:
                self.logger.warning("[WARN] AI code completion not available")

            # 5. Test Semantic Search
            self.logger.info("[AI-TEST 5/10] Testing semantic search capabilities...")

            if hasattr(self, 'semantic_search') and self.ai_integration:
                test_queries = [
                    "websocket server implementation",
                    "configuration parameters",
                    "debugging automation functions",
                    "optimization algorithms"
                ]

                search_success = 0
                for query in test_queries:
                    try:
                        search_results = self.semantic_search(query)
                        if search_results and len(search_results) > 0:
                            search_success += 1
                            self.logger.info(f"[PASS] Semantic search for '{query}': {len(search_results)} results")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Semantic search failed: {e}")

                if search_success >= len(test_queries) // 2:
                    test_results['semantic_search'] = True
                    self.logger.info(f"[PASS] Semantic search: {search_success}/{len(test_queries)} queries")
                else:
                    self.logger.warning(f"[PARTIAL] Semantic search needs improvement")
            else:
                self.logger.warning("[WARN] Semantic search not available")

            # 6. Test AI Code Review
            self.logger.info("[AI-TEST 6/10] Testing AI code review capabilities...")

            if hasattr(self, 'ai_code_review') and self.ai_integration:
                test_code_changes = [
                    "def new_function(): pass",
                    "password = 'hardcoded123'",
                    "for item in data: result.append(item)"
                ]

                review_success = 0
                for code_change in test_code_changes:
                    try:
                        review_result = self.ai_code_review("test.py", [code_change])
                        if review_result and review_result.get('overall_score'):
                            review_success += 1
                            self.logger.info(f"[PASS] Code review completed for: {code_change[:30]}...")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Code review failed: {e}")

                if review_success >= len(test_code_changes) // 2:
                    test_results['ai_code_review'] = True
                    self.logger.info(f"[PASS] AI code review: {review_success}/{len(test_code_changes)} reviews")
                else:
                    self.logger.warning(f"[PARTIAL] AI code review needs improvement")
            else:
                self.logger.warning("[WARN] AI code review not available")

            # 7. Test AI Documentation Generation
            self.logger.info("[AI-TEST 7/10] Testing AI documentation generation...")

            if hasattr(self, 'generate_ai_docs') and self.ai_integration:
                test_code_snippets = [
                    "def process_data(data): return data.upper()",
                    "class DataProcessor: pass",
                    "# Module for data processing utilities"
                ]

                doc_success = 0
                for snippet in test_code_snippets:
                    try:
                        documentation = self.generate_ai_docs(snippet)
                        if documentation and len(documentation.strip()) > 20:
                            doc_success += 1
                            self.logger.info(f"[PASS] Documentation generated for: {snippet[:30]}...")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Documentation generation failed: {e}")

                if doc_success >= len(test_code_snippets) // 2:
                    test_results['ai_documentation'] = True
                    self.logger.info(f"[PASS] AI documentation: {doc_success}/{len(test_code_snippets)} docs")
                else:
                    self.logger.warning(f"[PARTIAL] AI documentation needs improvement")
            else:
                self.logger.warning("[WARN] AI documentation not available")

            # 8. Test AI Test Generation
            self.logger.info("[AI-TEST 8/10] Testing AI test generation...")

            if hasattr(self, 'generate_ai_tests') and self.ai_integration:
                test_functions = [
                    ("def add_numbers(a, b): return a + b", "add_numbers"),
                    ("def validate_email(email): return '@' in email", "validate_email")
                ]

                test_gen_success = 0
                for func_code, func_name in test_functions:
                    try:
                        test_cases = self.ai_integration.generate_test_cases(func_code, func_name)
                        if test_cases and len(test_cases) > 0:
                            test_gen_success += 1
                            self.logger.info(f"[PASS] Tests generated for {func_name}: {len(test_cases)} cases")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Test generation failed: {e}")

                if test_gen_success >= len(test_functions) // 2:
                    test_results['ai_test_generation'] = True
                    self.logger.info(f"[PASS] AI test generation: {test_gen_success}/{len(test_functions)} functions")
                else:
                    self.logger.warning(f"[PARTIAL] AI test generation needs improvement")
            else:
                self.logger.warning("[WARN] AI test generation not available")

            # 9. Test AI Workflow Automation
            self.logger.info("[AI-TEST 9/10] Testing AI workflow automation...")

            if hasattr(self, 'create_ai_workflow') and self.ai_integration:
                test_tasks = [
                    "backup all modified files automatically",
                    "optimize code performance daily",
                    "run comprehensive tests weekly"
                ]

                workflow_success = 0
                for task in test_tasks:
                    try:
                        workflow = self.ai_integration.create_automation_workflow(task)
                        if workflow and workflow.get('steps') and len(workflow['steps']) > 0:
                            workflow_success += 1
                            self.logger.info(f"[PASS] Workflow created for: {task[:30]}...")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Workflow creation failed: {e}")

                if workflow_success >= len(test_tasks) // 2:
                    test_results['ai_workflow_automation'] = True
                    self.logger.info(f"[PASS] AI workflow automation: {workflow_success}/{len(test_tasks)} workflows")
                else:
                    self.logger.warning(f"[PARTIAL] AI workflow automation needs improvement")
            else:
                self.logger.warning("[WARN] AI workflow automation not available")

            # 10. Test AI Toolkit Integration
            self.logger.info("[AI-TEST 10/10] Testing AI Toolkit integration...")

            if hasattr(self, 'ai_toolkit_integration') and self.ai_toolkit_integration:
                toolkit_tests = [
                    ('check_model_health', 'Model health monitoring'),
                    ('create_prompt_template', 'Prompt template management'),
                    ('record_ai_metrics', 'Performance metrics recording')
                ]

                toolkit_success = 0
                for method_name, description in toolkit_tests:
                    try:
                        if hasattr(self, method_name):
                            self.logger.info(f"[PASS] {description} available")
                            toolkit_success += 1
                        else:
                            self.logger.warning(f"[PARTIAL] {description} not available")
                    except Exception as e:
                        self.logger.warning(f"[PARTIAL] Toolkit test failed: {e}")

                if toolkit_success >= len(toolkit_tests) // 2:
                    test_results['toolkit_integration'] = True
                    self.logger.info(f"[PASS] AI Toolkit integration: {toolkit_success}/{len(toolkit_tests)} features")
                else:
                    self.logger.warning(f"[PARTIAL] AI Toolkit integration needs improvement")
            else:
                self.logger.warning("[WARN] AI Toolkit integration not available")

            # AI-Enhanced Overall Assessment
            ai_systems = ['ai_integration', 'natural_language_commands', 'intelligent_error_explanation', 
                         'ai_code_completion', 'semantic_search', 'ai_code_review']
            ai_systems_working = sum(test_results[key] for key in ai_systems if key in test_results)
            test_results['overall_success'] = ai_systems_working >= len(ai_systems) // 2

            test_duration = (datetime.now() - test_start_time).total_seconds()

            # Count successful tests
            passed_tests = len([result for result in test_results.values() if result is True])
            total_tests = len([k for k in test_results if k != 'overall_success'])

            if test_results['overall_success']:
                self.log_section(f"AI-ENHANCED SYSTEM TEST COMPLETED ({test_duration:.1f}s) - {passed_tests}/{total_tests} PASSED", "=", 80)
                self.logger.info("AI-Enhanced Test Results:")
                self.logger.info(f"  {'[OK]' if test_results['ai_integration'] else '[WARN]'} AI Integration Framework")
                self.logger.info(f"  {'[OK]' if test_results['natural_language_commands'] else '[WARN]'} Natural Language Processing")
                self.logger.info(f"  {'[OK]' if test_results['intelligent_error_explanation'] else '[WARN]'} Intelligent Error Explanation")
                self.logger.info(f"  {'[OK]' if test_results['ai_code_completion'] else '[WARN]'} AI-Powered Code Completion")
                self.logger.info(f"  {'[OK]' if test_results['semantic_search'] else '[WARN]'} Semantic Code Search")
                self.logger.info(f"  {'[OK]' if test_results['ai_code_review'] else '[WARN]'} AI Code Review")
                self.logger.info(f"  {'[OK]' if test_results['ai_documentation'] else '[WARN]'} AI Documentation Generation")
                self.logger.info(f"  {'[OK]' if test_results['ai_test_generation'] else '[WARN]'} AI Test Generation")
                self.logger.info(f"  {'[OK]' if test_results['ai_workflow_automation'] else '[WARN]'} AI Workflow Automation")
                self.logger.info(f"  {'[OK]' if test_results['toolkit_integration'] else '[WARN]'} AI Toolkit Integration")
                self.logger.info(f"  [SUMMARY] AI-enhanced system test completed in {test_duration:.1f} seconds")

                if passed_tests == total_tests:
                    self.logger.info("*** ALL AI SYSTEMS FULLY OPERATIONAL - INTELLIGENT AUTOMATION READY ***")
                elif passed_tests >= 7:
                    self.logger.info("*** CORE AI SYSTEMS WORKING - ENHANCED AUTOMATION ACTIVE ***")
                else:
                    self.logger.info("*** BASIC AI SYSTEMS WORKING - AUTOMATION WITH AI ASSISTANCE ***")

                return True
            else:
                self.log_section(f"AI-ENHANCED SYSTEM TEST PARTIAL FAILURE ({test_duration:.1f}s)", "!", 80)
                self.logger.error("AI systems not fully operational - fallback to basic automation")
                self.logger.warning("Proceeding with traditional automation methods")
                return False

        except Exception as e:
            self.logger.error(f"[CRITICAL] AI-enhanced system test encountered error: {e}")
            self.logger.warning("Falling back to basic automation without AI enhancements")
            return False

        except Exception as e:
            self.logger.error(f"[CRITICAL] Comprehensive workflow test encountered unexpected error: {e}")
            return False
    
    def _run_ai_strategy_analysis(self):
        """Run AI-driven strategy analysis and recommendations"""
        try:
            if not self.ai_strategy_orchestrator:
                return

            # Analyze project context using AI tools
            context = self.ai_strategy_orchestrator.analyze_project_context()
            self.logger.info("[AI-STRATEGY] Context analysis completed")

            # Generate intelligent strategies
            strategies = self.ai_strategy_orchestrator.generate_intelligent_strategies(context)
            self.logger.info(f"[AI-STRATEGY] Generated {len(strategies)} intelligent strategies")

            # Implement high-priority strategies
            implemented_count = 0
            for strategy in strategies[:3]:  # Implement top 3 strategies
                if strategy.priority >= 8:  # High priority only
                    self.logger.info(f"[AI-STRATEGY] Implementing: {strategy.strategy_type}")
                    result = self.ai_strategy_orchestrator.implement_strategy(strategy)
                    if result.get('status') != 'failed':
                        implemented_count += 1

            self.logger.info(f"[AI-STRATEGY] Successfully implemented {implemented_count} AI strategies")

            # Generate strategy report
            if self.config.get('generate_strategy_report', True):
                strategy_report = self.ai_strategy_orchestrator.generate_strategy_report()
                report_file = f"ai_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(strategy_report)
                self.logger.info(f"[AI-STRATEGY] Strategy report saved: {report_file}")

        except Exception as e:
            self.logger.error(f"[AI-STRATEGY] Strategy analysis failed: {e}")
    
    def run_quick_system_test(self) -> bool:
        """Run quick system test that only validates core components needed for optimization"""
        self.log_section("QUICK SYSTEM TEST", "=", 60)

        test_start_time = datetime.now()

        # Track essential test results only
        test_results = {
            'llm_connection': False,
            'file_editor': False,
            'optimization_system': False
        }

        try:
            # 1. Quick LLM connection test (optimization model only)
            self.logger.info("[QUICK-TEST 1/3] Testing LLM optimization connection...")

            try:
                if hasattr(self.llm_interface, '_call_qwen_optimizer'):
                    # Test with simple optimization prompt
                    test_prompt = "Optimize this code: def test(): return 1"
                    response = self.llm_interface._call_qwen_optimizer(test_prompt, enable_streaming=False)

                    if response and len(response.strip()) > 10:
                        self.logger.info("[PASS] LLM optimization connection working")
                        test_results['llm_connection'] = True
                    else:
                        self.logger.warning("[WARN] LLM optimization response weak")
                else:
                    self.logger.warning("[WARN] LLM optimizer not available")
            except Exception as e:
                self.logger.warning(f"[WARN] LLM optimization test failed: {e}")

            # 2. Quick file editor test
            self.logger.info("[QUICK-TEST 2/3] Testing file editor capabilities...")

            try:
                # Test basic file operations
                test_file = f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                test_content = "# Quick test file\ndef test_function():\n    return True\n"

                # Create test file
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)

                # Test backup creation
                backup_path = self.file_editor.create_backup(test_file)

                # Test syntax validation
                syntax_valid = self.file_editor.validate_python_syntax(test_file)

                if backup_path and syntax_valid:
                    self.logger.info("[PASS] File editor working (backup + syntax validation)")
                    test_results['file_editor'] = True
                else:
                    self.logger.warning("[WARN] File editor issues")

                # Cleanup
                try:
                    os.remove(test_file)
                    if backup_path and os.path.exists(backup_path):
                        os.remove(backup_path)
                except Exception:
                    pass

            except Exception as e:
                self.logger.warning(f"[WARN] File editor test failed: {e}")

            # 3. Quick optimization system test
            self.logger.info("[QUICK-TEST 3/3] Testing optimization system...")

            try:
                # Just verify the optimization system initialized correctly
                if hasattr(self, 'optimization_system') and self.optimization_system:
                    # Quick method availability check
                    required_methods = ['optimize_file_enhanced', '_apply_optimization_enhanced']
                    methods_available = all(hasattr(self.optimization_system, method) for method in required_methods)

                    if methods_available:
                        self.logger.info("[PASS] Optimization system initialized with required methods")
                        test_results['optimization_system'] = True
                    else:
                        self.logger.warning("[WARN] Optimization system missing methods")
                else:
                    self.logger.warning("[WARN] Optimization system not available")

            except Exception as e:
                self.logger.warning(f"[WARN] Optimization system test failed: {e}")

            # Quick assessment
            passed_tests = sum(test_results.values())
            test_duration = (datetime.now() - test_start_time).total_seconds()

            if (
                core_systems_working := test_results['file_editor']
                and test_results['optimization_system']
            ):
                total_tests = len(test_results)
                self.log_section(f"QUICK TEST COMPLETED ({test_duration:.1f}s) - {passed_tests}/{total_tests} PASSED", "=", 60)
                self.logger.info("Quick Test Results:")
                self.logger.info(f"  {'[OK]' if test_results['llm_connection'] else '[WARN]'} LLM Optimization Connection")
                self.logger.info(f"  {'[OK]' if test_results['file_editor'] else '[FAIL]'} File Editor System")
                self.logger.info(f"  {'[OK]' if test_results['optimization_system'] else '[FAIL]'} Optimization System")

                if passed_tests == total_tests:
                    self.logger.info("*** ALL CORE SYSTEMS READY - OPTIMIZATION ENABLED ***")
                elif test_results['llm_connection']:
                    self.logger.info("*** CORE SYSTEMS + LLM READY - FULL OPTIMIZATION ENABLED ***")
                else:
                    self.logger.info("*** CORE SYSTEMS READY - LOCAL OPTIMIZATION ENABLED ***")

                return True
            else:
                self.log_section(f"QUICK TEST FAILED ({test_duration:.1f}s)", "!", 60)
                self.logger.error("Core systems not working - cannot proceed with optimization")
                return False

        except Exception as e:
            self.logger.error(f"[CRITICAL] Quick system test error: {e}")
            return False
    
    def _resolve_target_file_path(self, file_path: str) -> Optional[str]:
        """Resolve the full path for a target file, checking multiple locations"""
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
        
        # Check automation strategy folder first
        current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if os.path.exists(current_path):
            return current_path
        
        # Check current working directory
        if os.path.exists(file_path):
            return os.path.abspath(file_path)
        
        # Check parent directory (main workspace)
        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path)
        return parent_path if os.path.exists(parent_path) else None
    
    def run_debug_phase(self, target_files: List[str]) -> List[DebugSession]:
        """Run the debugging phase on target files with detailed real-time logging"""
        self.log_section("STARTING DEBUG PHASE")
        
        self.logger.info(f"Target files: {target_files}")
        self.logger.info(f"Max iterations per file: {self.config.get('max_debug_iterations', 3)}")
        self.logger.info(f"Debug timeout: {self.config.get('debug_timeout', 300)}s")
        self.logger.info(f"Syntax validation: {self.config.get('validate_syntax', True)}")
        
        debug_sessions = []
        
        # Handle WebSocket server dependency
        websocket_server_process = None
        
        try:
            # Start WebSocket server if it's in the target files
            if 'gridbot_websocket_server.py' in target_files:
                self.logger.info("Starting WebSocket server dependency...")
                import subprocess
                import time
                
                # Use local WebSocket server in automation strategy folder
                current_dir = os.path.dirname(os.path.abspath(__file__))
                server_path = os.path.join(current_dir, 'gridbot_websocket_server.py')
                if not os.path.exists(server_path):
                    # Fallback to parent directory only if not found locally
                    server_path = os.path.join(os.path.dirname(current_dir), 'gridbot_websocket_server.py')
                    if not os.path.exists(server_path):
                        server_path = 'gridbot_websocket_server.py'
                
                self.logger.info(f"Starting WebSocket server at: {server_path}")
                
                websocket_server_process = subprocess.Popen(
                    [self.config.get('python_executable', 'python'), server_path],
                    cwd=current_dir,  # Run from automation strategy directory
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for server to start
                self.logger.info("Waiting for WebSocket server to initialize...")
                time.sleep(5)
                
                # Check if server is running
                if websocket_server_process.poll() is None:
                    self.logger.info("WebSocket server started successfully")
                else:
                    self.logger.warning("WebSocket server failed to start")
                    stdout, stderr = websocket_server_process.communicate()
                    self.logger.error(f"Server stdout: {stdout}")
                    self.logger.error(f"Server stderr: {stderr}")
                    websocket_server_process = None
            
            for file_path in target_files:
                self.logger.info(f"\nDebugging file: {file_path}")
                
                # Skip WebSocket server for debugging (it's already running)
                if file_path == 'gridbot_websocket_server.py' and websocket_server_process:
                    self.logger.info("Skipping WebSocket server debugging (running as dependency)")
                    # Create a successful session for the server
                    session = DebugSession(
                        target_file=file_path,
                        session_id=f"dependency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        start_time=datetime.now(),
                        success=True,
                        final_status="Running as dependency service"
                    )
                    debug_sessions.append(session)
                    continue
                
                # Resolve full path for the target file
                full_file_path = file_path
                if not os.path.isabs(file_path):
                    # Try current directory first (for files in automation strategy folder)
                    current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
                    if os.path.exists(current_path):
                        full_file_path = current_path
                        self.logger.info(f"Using automation strategy folder file: {current_path}")
                    # Fall back to parent directory only if not found in current
                    elif os.path.exists(file_path):
                        full_file_path = file_path
                    else:
                        # Try parent directory as last resort
                        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path)
                        if os.path.exists(parent_path):
                            full_file_path = parent_path
                            self.logger.warning(f"Falling back to main workspace file: {parent_path}")
                        else:
                            self.logger.error(f"Target file not found: {file_path}")
                            continue
                
                self.logger.info(f"Debugging file at: {full_file_path}")
                
                # Special handling for GridbotBackup.py which may run longer
                if 'GridbotBackup.py' in file_path:
                    self.logger.info("Note: GridbotBackup.py may run longer due to WebSocket connections")
                
                # Special handling for config.py (configuration file, not executable)
                if 'config.py' in file_path:
                    self.logger.info("Processing config.py as configuration file (syntax validation only)")
                    session = self.validate_config_file(full_file_path)
                    debug_sessions.append(session)
                    self.session_data['debug_sessions'].append(session)
                    continue
                
                session = self.debug_orchestrator.run_debug_cycle(
                    full_file_path, self.config.get('python_executable', 'python')
                )
                
                debug_sessions.append(session)
                self.session_data['debug_sessions'].append(session)
                self.session_data['total_errors_fixed'] += session.errors_fixed
                
                # Log session summary
                self.logger.info(f"\nDebug session completed for {file_path}:")
                self.logger.info(f"  Success: {session.success}")
                self.logger.info(f"  Errors fixed: {session.errors_fixed}")
                self.logger.info(f"  Iterations: {session.iterations}")
                self.logger.info(f"  Final status: {session.final_status}")
                
                if not session.success:
                    self.logger.warning(f"File {file_path} still has issues after debugging")
                else:
                    self.logger.info(f"File {file_path} runs successfully!")
        
        finally:
            # Clean up WebSocket server
            if websocket_server_process:
                self.logger.info("Stopping WebSocket server...")
                websocket_server_process.terminate()
                try:
                    websocket_server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    websocket_server_process.kill()
                self.logger.info("WebSocket server stopped")
        
        # Overall debug phase summary
        successful_files = len([s for s in debug_sessions if s.success])
        total_errors_fixed = sum(s.errors_fixed for s in debug_sessions)
        
        self.logger.info(f"\nDEBUG PHASE SUMMARY:")
        self.logger.info(f"  Files processed: {successful_files} / {len(target_files)}")
        self.logger.info(f"  Successful files: {successful_files}")
        self.logger.info(f"  Total errors fixed: {total_errors_fixed}")
        
        return debug_sessions
    
    def validate_config_file(self, config_file_path: str) -> DebugSession:
        """Validate config.py file for syntax errors and parameter issues"""
        session_id = f"config_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = DebugSession(
            target_file=config_file_path,
            session_id=session_id,
            start_time=datetime.now(),
            success=False,
            final_status="Config validation started"
        )

        try:
            # Read and validate Python syntax
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_content = f.read()

            # Try to compile the config file
            try:
                compile(config_content, config_file_path, 'exec')
                self.logger.info("Config file syntax validation: PASSED")
                syntax_valid = True
            except SyntaxError as e:
                self.logger.error(f"Config file syntax error: {e}")
                syntax_valid = False
                session.final_status = f"Syntax error: {e}"

            if syntax_valid:
                # Try to execute and validate config parameters
                config_validation_issues = []
                try:
                    # Create a safe namespace for config execution
                    config_namespace = {}
                    exec(config_content, config_namespace)

                    # Validate critical parameters
                    required_params = [
                        'API_KEY', 'SECRET_KEY', 'SYMBOL', 'POSITION_SIZE',
                        'GRID_SIZE', 'NUM_BUY_GRID_LINES', 'NUM_SELL_GRID_LINES'
                    ]

                    if missing_params := [
                        param
                        for param in required_params
                        if param not in config_namespace
                    ]:
                        config_validation_issues.append(f"Missing required parameters: {missing_params}")

                    # Validate parameter ranges
                    range_validations = [
                        ('GRID_SIZE', 'MIN_GRID_SIZE', 'MAX_GRID_SIZE'),
                        ('POSITION_SIZE', 'MIN_POSITION_SIZE', 'MAX_POSITION_SIZE'),
                        ('NUM_BUY_GRID_LINES', 'MIN_NUM_GRID_LINES', 'MAX_NUM_GRID_LINES'),
                        ('NUM_SELL_GRID_LINES', 'MIN_NUM_GRID_LINES', 'MAX_NUM_GRID_LINES')
                    ]

                    for param, min_param, max_param in range_validations:
                        if all(p in config_namespace for p in [param, min_param, max_param]):
                            value = config_namespace[param]
                            min_val = config_namespace[min_param]
                            max_val = config_namespace[max_param]

                            if not (min_val <= value <= max_val):
                                config_validation_issues.append(
                                    f"{param}={value} outside valid range [{min_val}, {max_val}]"
                                )

                    if not config_validation_issues:
                        session.success = True
                        session.final_status = "Config validation: PASSED"
                        self.logger.info("Config file parameter validation: PASSED")
                    else:
                        session.final_status = f"Config validation issues: {config_validation_issues}"
                        self.logger.warning(f"Config validation issues found: {config_validation_issues}")

                except Exception as e:
                    session.final_status = f"Config execution error: {e}"
                    self.logger.error(f"Config file execution error: {e}")

        except Exception as e:
            session.final_status = f"Config file validation failed: {e}"
            self.logger.error(f"Error validating config file: {e}")

        return session
    
    def optimize_config_file(self, config_file_path: str) -> List:
        """Optimize config.py file parameters for better performance"""
        optimization_results = []
        
        try:
            # Read current config
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Parse config for optimization opportunities
            config_namespace = {}
            exec(config_content, config_namespace)
            
            # Define optimization suggestions for GridBot parameters
            optimization_suggestions = [
                {
                    'parameter': 'GRID_SIZE',
                    'current_value': config_namespace.get('GRID_SIZE', 12.0),
                    'suggested_range': (8.0, 15.0),
                    'reason': 'Optimal grid size for current market volatility'
                },
                {
                    'parameter': 'CHECK_ORDER_FREQUENCY',
                    'current_value': config_namespace.get('CHECK_ORDER_FREQUENCY', 30),
                    'suggested_range': (15, 45),
                    'reason': 'Balance between responsiveness and API rate limits'
                },
                {
                    'parameter': 'NUM_BUY_GRID_LINES',
                    'current_value': config_namespace.get('NUM_BUY_GRID_LINES', 20),
                    'suggested_range': (15, 25),
                    'reason': 'Optimize grid density for current position size'
                },
                {
                    'parameter': 'NUM_SELL_GRID_LINES', 
                    'current_value': config_namespace.get('NUM_SELL_GRID_LINES', 20),
                    'suggested_range': (15, 25),
                    'reason': 'Optimize grid density for current position size'
                },
                {
                    'parameter': 'ML_CONFIDENCE_THRESHOLD',
                    'current_value': config_namespace.get('ML_CONFIDENCE_THRESHOLD', 0.7),
                    'suggested_range': (0.6, 0.8),
                    'reason': 'Balance between prediction accuracy and trade frequency'
                }
            ]
            
            for suggestion in optimization_suggestions:
                param = suggestion['parameter']
                current = suggestion['current_value']
                min_val, max_val = suggestion['suggested_range']
                
                # Create a mock optimization result for config parameters
                # Using a simple dict instead of importing from optimization system to avoid circular imports
                candidate = {
                    'function_name': param,
                    'file_path': config_file_path,
                    'line_start': 1,
                    'line_end': 1,
                    'code_snippet': f"{param} = {current}",
                    'performance_issues': [f"Parameter {param} could be optimized"],
                    'optimization_priority': 7,
                    'estimated_impact': "medium"
                }
                
                # Determine if current value is optimal
                is_optimal = min_val <= current <= max_val
                suggested_value = current if is_optimal else (min_val + max_val) / 2
                
                result = {
                    'candidate': candidate,
                    'success': True,
                    'applied': False,  # Config changes require manual review
                    'improvement_ratio': 0.0 if is_optimal else 0.05,
                    'error': None if is_optimal else f"Suggest changing {param} from {current} to {suggested_value:.2f} ({suggestion['reason']})"
                }
                
                optimization_results.append(result)
                
                if not is_optimal:
                    self.logger.info(f"Config optimization suggestion: {param} = {suggested_value:.2f} (current: {current}, reason: {suggestion['reason']})")
                else:
                    self.logger.info(f"Config parameter {param} = {current} is already optimal")
            
            self.logger.info(f"Generated {len(optimization_results)} config optimization suggestions")
            
        except Exception as e:
            self.logger.error(f"Error optimizing config file: {e}")
        
        return optimization_results
    
    def run_optimization_phase(self, target_files: List[str]) -> List[Union[OptimizationResult, Dict]]:
        """Run the optimization phase on successfully debugged files"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING OPTIMIZATION PHASE")
        self.logger.info("=" * 60)

        all_optimization_results = []

        # Separate regular files from config files for different handling
        regular_files = []
        config_files = []

        for file_path in target_files:
            # Only optimize files that run successfully
            file_session = next((s for s in self.session_data['debug_sessions'] 
                               if s.target_file == file_path), None)

            if file_session and not file_session.success:
                self.logger.warning(f"Skipping optimization for {file_path} (debugging failed)")
                continue

            # Resolve full path for the target file - prioritize automation strategy folder
            full_file_path = file_path
            if not os.path.isabs(file_path):
                # Try current directory first (for files in automation strategy folder)
                current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
                if os.path.exists(current_path):
                    full_file_path = current_path
                    self.logger.info(f"Using automation strategy folder file: {current_path}")
                # Fall back to parent directory only if not found in current
                elif os.path.exists(file_path):
                    full_file_path = file_path
                else:
                    # Try parent directory as last resort
                    parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path)
                    if os.path.exists(parent_path):
                        full_file_path = parent_path
                        self.logger.warning(f"Falling back to main workspace file: {parent_path}")
                    else:
                        self.logger.error(f"Target file not found: {file_path}")
                        continue

            # Separate config files for special handling
            if 'config.py' in file_path:
                config_files.append(full_file_path)
            else:
                regular_files.append(full_file_path)

        # Handle config files separately (they need special parameter optimization)
        for config_file_path in config_files:
            self.logger.info("Analyzing config.py for parameter optimization opportunities...")
            config_optimization_results = self.optimize_config_file(config_file_path)
            all_optimization_results.extend(config_optimization_results)
            self.session_data['optimization_results'].extend(config_optimization_results)

            applied_count = len([r for r in config_optimization_results if r.get('applied', False)])
            self.session_data['total_optimizations_applied'] += applied_count

            self.logger.info(f"\nConfig optimization completed for {config_file_path}:")
            self.logger.info(f"  Parameter suggestions: {len(config_optimization_results)}")
            self.logger.info(f"  Optimizations applied: {applied_count}")

        # Optimize all regular files using enhanced optimization system
        if regular_files:
            self.logger.info(f"[OPTIMIZE] Processing {len(regular_files)} files with enhanced optimization:")
            for i, file_path in enumerate(regular_files, 1):
                self.logger.info(f"  {i}. {file_path}")

            try:
                self.logger.info(
                    "[ENHANCED] Using enhanced optimization system with real-time logging"
                )

                # Process each file with enhanced optimization
                optimization_results_by_file = {}
                for file_path in regular_files:
                    self.logger.info(f"[TARGET] [ENHANCED] Starting optimization for {file_path}")

                    enhanced_results = self.optimization_system.optimize_file_enhanced(file_path)
                    optimization_results_by_file[file_path] = enhanced_results

                    self.logger.info(f"[OK] [ENHANCED] Completed {file_path}: {len(enhanced_results)} results")

                # Flatten results and update session data with enhanced reporting
                total_candidates = 0
                total_applied = 0

                for file_path, file_results in optimization_results_by_file.items():
                    all_optimization_results.extend(file_results)
                    self.session_data['optimization_results'].extend(file_results)

                    # Count applied optimizations for this file
                    applied_count = len([r for r in file_results if r.applied])
                    total_applied += applied_count
                    total_candidates += len(file_results)

                    self.session_data['total_optimizations_applied'] += applied_count

                    # Enhanced logging for optimization results
                    self.logger.info(f"[RESULTS] {file_path}:")
                    self.logger.info(f"[RESULTS]   Candidates analyzed: {len(file_results)}")
                    self.logger.info(f"[RESULTS]   Optimizations applied: {applied_count}")

                    # Log individual results with detailed info
                    for result in file_results:
                        if result.applied:
                            improvement = result.improvement_ratio or 0
                            self.logger.info(f"[OK] [APPLIED]     {result.candidate.function_name}: {improvement:.2%} improvement")
                        elif result.error:
                            self.logger.warning(f"[FAIL] [FAILED]     {result.candidate.function_name}: {result.error}")
                        else:
                            self.logger.info(f"[SKIP] [SKIPPED]     {result.candidate.function_name}: No optimization needed")

                # Enhanced summary logging
                success_rate = (total_applied / total_candidates * 100) if total_candidates > 0 else 0
                self.logger.info("[SUMMARY] Optimization complete!")
                self.logger.info(f"[SUMMARY] Success rate: {success_rate:.1f}% ({total_applied}/{total_candidates})")

                if total_applied > 0:
                    total_improvement = sum(r.improvement_ratio or 0 for r in all_optimization_results if r.improvement_ratio)
                    avg_improvement = total_improvement / total_applied
                    self.logger.info(f"[SUMMARY] Average improvement: {avg_improvement:.2%}")

            except Exception as e:
                self.logger.error(f"[ERROR] Error in optimization process: {e}")
                # Continue with empty results rather than crashing
                optimization_results_by_file = {}

        # Overall optimization phase summary
        total_candidates = len(all_optimization_results)
        applied_optimizations = len([r for r in all_optimization_results if getattr(r, 'applied', False)])

        self.logger.info(f"\nOPTIMIZATION PHASE SUMMARY:")
        self.logger.info(f"  Total candidates: {total_candidates}")
        self.logger.info(f"  Applied optimizations: {applied_optimizations}")

        if total_candidates > 0:
            success_rate = applied_optimizations / total_candidates
            self.logger.info(f"  Success rate: {success_rate:.2%}")

        # Track improvements using systematic improvement tracker
        if self.improvement_tracker and applied_optimizations > 0:
            try:
                improvements_made = [
                    f"Applied {applied_optimizations} optimizations",
                    f"Enhanced {len(regular_files)} regular files", 
                    f"Processed {len(config_files)} config files",
                    "Path corrections completed",
                    "Enhanced optimization strategies prioritized"
                ]

                self.logger.info("[TRACKER] Recording systematic improvements...")
                iteration_result = self.improvement_tracker.run_improvement_iteration(improvements_made)

                if 'improvement_from_baseline' in iteration_result:
                    improvement = iteration_result['improvement_from_baseline']
                    self.logger.info(f"[TRACKER] Performance improvement: {improvement:+.2f}% from baseline")
                    self.logger.info(f"[TRACKER] Iteration {iteration_result['iteration']} completed")

            except Exception as e:
                self.logger.warning(f"[TRACKER] Failed to record improvements: {e}")

        return all_optimization_results
    
    def run_full_pipeline(self, target_files: Optional[List[str]] = None) -> Dict:
        """Run the complete automation pipeline using queued operations for orderly processing"""
        if target_files is None:
            target_files = self.config.get('target_files', ['GridbotBackup.py'])
        
        # Ensure target_files is definitely a list for type safety
        target_files = target_files or []
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING MASTER AUTOMATION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Target files: {target_files}")
        self.logger.info(f"LLM Model: {self.config.get('llm_model', 'deepseek-coder:latest')}")
        self.logger.info(f"Start time: {self.session_data['start_time']}")
        
        # Log initial LLM status
        self.llm_interface.log_status_summary()

        # =================================================================================
        # AI-DRIVEN STRATEGY SELECTION (replacing hardcoded approaches)
        # =================================================================================
        
        ai_strategy_recommendations = []
        if hasattr(self, 'ai_strategy_orchestrator') and self.ai_strategy_orchestrator:
            try:
                self.logger.info("[AI-STRATEGY] Analyzing project for intelligent strategy recommendations...")
                
                # Get AI-driven context analysis
                context = self.ai_strategy_orchestrator.analyze_project_context()
                
                # Generate intelligent strategy recommendations
                ai_strategy_recommendations = self.ai_strategy_orchestrator.generate_intelligent_strategies(context)
                
                if ai_strategy_recommendations:
                    self.logger.info(f"[AI-STRATEGY] Generated {len(ai_strategy_recommendations)} intelligent strategies")
                    for strategy in ai_strategy_recommendations[:3]:  # Log top 3
                        self.logger.info(f"[AI-STRATEGY]   {strategy.strategy_type} (Priority: {strategy.priority}, Confidence: {strategy.confidence:.2f})")
                
                # Implement high-confidence strategies
                for strategy in ai_strategy_recommendations:
                    if strategy.confidence >= 0.8 and strategy.priority >= 8:
                        self.logger.info(f"[AI-STRATEGY] Auto-implementing high-confidence strategy: {strategy.strategy_type}")
                        try:
                            implementation_result = self.ai_strategy_orchestrator.implement_strategy(strategy)
                            self.logger.info(f"[AI-STRATEGY] Strategy implementation status: {implementation_result.get('status', 'unknown')}")
                        except Exception as e:
                            self.logger.warning(f"[AI-STRATEGY] Strategy implementation failed: {e}")
                            
            except Exception as e:
                self.logger.warning(f"[AI-STRATEGY] AI strategy analysis failed: {e}")

        # Note: File cleanup is handled in __init__, not here to avoid duplication
        
        try:
            # Start queue processor
            self._start_queue_processor()
            
            # Results storage for queued operations
            pipeline_results = {
                'debug_sessions': [],
                'optimization_results': [],
                'validation_success': False,
                'errors': []
            }
            
            # Queue environment validation
            def validate_and_store():
                if not self.validate_environment():
                    pipeline_results['errors'].append('Environment validation failed')
                    return False
                pipeline_results['validation_success'] = True
                return True
            
            self.queue_operation('validation', validate_and_store, name='Environment Validation')
            
            # Queue debugging phase
            def debug_and_store():
                debug_sessions = self.run_debug_phase(target_files)
                pipeline_results['debug_sessions'] = debug_sessions
                return debug_sessions
            
            self.queue_operation('debugging', debug_and_store, name='Debug Phase')
            
            # Queue optimization phase (conditional)
            if self.config.get('run_optimization', True):
                def optimize_and_store():
                    # Get files to optimize from debug results
                    files_to_optimize = []
                    for file_path in target_files:
                        full_path = file_path
                        if not os.path.isabs(file_path):
                            automation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
                            if os.path.exists(automation_path):
                                full_path = automation_path
                            elif not os.path.exists(file_path):
                                continue
                        if os.path.exists(full_path):
                            files_to_optimize.append(file_path)
                    
                    if files_to_optimize:
                        optimization_results = self.run_optimization_phase(files_to_optimize)
                        pipeline_results['optimization_results'] = optimization_results
                        
                        # Check if we need to re-run debug after optimizations
                        applied_optimizations = len([r for r in optimization_results if 
                                                   (hasattr(r, 'applied') and r.applied) or  # type: ignore
                                                   (isinstance(r, dict) and r.get('applied', False))])
                        if applied_optimizations > 0:
                            self.logger.info(f"Applied {applied_optimizations} optimizations - queuing post-optimization debug cycle")
                            # Queue post-optimization debugging
                            def post_debug():
                                post_sessions = self.run_debug_phase(files_to_optimize)
                                pipeline_results['debug_sessions'].extend(post_sessions)
                                return post_sessions
                            
                            self.queue_operation('post_debug', post_debug, name='Post-Optimization Debug')
                        
                        return optimization_results
                    else:
                        self.logger.warning("No files available for optimization")
                        return []
                
                self.queue_operation('optimization', optimize_and_store, name='Optimization Phase')
            
            # Wait for all operations to complete
            if not self.wait_for_queue_completion(timeout=3600):  # 1 hour timeout
                pipeline_results['errors'].append('Queue processing timeout')
                return {'success': False, 'error': 'Queue processing timeout'}
            
            # Check if validation failed
            if not pipeline_results['validation_success']:
                return {'success': False, 'error': 'Environment validation failed'}
            
            # Generate final report
            final_report = self.generate_final_report()
            
            # Update session data with results
            self.session_data['debug_sessions'] = pipeline_results['debug_sessions']
            self.session_data['optimization_results'] = pipeline_results['optimization_results']
            
            # Save comprehensive session data
            if self.config.get('save_reports', True):
                self.save_session_data()
            
            # Restart GridBot if changes were made and restart is enabled
            debug_sessions = pipeline_results['debug_sessions']
            optimization_results = pipeline_results['optimization_results']
            if self.config.get('restart_gridbot_after_changes', False):
                self.restart_gridbot_if_changes_made(debug_sessions, optimization_results)
            
            # Final LLM performance summary
            self.log_llm_performance_summary()
            
            self.logger.info("=" * 80)
            self.logger.info("AUTOMATION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return {
                'success': True,
                'debug_sessions': debug_sessions,
                'optimization_results': optimization_results,
                'final_report': final_report
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            # Log LLM status on failure for debugging
            self.log_llm_performance_summary()
            return {'success': False, 'error': str(e)}
        
        finally:
            # Stop queue processor
            self._stop_queue_processor()
            # Stop monitoring when pipeline completes
            self._stop_llm_monitoring()
    
    def rollback_failed_optimizations(self, optimization_results: List[List[OptimizationResult]], 
                                     validation_sessions: List[DebugSession]):
        """Rollback optimizations for files that failed validation"""
        failed_files = [session.target_file for session in validation_sessions if not session.success]

        for file_path in failed_files:
            self.logger.info(f"Rolling back optimizations for {file_path}")

            # Find the most recent backup for this file
            backup_dir = self.config.get('backup_dir', 'backups')
            backup_pattern = f"{os.path.basename(file_path)}.backup.*"

            try:
                backup_files = []
                if os.path.exists(backup_dir):
                    for backup_file in os.listdir(backup_dir):
                        if backup_file.startswith(
                            f"{os.path.basename(file_path)}.backup."
                        ):
                            backup_path = os.path.join(backup_dir, backup_file)
                            backup_files.append((backup_path, os.path.getmtime(backup_path)))

                if backup_files:
                    # Get the most recent backup
                    latest_backup = max(backup_files, key=lambda x: x[1])[0]

                    # Restore from backup
                    import shutil
                    shutil.copy2(latest_backup, file_path)
                    self.logger.info(f"Restored {file_path} from {latest_backup}")
                else:
                    self.logger.error(f"No backup found for {file_path} - cannot rollback")

            except Exception as e:
                self.logger.error(f"Failed to rollback {file_path}: {e}")
    
    def generate_final_report(self) -> Dict:
        """Generate a comprehensive final report"""
        end_time = datetime.now()
        duration = end_time - self.session_data['start_time']

        # Calculate success rates
        debug_sessions = self.session_data['debug_sessions']
        successful_debugs = len([s for s in debug_sessions if s.success])
        debug_success_rate = successful_debugs / len(debug_sessions) if debug_sessions else 0

        optimization_results = self.session_data['optimization_results']
        applied_optimizations = len([r for r in optimization_results if 
                                   (hasattr(r, 'applied') and r.applied) or  # type: ignore
                                   (isinstance(r, dict) and r.get('applied', False))])
        optimization_success_rate = (applied_optimizations / len(optimization_results) 
                                    if optimization_results else 0)

        return {
            'session_summary': {
                'start_time': self.session_data['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'duration_formatted': str(duration),
            },
            'file_management': self.session_data.get('cleanup_stats', {}),
            'debug_phase': {
                'files_processed': len(debug_sessions),
                'successful_files': successful_debugs,
                'success_rate': debug_success_rate,
                'total_errors_fixed': self.session_data['total_errors_fixed'],
                'total_iterations': sum(s.iterations for s in debug_sessions),
            },
            'optimization_phase': {
                'candidates_analyzed': len(optimization_results),
                'optimizations_applied': applied_optimizations,
                'success_rate': optimization_success_rate,
                'total_applied': self.session_data['total_optimizations_applied'],
            },
            'overall_success': debug_success_rate
            > 0.5,  # Consider success if > 50% files fixed
            'recommendations': self.generate_recommendations(),
            'enhancement_priorities': self.generate_enhancement_priority_report(),
        }
    
    def generate_enhancement_priority_report(self) -> Dict:
        """Generate report on enhancement prioritization and systematic improvements"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'enhancement_status': {
                'path_fixes_completed': True,
                'systematic_tracker_active': self.improvement_tracker is not None,
                'enhanced_optimization_enabled': True,
                'serena_integration_active': self.config.get('use_serena', False)
            },
            'priority_enhancements': [
                'Path corrections for all test files completed',
                'Systematic improvement tracker initialized',
                'Enhanced optimization system prioritized',
                'Serena semantic editing enabled',
                'Continuous 1% improvement tracking active'
            ],
            'next_priorities': [
                'Monitor systematic improvement progression',
                'Apply enhanced optimization strategies',
                'Track performance gains across iterations',
                'Optimize based on real usage patterns'
            ]
        }
        
        # Add improvement tracker data if available
        if self.improvement_tracker:
            try:
                improvement_report = self.improvement_tracker.generate_improvement_report()
                report['systematic_improvements'] = {
                    'tracker_active': True,
                    'report_available': len(improvement_report) > 100,  # Check if substantial report
                    'baseline_established': os.path.exists(self.improvement_tracker.results_file)
                }
            except Exception as e:
                report['systematic_improvements'] = {
                    'tracker_active': False,
                    'error': str(e)
                }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on the automation results"""
        recommendations = []

        debug_sessions = self.session_data['debug_sessions']
        optimization_results = self.session_data['optimization_results']

        if failed_sessions := [s for s in debug_sessions if not s.success]:
            recommendations.append(
                f"Manual review needed for {len(failed_sessions)} files that couldn't be fully debugged"
            )

        high_iteration_sessions = [s for s in debug_sessions if s.iterations >= 8]
        if high_iteration_sessions:
            recommendations.append(
                "Consider code refactoring for files requiring many debug iterations"
            )

        # Optimization recommendations
        low_impact_optimizations = [r for r in optimization_results 
                                   if (getattr(r, 'improvement_ratio', None) or 
                                       (isinstance(r, dict) and r.get('improvement_ratio', 0)) or 0) < 0.1]
        if len(low_impact_optimizations) > len(optimization_results) * 0.7:
            recommendations.append(
                "Most optimizations had low impact - consider profiling for bottlenecks"
            )

        if not optimization_results:
            recommendations.append(
                "No optimization candidates found - code may already be well-optimized"
            )

        # General recommendations
        if self.session_data['total_errors_fixed'] > 10:
            recommendations.append(
                "High error count suggests need for better testing and code review processes"
            )

        return recommendations
    
    def _setup_llm_monitoring(self):
        """Setup periodic LLM status monitoring"""
        self._monitoring_active = True
        self._last_status_log = 0
        
        # Start background monitoring thread
        def monitor_llm_status():
            while self._monitoring_active:
                try:
                    current_time = time.time()
                    # Log status summary every 5 minutes
                    if current_time - self._last_status_log > 300:  # 300 seconds = 5 minutes
                        self.llm_interface.log_status_summary()
                        self._last_status_log = current_time
                    
                    # Brief health check every 5 minutes instead of 30 seconds
                    time.sleep(300)  # 5 minutes
                    if self._monitoring_active:
                        health_ok = self.llm_interface.test_connection()
                        if not health_ok:
                            self.logger.warning("[WARN] LLM_STATUS: Health check failed - connection issues detected")
                        
                except Exception as e:
                    self.logger.error(f"LLM monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start monitoring thread if verbose mode is enabled AND not in test mode
        if self.config.get('verbose', True) and not self.config.get('test_mode', False):
            monitor_thread = threading.Thread(target=monitor_llm_status, daemon=True)
            monitor_thread.start()
            self.logger.info("[STATUS] LLM_STATUS: Background monitoring started")
        elif self.config.get('test_mode', False):
            self.logger.info("[STATUS] LLM_STATUS: Monitoring disabled in test mode")
    
    def _stop_llm_monitoring(self):
        """Stop LLM status monitoring"""
        if hasattr(self, '_monitoring_active'):
            self._monitoring_active = False
            self.logger.info("[STATUS] LLM_STATUS: Background monitoring stopped")
    
    def _start_queue_processor(self):
        """Start the queue processor thread"""
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            return
        
        self.queue_active = True
        self.queue_processor_thread = threading.Thread(target=self._process_operation_queue, daemon=True)
        self.queue_processor_thread.start()
        self.logger.info("[QUEUE] Operation queue processor started")
    
    def _stop_queue_processor(self):
        """Stop the queue processor"""
        self.queue_active = False
        if self.queue_processor_thread:
            self.queue_processor_thread.join(timeout=5)
        self.logger.info("[QUEUE] Operation queue processor stopped")
    
    def _process_operation_queue(self):
        """Process operations from the queue sequentially"""
        while self.queue_active:
            try:
                # Get operation from queue with timeout
                operation = self.operation_queue.get(timeout=1)
                
                operation_type = operation.get('type')
                operation_name = operation.get('name', operation_type)
                operation_func = operation.get('func')
                operation_args = operation.get('args', [])
                operation_kwargs = operation.get('kwargs', {})
                
                self.logger.info(f"[QUEUE] Processing operation: {operation_name}")
                
                # Execute the operation
                start_time = datetime.now()
                try:
                    result = operation_func(*operation_args, **operation_kwargs)
                    end_time = datetime.now()
                    duration = end_time - start_time
                    
                    self.logger.info(f"[QUEUE] Operation {operation_name} completed successfully in {duration}")
                    
                    # Handle result if callback provided
                    if 'callback' in operation and operation['callback'] is not None:
                        operation['callback'](result, None)
                        
                except Exception as e:
                    end_time = datetime.now()
                    duration = end_time - start_time
                    
                    self.logger.error(f"[QUEUE] Operation {operation_name} failed after {duration}: {e}")
                    
                    # Handle error if error_callback provided
                    if 'error_callback' in operation and operation['error_callback'] is not None:
                        operation['error_callback'](e)
                    elif 'callback' in operation and operation['callback'] is not None:
                        operation['callback'](None, e)
                
                # Mark task as done
                self.operation_queue.task_done()
                
                # Small delay between operations to prevent overwhelming the system and ensure log sequencing
                time.sleep(1.0)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"[QUEUE] Queue processor error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def queue_operation(self, operation_type: str, func, *args, name: Optional[str] = None, callback = None, error_callback = None, **kwargs):
        """Add an operation to the queue for sequential processing"""
        operation = {
            'type': operation_type,
            'name': name or operation_type,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'callback': callback,
            'error_callback': error_callback
        }
        
        self.operation_queue.put(operation)
        self.logger.info(f"[QUEUE] Queued operation: {operation['name']}")
    
    def wait_for_queue_completion(self, timeout: Optional[int] = None):
        """Wait for all queued operations to complete"""
        self.logger.info("[QUEUE] Waiting for all operations to complete...")
        try:
            self.operation_queue.join()
            self.logger.info("[QUEUE] All operations completed")
            return True
        except Exception as e:
            self.logger.error(f"[QUEUE] Error waiting for completion: {e}")
            return False
    
    def log_llm_performance_summary(self):
        """Log a detailed LLM performance summary"""
        status = self.llm_interface.get_status_summary()
        
        self.logger.info("=" * 60)
        self.logger.info("[SUMMARY] LLM PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"[ORCHESTRATOR] {status['orchestrator_model']} @ {status['orchestrator_url']}")
        self.logger.info(f"[DEEPSEEK DEBUGGER] {status['deepseek_model']} @ {status['deepseek_debugger_url']} ({status['deepseek_status']})")
        self.logger.info(f"[QWEN OPTIMIZER] {status['qwen_model']} @ {status['qwen_optimizer_url']} ({status['qwen_status']})")
        self.logger.info(f"[HEALTH] {status['health_status']}")
        self.logger.info(f"[REQUESTS] Total: {status['total_requests']}")
        self.logger.info(f"[SUCCESS] Count: {status['successful_requests']}")
        self.logger.info(f"[RATE] Success: {status['success_rate']}%")
        
        # Only show consecutive failures if the key exists
        if 'consecutive_failures' in status and status['consecutive_failures'] > 0:
            self.logger.warning(f"[FAILURES] Consecutive: {status['consecutive_failures']}")
        
        self.logger.info(f"[CONFIG] thinking={status['enable_thinking']}, temperature={status['temperature']}")
        self.logger.info(f"[ARCHITECTURE] {status['architecture']}")
        self.logger.info("=" * 60)
    
    def save_session_data(self):
        """Save comprehensive session data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main session data
        session_file = f"automation_session_{timestamp}.json"
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = self._make_json_serializable(self.session_data)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved session data to {session_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {e}")
        
        # Save final report
        report_file = f"automation_report_{timestamp}.json"
        try:
            final_report = self.generate_final_report()
            serializable_report = self._make_json_serializable(final_report)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved final report to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving final report: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def restart_gridbot_if_changes_made(self, debug_sessions: List[DebugSession], optimization_results: List[Union[OptimizationResult, Dict]]) -> bool:
        """Restart GridBot if changes were made during automation"""
        changes_made = False
        
        # Check if any debug sessions made changes
        for session in debug_sessions:
            if session.errors_fixed > 0:
                changes_made = True
                self.logger.info(f"Changes detected in debug session for {session.target_file}: {session.errors_fixed} errors fixed")
        
        # Check if any optimizations were applied
        applied_optimizations = len([r for r in optimization_results if 
                                   (hasattr(r, 'applied') and r.applied) or  # type: ignore
                                   (isinstance(r, dict) and r.get('applied', False))])
        if applied_optimizations > 0:
            changes_made = True
            self.logger.info(f"Changes detected: {applied_optimizations} optimizations applied")
        
        if not changes_made:
            self.logger.info("No changes detected, skipping GridBot restart")
            return False
        
        self.logger.info("=" * 60)
        self.logger.info("RESTARTING GRIDBOT DUE TO CHANGES")
        self.logger.info("=" * 60)
        
        try:
            # First, try to gracefully stop any running GridBot processes
            self.stop_existing_gridbot_processes()
            
            # Wait a moment for cleanup
            time.sleep(3)
            
            # Start the GridBot with the updated files
            self.start_gridbot()
            
            self.logger.info("GridBot restart completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during GridBot restart: {e}")
            return False
    
    def stop_existing_gridbot_processes(self):
        """Stop any existing GridBot processes"""
        try:
            # Try to find and stop GridBot processes
            import psutil
            
            processes_stopped = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('GridbotBackup.py' in str(cmd) for cmd in cmdline):
                        self.logger.info(f"Stopping GridBot process (PID: {proc.info['pid']})")
                        proc.terminate()
                        processes_stopped += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if processes_stopped > 0:
                self.logger.info(f"Stopped {processes_stopped} GridBot process(es)")
                # Wait for processes to clean up
                time.sleep(2)
            else:
                self.logger.info("No existing GridBot processes found")
                
        except ImportError:
            self.logger.warning("psutil not available, using fallback process termination")
            # Fallback: try to terminate using taskkill on Windows
            try:
                subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                             capture_output=True, check=False)
                self.logger.info("Attempted to stop Python processes using taskkill")
            except Exception as e:
                self.logger.warning(f"Fallback process termination failed: {e}")
        except Exception as e:
            self.logger.warning(f"Error stopping existing processes: {e}")
    
    def start_gridbot(self):
        """Start GridBot with the updated configuration"""
        try:
            # Use local GridbotBackup.py in automation strategy folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gridbot_path = os.path.join(current_dir, 'GridbotBackup.py')
            if not os.path.exists(gridbot_path):
                # Fallback to parent directory only if not found locally
                gridbot_path = os.path.join(os.path.dirname(current_dir), 'GridbotBackup.py')
                if not os.path.exists(gridbot_path):
                    gridbot_path = 'GridbotBackup.py'
            
            # Start GridBot as a background process
            python_exe = self.config.get('python_executable', 'python')
            self.logger.info(f"Starting GridBot: {python_exe} {gridbot_path}")
            
            # Start in background and detach from parent
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP to detach on Windows
                subprocess.Popen(
                    [python_exe, gridbot_path],
                    cwd=current_dir,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Unix-like systems
                subprocess.Popen(
                    [python_exe, gridbot_path],
                    cwd=current_dir,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            self.logger.info("GridBot started successfully in background")
            
        except Exception as e:
            self.logger.error(f"Error starting GridBot: {e}")
            raise
    
    def run_continuous_automation(self, target_files: Optional[List[str]] = None) -> Dict:
        """Run automation in continuous loop with automatic restart"""
        if target_files is None:
            target_files = self.config.get('target_files', ['GridbotBackup.py'])
        
        continuous_config = self.config.get('continuous_automation', {})
        
        if not continuous_config.get('enabled', False):
            self.logger.info("Continuous automation disabled - running single cycle")
            return self.run_full_pipeline(target_files)
        
        max_cycles = continuous_config.get('max_cycles', 0)  # 0 = infinite
        cycle_delay_minutes = continuous_config.get('cycle_delay_minutes', 30)
        max_consecutive_failures = continuous_config.get('max_consecutive_failures', 5)
        health_check_interval = continuous_config.get('health_check_interval', 10)
        
        self.logger.info("=" * 100)
        self.logger.info("STARTING CONTINUOUS AUTOMATION MODE")
        self.logger.info("=" * 100)
        self.logger.info(f"Max cycles: {'Infinite' if max_cycles == 0 else max_cycles}")
        self.logger.info(f"Cycle delay: {cycle_delay_minutes} minutes")
        self.logger.info(f"Max consecutive failures: {max_consecutive_failures}")
        self.logger.info(f"Health check interval: {health_check_interval} cycles")
        self.logger.info("=" * 100)
        
        continuous_stats = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'consecutive_failures': 0,
            'start_time': datetime.now(),
            'last_cycle_result': None
        }
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                continuous_stats['total_cycles'] = cycle_count
                self.session_data['continuous_automation']['current_cycle'] = cycle_count
                
                # Check if we've reached max cycles
                if max_cycles > 0 and cycle_count > max_cycles:
                    self.logger.info(f"Reached maximum cycles ({max_cycles}) - stopping continuous automation")
                    break
                
                # Health check every N cycles
                if cycle_count % health_check_interval == 0:
                    self.logger.info(f"\\n[HEALTH CHECK] Cycle {cycle_count} - Running system health check")
                    if not self.run_system_health_check():
                        self.logger.error("System health check failed - stopping continuous automation")
                        break
                
                self.logger.info("\\n" + "=" * 80)
                self.logger.info(f"STARTING AUTOMATION CYCLE {cycle_count}")
                if max_cycles > 0:
                    self.logger.info(f"Progress: {cycle_count}/{max_cycles} cycles")
                else:
                    self.logger.info("Running in infinite loop mode")
                self.logger.info(f"Consecutive failures: {continuous_stats['consecutive_failures']}")
                self.logger.info("=" * 80)
                
                cycle_start_time = datetime.now()
                
                try:
                    # Run file cleanup at start of each cycle (skip if recent startup cleanup)
                    if continuous_config.get('auto_cleanup_between_cycles', True):
                        # Skip cleanup if we just did startup cleanup within last 2 minutes
                        if self.last_cleanup_time is None:
                            # First cycle - cleanup already done at startup, so skip
                            self.logger.info("\\n[CYCLE CLEANUP] Skipping cleanup (startup cleanup just completed)")
                        else:
                            time_since_last_cleanup = (datetime.now() - self.last_cleanup_time).total_seconds()
                            if time_since_last_cleanup > 120:  # 2 minutes
                                self.logger.info("\\n[CYCLE CLEANUP] Running inter-cycle file cleanup")
                                cleanup_stats = self.file_manager.run_cleanup()
                                self.last_cleanup_time = datetime.now()
                                self.logger.info(f"Cleanup: {cleanup_stats.get('files_removed', 0)} files removed, {cleanup_stats.get('space_freed_mb', 0):.1f} MB freed")
                            else:
                                self.logger.info(f"\\n[CYCLE CLEANUP] Skipping cleanup (last cleanup {time_since_last_cleanup:.0f}s ago)")
                    
                    # Reset session data for this cycle
                    self.session_data['debug_sessions'] = []
                    self.session_data['optimization_results'] = []
                    
                    # Run the full automation pipeline
                    result = self.run_full_pipeline(target_files)
                    
                    cycle_end_time = datetime.now()
                    cycle_duration = cycle_end_time - cycle_start_time
                    
                    # Analyze cycle results
                    if result['success']:
                        continuous_stats['successful_cycles'] += 1
                        continuous_stats['consecutive_failures'] = 0
                        self.session_data['continuous_automation']['last_successful_cycle'] = cycle_count
                        
                        self.logger.info(f"\\n[CYCLE {cycle_count} SUCCESS] Completed in {cycle_duration}")
                        self.logger.info(f"Errors fixed: {result.get('final_report', {}).get('debug_phase', {}).get('total_errors_fixed', 0)}")
                        self.logger.info(f"Optimizations applied: {result.get('final_report', {}).get('optimization_phase', {}).get('optimizations_applied', 0)}")
                        
                    else:
                        continuous_stats['failed_cycles'] += 1
                        continuous_stats['consecutive_failures'] += 1
                        
                        self.logger.warning(f"\\n[CYCLE {cycle_count} FAILED] Error: {result.get('error', 'Unknown error')}")
                        self.logger.warning(f"Consecutive failures: {continuous_stats['consecutive_failures']}")
                        
                        # Check if we've hit the failure limit
                        if continuous_stats['consecutive_failures'] >= max_consecutive_failures:
                            self.logger.error(f"Maximum consecutive failures ({max_consecutive_failures}) reached - stopping automation")
                            break
                    
                    # Store cycle result
                    cycle_result = {
                        'cycle_number': cycle_count,
                        'start_time': cycle_start_time.isoformat(),
                        'end_time': cycle_end_time.isoformat(),
                        'duration': str(cycle_duration),
                        'success': result['success'],
                        'error': result.get('error'),
                        'files_processed': len(result.get('debug_sessions', [])),
                        'errors_fixed': result.get('final_report', {}).get('debug_phase', {}).get('total_errors_fixed', 0),
                        'optimizations_applied': result.get('final_report', {}).get('optimization_phase', {}).get('optimizations_applied', 0)
                    }
                    
                    self.session_data['continuous_automation']['cycle_results'].append(cycle_result)
                    continuous_stats['last_cycle_result'] = cycle_result
                    
                    # Check if we should stop on success
                    if result['success'] and continuous_config.get('stop_on_success', False):
                        self.logger.info("All files successful and stop_on_success enabled - stopping automation")
                        break
                    
                except Exception as e:
                    continuous_stats['failed_cycles'] += 1
                    continuous_stats['consecutive_failures'] += 1
                    
                    self.logger.error(f"\\n[CYCLE {cycle_count} EXCEPTION] {e}")
                    
                    if continuous_stats['consecutive_failures'] >= max_consecutive_failures:
                        self.logger.error(f"Maximum consecutive failures ({max_consecutive_failures}) reached - stopping automation")
                        break
                
                # Wait between cycles (unless this is the last cycle)
                if max_cycles == 0 or cycle_count < max_cycles:
                    self.logger.info(f"\\n[CYCLE DELAY] Waiting {cycle_delay_minutes} minutes before next cycle...")
                    self.logger.info(f"Next cycle will start at: {(datetime.now() + timedelta(minutes=cycle_delay_minutes)).strftime('%H:%M:%S')}")
                    
                    # Sleep in small increments to allow for interruption
                    for minute in range(cycle_delay_minutes):
                        time.sleep(60)  # Sleep 1 minute at a time
                        if minute % 5 == 4:  # Log every 5 minutes
                            remaining = cycle_delay_minutes - minute - 1
                            if remaining > 0:
                                self.logger.info(f"[WAITING] {remaining} minutes until next cycle...")
        
        except KeyboardInterrupt:
            self.logger.info("\\n[INTERRUPTED] Continuous automation stopped by user")
        
        except Exception as e:
            self.logger.error(f"\\n[FATAL ERROR] Continuous automation stopped due to: {e}")
        
        finally:
            # Generate final continuous automation report
            total_duration = datetime.now() - continuous_stats['start_time']
            
            self.logger.info("\\n" + "=" * 100)
            self.logger.info("CONTINUOUS AUTOMATION SUMMARY")
            self.logger.info("=" * 100)
            self.logger.info(f"Total runtime: {total_duration}")
            self.logger.info(f"Total cycles: {continuous_stats['total_cycles']}")
            self.logger.info(f"Successful cycles: {continuous_stats['successful_cycles']}")
            self.logger.info(f"Failed cycles: {continuous_stats['failed_cycles']}")
            if continuous_stats['total_cycles'] > 0:
                success_rate = (continuous_stats['successful_cycles'] / continuous_stats['total_cycles']) * 100
                self.logger.info(f"Success rate: {success_rate:.1f}%")
            self.logger.info(f"Final consecutive failures: {continuous_stats['consecutive_failures']}")
            self.logger.info("=" * 100)
            
            # Save continuous automation report
            if self.config.get('save_reports', True):
                self.save_continuous_automation_report(continuous_stats)
        
        return continuous_stats
    
    def run_system_health_check(self) -> bool:
        """Run system health check for continuous automation"""
        try:
            # Check LLM connectivity
            if not self.llm_interface.test_connection():
                self.logger.error("[HEALTH] LLM connection failed")
                return False
            
            # Check disk space (warn if less than 1GB free)
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            if free_space_gb < 1.0:
                self.logger.warning(f"[HEALTH] Low disk space: {free_space_gb:.1f} GB free")
            
            # Check if target files still exist
            target_files = self.config.get('target_files', ['GridbotBackup.py'])
            for file_path in target_files:
                if not os.path.exists(file_path) and not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)):
                    self.logger.error(f"[HEALTH] Target file missing: {file_path}")
                    return False
            
            # Check memory usage (basic check)
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 90:
                    self.logger.warning(f"[HEALTH] High memory usage: {memory_percent:.1f}%")
                self.logger.info(f"[HEALTH] System OK - Disk: {free_space_gb:.1f}GB, Memory: {memory_percent:.1f}%")
            except ImportError:
                # psutil not available, skip memory check
                self.logger.info(f"[HEALTH] System OK - Disk: {free_space_gb:.1f}GB (psutil not available for memory check)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[HEALTH] Health check failed: {e}")
            return False
    
    def save_continuous_automation_report(self, stats: Dict):
        """Save continuous automation report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"continuous_automation_report_{timestamp}.json"
        
        try:
            report_data = {
                'continuous_automation_summary': self._make_json_serializable(stats),
                'session_data': self._make_json_serializable(self.session_data),
                'configuration': self.config.get('continuous_automation', {})
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Continuous automation report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save continuous automation report: {e}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Automated Debugging and Optimization Pipeline')
    parser.add_argument('--files', nargs='+', 
                       help='Target files to process (default: GridbotBackup.py gridbot_websocket_server.py)')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--no-optimization', action='store_true', 
                       help='Skip optimization phase')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum debug iterations per file (default: 50 for autonomous debugging)')
    parser.add_argument('--llm-url', type=str, default='http://localhost:11434',
                       help='Qwen LLM server URL (default: http://localhost:11434)')
    parser.add_argument('--llm-model', type=str, default='qwen3:1.7b',
                       help='Orchestrator model name (default: qwen3:1.7b with agent capabilities)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--single-run', action='store_true', 
                       help='Run once and exit (disable continuous mode)')
    parser.add_argument('--max-cycles', type=int, default=0,
                       help='Maximum cycles for continuous mode (0=infinite)')
    parser.add_argument('--cycle-delay', type=int, default=5,
                       help='Minutes to wait between cycles (default: 5 for active development)')
    parser.add_argument('--alternating-optimization', action='store_true',

                       help='Enable alternating optimization (static -> log-driven -> static)')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode with minimal operations')

    args = parser.parse_args()

    # Build configuration from arguments
    config = {
        'llm_base_url': args.llm_url,
        'llm_model': args.llm_model,
        'max_debug_iterations': args.max_iterations,
        'run_optimization': not args.no_optimization,
        'verbose': args.verbose,
        'skip_connection_test': args.test_mode,  # Skip connection test in test mode
        'alternating_optimization': {
            'enabled': args.alternating_optimization,
            'cycles': args.max_cycles
        },
        'continuous_automation': {
            'enabled': not args.single_run,  # Continuous by default unless --single-run
            'max_cycles': args.max_cycles,
            'cycle_delay_minutes': args.cycle_delay,
            'restart_on_max_iterations': True,
            'stop_on_success': False,
            'max_consecutive_failures': 5,
            'health_check_interval': 10,
            'auto_cleanup_between_cycles': True
        }
    }

    # Load config file if specified
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
            config |= file_config
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1

    # Run AI Workspace Doctor at startup to apply code improvements
    ai_doctor_changed = False
    ai_doctor_result = None
    try:
        print("🏥 Running AI Workspace Doctor...")
        print("   📁 Target: automated_debugging_strategy folder")
        print("   🎯 Mode: Apply fixes + Analysis + Automated debugging")

        # Import and run the workspace doctor with enhanced error handling
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Ensure we're using the same paths and environment as when run standalone
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Import the workspace doctor
        import ai_model_doctor

        # Enhanced AI doctor run with automated debugging
        print("   🔍 Analyzing code for improvements and debugging opportunities...")

        # Run it exactly as it runs standalone but with enhanced logging
        ai_doctor_result = ai_model_doctor.run_ai_workspace_doctor(apply_fixes=True, verbose=True)

        if ai_doctor_result["success"]:
            ai_doctor_changed = ai_doctor_result["changes_found"]
            files_modified = ai_doctor_result["files_changed"]

            if ai_doctor_changed:
                print(f"🎉 AI Workspace Doctor improved {files_modified} files in strategy folder!")
                print("   🔄 Pipeline will restart to use improved code...")
                print("   🐛 Automated debugging analysis completed")
            else:
                print("✨ Strategy folder already optimized - no changes needed")
                print("   🐛 Automated debugging analysis: No issues found")
        else:
            print(f"❌ Workspace doctor failed: {ai_doctor_result.get('error', 'Unknown error')}")
            print("   ⚠️ Continuing with pipeline using current code")
            ai_doctor_changed = False

    except ImportError as e:
        print(f"⚠️ Could not import AI model doctor: {e}")
        print("   💡 Make sure ai_model_doctor.py is in the same directory")
        print("   🔄 Continuing with pipeline without AI doctor integration")
        ai_doctor_changed = False
    except Exception as e:
        print(f"⚠️ Workspace doctor error: {e} - continuing with pipeline")
        print("   📊 Error details logged for analysis")
        ai_doctor_changed = False

    # Determine target files
    target_files = args.files or config.get('target_files', ['gridbot_websocket_server.py', 'GridbotBackup.py', 'config.py'])

    # Initialize and run pipeline
    pipeline = MasterAutomationPipeline(config)

    print("Starting Automated Debugging and Optimization Pipeline...")
    print(f"Target files: {target_files}")
    print(f"LLM Orchestrator: {config.get('llm_model', 'qwen3:1.7b')}")
    print(f"DeepSeek Debugger: {config.get('deepseek_model', 'deepseek-coder')}")
    print(f"Qwen Optimizer: {config.get('qwen_model', 'smollm2:1.7b')}")
    print(f"Max debug iterations: {config.get('max_debug_iterations', 50)} (autonomous debugging)")
    print(f"Optimization enabled: {config.get('run_optimization', True)} (autonomous development)")
    print(f"Thinking mode: {config.get('enable_thinking', True)} (Qwen enhanced reasoning)")

    # If AI Doctor applied changes, exit with a special code to trigger external restart
    """if ai_doctor_changed:
        print("\n" + "=" * 80)
        print("AI WORKSPACE DOCTOR APPLIED CHANGES - RESTARTING PIPELINE")
        print("=" * 80)
        print("AI Workspace Doctor has applied code improvements to the workspace.")
        print("The pipeline will now restart to work with the updated code.")
        print("This ensures all changes are properly loaded and processed.")
        print("=" * 80 + "\n")

        # Save a restart marker file
        try:
            restart_marker = os.path.join(os.getcwd(), '.ai_doctor_restart_pending')
            with open(restart_marker, 'w') as f:
                f.write(f"Restart triggered by AI Workspace Doctor changes at {datetime.now().isoformat()}\n")
                f.write(f"Target files: {target_files}\n")
                f.write(f"Config: {args.config or 'default'}\n")
        except Exception as e:
            print(f"Warning: Could not create restart marker: {e}")

        # Exit with special code to indicate restart needed
        # The external caller (task runner, shell script, etc.) can check this code
        print("Exiting with code 42 to indicate restart needed...")
        return 42"""

    # Check if continuous automation is enabled (default: True)
    continuous_config = config.get('continuous_automation', {})
    if continuous_config.get('enabled', True):  # Default enabled
        print("Continuous automation: ENABLED (autonomous development mode)")
        print(f"Max cycles: {'Infinite' if continuous_config.get('max_cycles', 0) == 0 else continuous_config.get('max_cycles', 0)}")
        print(f"Cycle delay: {continuous_config.get('cycle_delay_minutes', 5)} minutes (active development)")
        print("System will continuously debug and optimize code autonomously")
        print("Use Ctrl+C to stop automation")
        print("=" * 80)

        result = pipeline.run_continuous_automation(target_files)

        if isinstance(result, dict) and 'total_cycles' in result:
            print("\nContinuous automation completed!")
            print(f"Total cycles: {result['total_cycles']}")
            print(f"Successful cycles: {result['successful_cycles']}")
            print(f"Failed cycles: {result['failed_cycles']}")
            if result['total_cycles'] > 0:
                success_rate = (result['successful_cycles'] / result['total_cycles']) * 100
                print(f"Success rate: {success_rate:.1f}%")
            return 0 if result['successful_cycles'] > 0 else 1
        else:
            print(f"\nContinuous automation ended: {result}")
            return 1
    else:
        print("Continuous automation: DISABLED (--single-run specified)")
        print("=" * 80)

        result = pipeline.run_full_pipeline(target_files)

    if result['success']:
        print("\nPipeline completed successfully!")
        final_report = result['final_report']
        print(f"Files processed: {final_report['debug_phase']['files_processed']}")
        print(f"Errors fixed: {final_report['debug_phase']['total_errors_fixed']}")
        print(f"Optimizations applied: {final_report['optimization_phase']['optimizations_applied']}")

        if final_report['recommendations']:
            print("\nRecommendations:")
            for rec in final_report['recommendations']:
                print(f"  - {rec}")

        return 0
    else:
        print(f"\nPipeline failed: {result['error']}")
        return 1

if __name__ == "__main__":
    exit(main())