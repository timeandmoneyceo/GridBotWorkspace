"""
Automated Debugging Strategy Package

This package contains all the components for automated debugging and optimization:
- Master automation pipeline
- LLM interfaces (Qwen, DeepSeek)  
- Optimization systems
- File editors with Serena integration
- Debug orchestrators
- Log analyzers
"""

__version__ = "1.0.0"
__author__ = "GridBot Automation System"

# Import main components for easier access
try:
    from .master_automation_pipeline import MasterAutomationPipeline
    from .qwen_agent_interface import QwenAgentInterface
    from .optimization_automation_system import OptimizationAutomationSystem
    from .automated_file_editor import SafeFileEditor
    from .debug_automation_orchestrator import DebugAutomationOrchestrator
    
    __all__ = [
        'MasterAutomationPipeline',
        'QwenAgentInterface', 
        'OptimizationAutomationSystem',
        'SafeFileEditor',
        'DebugAutomationOrchestrator'
    ]
except ImportError as e:
    # Allow package to be imported even if some dependencies are missing
    print(f"Warning: Some components could not be imported: {e}")
    __all__ = []