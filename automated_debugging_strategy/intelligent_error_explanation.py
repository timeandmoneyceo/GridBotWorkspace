"""
Intelligent Error Explanation System

This module enhances the existing debug orchestrator with AI-powered error explanations
and intelligent fix suggestions integrated with VS Code's intelligent apps capabilities.
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

class IntelligentErrorExplainer:
    """
    AI-powered error explanation system that integrates with the debug orchestrator
    """
    
    def __init__(self, llm_interface=None, ai_debugging_assistant=None):
        self.llm_interface = llm_interface
        self.ai_debugging_assistant = ai_debugging_assistant
        self.logger = logging.getLogger(__name__)
        
        # Error explanation cache to avoid re-analyzing identical errors
        self.explanation_cache = {}
        
        # Performance tracking
        self.explanation_stats = {
            'total_explanations': 0,
            'cache_hits': 0,
            'successful_fixes': 0,
            'avg_confidence': 0.0
        }
        
    def explain_error_intelligently(self, error_info: Dict, code_context: str = "") -> Dict:
        """
        Provide comprehensive AI-powered error explanation
        
        Args:
            error_info: Structured error information
            code_context: Surrounding code context
            
        Returns:
            Dict with intelligent explanation and fix suggestions
        """
        self.explanation_stats['total_explanations'] += 1
        
        # Create cache key for this error
        cache_key = self._create_cache_key(error_info, code_context)
        
        # Check cache first
        if cache_key in self.explanation_cache:
            self.explanation_stats['cache_hits'] += 1
            cached_result = self.explanation_cache[cache_key].copy()
            cached_result['from_cache'] = True
            return cached_result
        
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'error_summary': self._create_error_summary(error_info),
            'ai_analysis': {},
            'intelligent_suggestions': [],
            'vs_code_actions': [],
            'learning_resources': [],
            'confidence_score': 0.0,
            'fix_complexity': 'unknown',
            'from_cache': False
        }
        
        try:
            # Use AI debugging assistant if available
            if self.ai_debugging_assistant:
                ai_analysis = self.ai_debugging_assistant.analyze_error(error_info)
                explanation['ai_analysis'] = ai_analysis
                explanation['confidence_score'] = ai_analysis.get('confidence_score', 0.0)
                explanation['fix_complexity'] = self._determine_fix_complexity(ai_analysis)
                
            # Generate intelligent suggestions
            explanation['intelligent_suggestions'] = self._generate_intelligent_suggestions(
                error_info, code_context, explanation['ai_analysis']
            )
            
            # Create VS Code specific actions
            explanation['vs_code_actions'] = self._create_vscode_actions(
                error_info, explanation['ai_analysis']
            )
            
            # Add learning resources
            explanation['learning_resources'] = self._suggest_learning_resources(
                error_info['type'], explanation['ai_analysis']
            )
            
            # Update average confidence
            if explanation['confidence_score'] > 0:
                self.explanation_stats['avg_confidence'] = (
                    (self.explanation_stats['avg_confidence'] * (self.explanation_stats['total_explanations'] - 1) + 
                     explanation['confidence_score']) / self.explanation_stats['total_explanations']
                )
            
            # Cache the result
            self.explanation_cache[cache_key] = explanation.copy()
            
            self.logger.info(f"[AI-EXPLAIN] Generated explanation for {error_info['type']} (confidence: {explanation['confidence_score']:.2%})")
            
        except Exception as e:
            self.logger.error(f"[AI-EXPLAIN] Error generating explanation: {e}")
            explanation['ai_analysis'] = {'error': str(e)}
            explanation['intelligent_suggestions'] = [
                {
                    'type': 'fallback',
                    'description': 'AI explanation failed - using basic error analysis',
                    'action': 'manual_debugging'
                }
            ]
            
        return explanation
        
    def _create_cache_key(self, error_info: Dict, code_context: str) -> str:
        """Create a cache key for error explanation"""
        key_components = [
            error_info.get('type', 'Unknown'),
            error_info.get('message', '')[:100],  # First 100 chars of message
            str(hash(code_context[:500]))  # Hash of first 500 chars of context
        ]
        return '|'.join(key_components)
        
    def _create_error_summary(self, error_info: Dict) -> Dict:
        """Create a human-readable error summary"""
        return {
            'error_type': error_info.get('type', 'Unknown'),
            'error_message': error_info.get('message', ''),
            'file_location': f"{error_info.get('file', 'Unknown')}:{error_info.get('line', 0)}",
            'severity': self._determine_severity(error_info),
            'category': self._categorize_error(error_info['type'])
        }
        
    def _determine_severity(self, error_info: Dict) -> str:
        """Determine error severity"""
        error_type = error_info.get('type', '')
        
        critical_errors = ['SyntaxError', 'IndentationError', 'ImportError']
        major_errors = ['NameError', 'AttributeError', 'TypeError']
        minor_errors = ['ValueError', 'KeyError', 'IndexError']
        
        if error_type in critical_errors:
            return 'critical'
        elif error_type in major_errors:
            return 'major'
        elif error_type in minor_errors:
            return 'minor'
        else:
            return 'unknown'
            
    def _categorize_error(self, error_type: str) -> str:
        """Categorize error for better organization"""
        categories = {
            'syntax': ['SyntaxError', 'IndentationError'],
            'runtime': ['NameError', 'AttributeError', 'TypeError', 'ValueError'],
            'import': ['ImportError', 'ModuleNotFoundError'],
            'data': ['KeyError', 'IndexError', 'ValueError'],
            'logic': ['ZeroDivisionError', 'AssertionError']
        }

        return next(
            (
                category
                for category, error_types in categories.items()
                if error_type in error_types
            ),
            'other',
        )
        
    def _determine_fix_complexity(self, ai_analysis: Dict) -> str:
        """Determine complexity of fix based on AI analysis"""
        confidence = ai_analysis.get('confidence_score', 0.0)
        fix_suggestions = ai_analysis.get('fix_suggestions', [])

        if confidence > 0.8 and len(fix_suggestions) > 0:
            # Check if any suggestions are auto-fixable
            auto_fixable = any(sug.get('auto_fixable', False) for sug in fix_suggestions)
            return 'simple' if auto_fixable else 'moderate'
        elif confidence > 0.5:
            return 'moderate'
        else:
            return 'complex'
            
    def _generate_intelligent_suggestions(self, error_info: Dict, code_context: str, ai_analysis: Dict) -> List[Dict]:
        """Generate intelligent fix suggestions"""
        # Use AI analysis suggestions if available
        ai_suggestions = ai_analysis.get('fix_suggestions', [])
        suggestions = [
            {
                'type': 'ai_generated',
                'description': ai_sug.get('description', ''),
                'priority': ai_sug.get('priority', 'medium'),
                'auto_fixable': ai_sug.get('auto_fixable', False),
                'code_example': ai_sug.get('suggested_code', ''),
                'command': ai_sug.get('command', ''),
            }
            for ai_sug in ai_suggestions
        ]
        # Add context-specific suggestions
        error_type = error_info.get('type', '')

        if error_type == 'ImportError':
            suggestions.append({
                'type': 'import_fix',
                'description': 'Use VS Code command palette to install missing packages',
                'priority': 'high',
                'auto_fixable': True,
                'vs_code_action': 'python.analysis.autoImportCompletions'
            })

        elif error_type == 'SyntaxError':
            suggestions.append({
                'type': 'syntax_help',
                'description': 'Use VS Code Python extension syntax highlighting to identify issues',
                'priority': 'high',
                'auto_fixable': False,
                'vs_code_action': 'python.analysis.autoImportCompletions'
            })

        if error_type in ['NameError', 'AttributeError']:
            suggestions.append({
                'type': 'intellisense_help',
                'description': 'Use VS Code IntelliSense to verify available attributes and methods',
                'priority': 'medium',
                'auto_fixable': False,
                'vs_code_action': 'editor.action.triggerSuggest'
            })

        return suggestions
        
    def _create_vscode_actions(self, error_info: Dict, ai_analysis: Dict) -> List[Dict]:
        """Create VS Code specific actions for error fixing"""
        error_type = error_info.get('type', '')
        file_path = error_info.get('file', '')
        line_number = error_info.get('line', 0)

        actions = [
            {
                'action': 'goto_error',
                'label': 'Go to Error Location',
                'command': 'vscode.open',
                'args': [
                    file_path,
                    {
                        'selection': {
                            'start': {'line': line_number - 1, 'character': 0}
                        }
                    },
                ],
                'priority': 'high',
            },
            {
                'action': 'show_problems',
                'label': 'Show Problems Panel',
                'command': 'workbench.actions.view.problems',
                'priority': 'medium',
            },
        ]
        # Error-specific actions
        if error_type == 'ImportError':
            actions.append({
                'action': 'install_package',
                'label': 'Install Missing Package',
                'command': 'python.analysis.autoImportCompletions',
                'priority': 'high'
            })

        if error_type in ['SyntaxError', 'IndentationError']:
            actions.append({
                'action': 'format_document',
                'label': 'Format Document',
                'command': 'editor.action.formatDocument',
                'priority': 'medium'
            })

        if error_type in ['NameError', 'AttributeError']:
            actions.append({
                'action': 'trigger_suggest',
                'label': 'Show IntelliSense Suggestions',
                'command': 'editor.action.triggerSuggest',
                'priority': 'medium'
            })

        actions.extend(
            (
                {
                    'action': 'ai_debug',
                    'label': 'Run AI Debug Analysis',
                    'command': 'gridbot.ai.explainError',
                    'priority': 'high',
                },
                {
                    'action': 'ai_fix',
                    'label': 'Generate AI Fix Suggestions',
                    'command': 'gridbot.ai.generateFix',
                    'priority': 'medium',
                },
            )
        )
        return actions
        
    def _suggest_learning_resources(self, error_type: str, ai_analysis: Dict) -> List[Dict]:
        """Suggest learning resources for better understanding"""
        resources = []
        
        base_resources = {
            'SyntaxError': [
                {
                    'title': 'Python Syntax Fundamentals',
                    'url': 'https://docs.python.org/3/tutorial/introduction.html',
                    'type': 'documentation'
                },
                {
                    'title': 'Common Python Syntax Errors',
                    'url': 'https://realpython.com/invalid-syntax-python/',
                    'type': 'tutorial'
                }
            ],
            'ImportError': [
                {
                    'title': 'Python Import System',
                    'url': 'https://docs.python.org/3/reference/import.html',
                    'type': 'documentation'
                },
                {
                    'title': 'Managing Python Packages with pip',
                    'url': 'https://packaging.python.org/tutorials/installing-packages/',
                    'type': 'tutorial'
                }
            ],
            'NameError': [
                {
                    'title': 'Python Variable Scope',
                    'url': 'https://docs.python.org/3/tutorial/classes.html#scopes-and-namespaces',
                    'type': 'documentation'
                }
            ]
        }
        
        if error_type in base_resources:
            resources.extend(base_resources[error_type])
            
        # Add AI-specific resources
        resources.append({
            'title': 'GridBot AI Debugging Guide',
            'url': 'internal://gridbot/ai-debugging-guide',
            'type': 'internal_guide',
            'description': 'Learn how to use GridBot AI features for debugging'
        })
        
        return resources
        
    def integrate_with_debug_orchestrator(self, debug_orchestrator):
        """
        Integrate with existing debug orchestrator to enhance error handling
        
        Args:
            debug_orchestrator: Instance of DebugAutomationOrchestrator
        """
        try:
            # Enhance the orchestrator's error handling
            original_handle_error = getattr(debug_orchestrator, 'handle_execution_error', None)

            def enhanced_error_handler(error_output: str, file_path: str) -> Dict:
                # Parse error information
                error_info = self._parse_error_output(error_output, file_path)

                # Get AI explanation
                explanation = self.explain_error_intelligently(error_info)

                # Call original handler if it exists
                original_result = {}
                if original_handle_error:
                    original_result = original_handle_error(error_output, file_path)

                # Merge results
                enhanced_result = {
                    'original_analysis': original_result,
                    'ai_explanation': explanation,
                    'combined_suggestions': self._combine_suggestions(
                        original_result.get('suggestions', []),
                        explanation['intelligent_suggestions']
                    ),
                    'confidence_score': explanation['confidence_score'],
                    'vs_code_actions': explanation['vs_code_actions']
                }

                # Log enhanced analysis
                self.logger.info("[AI-ENHANCED] Error analysis completed:")
                self.logger.info(f"  Error: {error_info['type']} at {file_path}:{error_info.get('line', 0)}")
                self.logger.info(f"  AI Confidence: {explanation['confidence_score']:.2%}")
                self.logger.info(f"  Fix Complexity: {explanation['fix_complexity']}")
                self.logger.info(f"  Suggestions: {len(explanation['intelligent_suggestions'])}")

                return enhanced_result

            # Replace the error handler
            debug_orchestrator.handle_execution_error = enhanced_error_handler

            # Add AI explanation method
            debug_orchestrator.explain_error_ai = self.explain_error_intelligently

            self.logger.info("[AI-INTEGRATION] Successfully integrated with debug orchestrator")
            return True

        except Exception as e:
            self.logger.error(f"[AI-INTEGRATION] Failed to integrate with debug orchestrator: {e}")
            return False
            
    def _parse_error_output(self, error_output: str, file_path: str) -> Dict:
        """Parse error output to extract structured information"""
        lines = error_output.strip().split('\n')

        error_info = {
            'type': 'Unknown',
            'message': '',
            'file': file_path,
            'line': 0,
            'context': error_output
        }

        # Find the error type and message (usually the last line)
        for line in reversed(lines):
            if ':' in line and any(keyword in line for keyword in ['Error', 'Exception']):
                try:
                    parts = line.split(':', 1)
                    error_info['type'] = parts[0].strip()
                    error_info['message'] = parts[1].strip() if len(parts) > 1 else ''
                    break
                except Exception:
                    pass

        # Find line number
        for line in lines:
            if 'line' in line.lower() and file_path in line:
                try:
                    # Extract line number using regex or string parsing
                    import re
                    if line_match := re.search(r'line (\d+)', line):
                        error_info['line'] = int(line_match.group(1))
                        break
                except Exception:
                    pass

        return error_info
        
    def _combine_suggestions(self, original_suggestions: List, ai_suggestions: List[Dict]) -> List[Dict]:
        """Combine original and AI suggestions intelligently"""
        combined = [
            {
                'source': 'ai',
                'type': ai_sug.get('type', 'unknown'),
                'description': ai_sug.get('description', ''),
                'priority': ai_sug.get('priority', 'medium'),
                'auto_fixable': ai_sug.get('auto_fixable', False),
            }
            for ai_sug in ai_suggestions
        ]
        # Add original suggestions
        for orig_sug in original_suggestions:
            if isinstance(orig_sug, str):
                combined.append({
                    'source': 'original',
                    'type': 'manual',
                    'description': orig_sug,
                    'priority': 'medium',
                    'auto_fixable': False
                })
            elif isinstance(orig_sug, dict):
                orig_sug['source'] = 'original'
                combined.append(orig_sug)

        return combined
        
    def get_explanation_stats(self) -> Dict:
        """Get statistics about error explanations"""
        return {
            'stats': self.explanation_stats.copy(),
            'cache_size': len(self.explanation_cache),
            'cache_hit_rate': (self.explanation_stats['cache_hits'] / 
                              max(self.explanation_stats['total_explanations'], 1)),
            'avg_confidence': self.explanation_stats['avg_confidence']
        }

def create_intelligent_error_explainer(llm_interface, ai_debugging_assistant) -> IntelligentErrorExplainer:
    """
    Factory function to create an intelligent error explainer
    
    Args:
        llm_interface: LLM interface for error analysis
        ai_debugging_assistant: AI debugging assistant instance
        
    Returns:
        IntelligentErrorExplainer instance
    """
    return IntelligentErrorExplainer(llm_interface, ai_debugging_assistant)

def enhance_debug_orchestrator_with_ai(debug_orchestrator, llm_interface, ai_debugging_assistant) -> bool:
    """
    Helper function to enhance debug orchestrator with AI error explanation
    
    Args:
        debug_orchestrator: Debug orchestrator to enhance
        llm_interface: LLM interface
        ai_debugging_assistant: AI debugging assistant
        
    Returns:
        bool indicating success
    """
    explainer = create_intelligent_error_explainer(llm_interface, ai_debugging_assistant)
    return explainer.integrate_with_debug_orchestrator(debug_orchestrator)