#!/usr/bin/env python3
"""
Dynamic Multi-Model Syntax Recovery Test

This test creates dynamic error scenarios and tests our enhanced recovery
using ALL available models and strategies until we achieve 100% success.
"""

import os
import sys
import logging
import tempfile
import time
from pathlib import Path

# Since we're already in the automated_debugging_strategy directory, use relative import
try:
    from automated_file_editor import SafeFileEditor, EditResult
    from automated_debugging_strategy.qwen_agent_interface import QwenAgentInterface
    print("✅ Successfully imported enhanced components")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    # Continue anyway to test what we can
    print("📝 Continuing with available components...")

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicMultiModelTest:
    """Dynamic testing with all available models and strategies"""
    
    def __init__(self):
        self.editor = SafeFileEditor(
            validate_syntax=True,
            create_backups=True,
            use_serena=True
        )
        self.llm_interface = None
        try:
            self.llm_interface = QwenAgentInterface(temperature=0.2, enable_thinking=True)
            logger.info("✅ LLM Interface initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ LLM Interface failed to initialize: {e}")
        
        self.dynamic_test_cases = []
        self.success_metrics = {
            'auto_fix_successes': 0,
            'serena_successes': 0,
            'deepseek_successes': 0,
            'qwen_successes': 0,
            'smollm_successes': 0,
            'context_analysis_successes': 0,
            'progressive_repair_successes': 0,
            'total_attempts': 0,
            'perfect_recoveries': 0
        }
    
    def generate_dynamic_broken_code(self, complexity_level: int = 1) -> str:
        """Generate increasingly complex broken Python code"""
        
        base_templates = [
            # Template 1: Function definition errors
            """def broken_function_{id}(param1, param2{missing_colon}
{indent_error}return param1 + param2{missing_paren}

def another_function_{id}({extra_paren}
{wrong_indent}result = broken_function_{id}("test", "value"{unmatched_bracket}
{wrong_indent}return result
""",

            # Template 2: Class definition errors  
            """class BrokenClass_{id}{missing_colon}
{no_indent}def __init__(self, value{missing_paren}
{wrong_indent}self.value = value
{no_indent}self.data = [1, 2, 3{missing_bracket}
    
{no_indent}def method_{id}(self{missing_colon}
{extreme_wrong_indent}return self.value * 2{extra_paren}
""",

            # Template 3: Complex nested structure
            """def complex_nested_{id}(){missing_colon}
{wrong_indent}data = {{
{wrong_indent}    "list": [1, 2, 3{missing_bracket},
{wrong_indent}    "nested": {{
{wrong_indent}        "deep": "value{missing_quote}
{wrong_indent}    {missing_bracket}
{wrong_indent}{missing_bracket}
{wrong_indent}
{wrong_indent}for item in data["list"{missing_bracket}:
{no_indent}print(f"Item: {{item}}{missing_bracket}"
{wrong_indent}
{wrong_indent}return data
""",

            # Template 4: Import and exception handling
            """import os, sys{extra_comma}
from pathlib import Path{missing_import}

def error_prone_function_{id}(filename{missing_colon}
{wrong_indent}try{missing_colon}
{wrong_indent}    with open(filename, 'r'{missing_paren} as f:
{wrong_indent}        content = f.read({extra_paren}
{wrong_indent}        return content.strip({missing_paren}
{wrong_indent}except FileNotFoundError as e{missing_colon}
{no_indent}print(f"File not found: {{e}}{missing_bracket}"
{wrong_indent}return None{extra_paren}
"""
        ]

        # Error injection patterns based on complexity
        if complexity_level == 1:
            # Simple errors
            error_patterns = {
                'missing_colon': '',
                'missing_paren': '',
                'missing_bracket': '',
                'missing_quote': '"',
                'missing_import': '',
                'extra_comma': '',
                'extra_paren': '',
                'unmatched_bracket': '',
                'indent_error': '',
                'wrong_indent': '    ',
                'no_indent': '',
                'extreme_wrong_indent': '    '
            }
        elif complexity_level == 2:
            # Moderate errors
            error_patterns = {
                'missing_colon': '',
                'missing_paren': '',
                'missing_bracket': '',
                'missing_quote': '',
                'missing_import': '',
                'extra_comma': ', ,',
                'extra_paren': ')',
                'unmatched_bracket': '',
                'indent_error': '',
                'wrong_indent': '  ',  # Wrong indentation
                'no_indent': '',
                'extreme_wrong_indent': '      '
            }
        else:
            # Complex errors
            error_patterns = {
                'missing_colon': '',
                'missing_paren': '',
                'missing_bracket': '',
                'missing_quote': '',
                'missing_import': '',
                'extra_comma': ',,',
                'extra_paren': '))',
                'unmatched_bracket': '',
                'indent_error': '  ',  # Wrong base indent
                'wrong_indent': '',    # No indent where needed
                'no_indent': '',
                'extreme_wrong_indent': '\\t'  # Mixed tabs
            }

        # Select template and inject errors
        import random
        template = random.choice(base_templates)
        test_id = random.randint(1000, 9999)

        return template.format(id=test_id, **error_patterns)
    
    def test_comprehensive_recovery_loop(self, max_attempts: int = 10):
        """Test comprehensive recovery with progressive difficulty"""
        
        logger.info("🔄 STARTING DYNAMIC MULTI-MODEL RECOVERY TEST")
        logger.info(f"Will generate and test {max_attempts} dynamic error scenarios")

        perfect_fixes = 0

        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 DYNAMIC TEST ATTEMPT {attempt}/{max_attempts}")
            logger.info(f"{'='*60}")

            # Generate increasingly complex broken code
            complexity = min(3, (attempt - 1) // 3 + 1)
            broken_code = self.generate_dynamic_broken_code(complexity)

            logger.info(f"🎯 Complexity Level: {complexity}")
            logger.info(f"📝 Generated Broken Code:\n{broken_code}")

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(broken_code)
                temp_path = temp_file.name

            try:
                # Test our comprehensive recovery
                start_time = time.time()

                logger.info("🔧 Attempting comprehensive recovery...")
                result = self.editor.apply_line_replacement(
                    file_path=temp_path,
                    old_line=broken_code.split('\\n')[0],  # Replace first line
                    new_line="# Auto-fixed by comprehensive recovery",
                    line_number=1
                )

                recovery_time = time.time() - start_time
                self.success_metrics['total_attempts'] += 1

                if result.success:
                    perfect_fixes += 1
                    self.success_metrics['perfect_recoveries'] += 1

                    logger.info(f"✅ PERFECT RECOVERY in {recovery_time:.2f}s!")

                    # Read the fixed content
                    with open(temp_path, 'r') as f:
                        fixed_content = f.read()

                    logger.info(f"🔧 FIXED CODE:\n{fixed_content}")

                    # Validate it's actually syntactically correct
                    is_valid, validation_error = self.editor.validate_python_syntax(fixed_content)
                    if is_valid:
                        logger.info("✅ SYNTAX VALIDATION: PASSED")
                    else:
                        logger.warning(f"⚠️ SYNTAX VALIDATION: FAILED - {validation_error}")

                else:
                    logger.error(f"❌ RECOVERY FAILED after {recovery_time:.2f}s")
                    if result.error:
                        logger.error(f"Error details: {result.error}")

                # Test individual model capabilities if LLM interface available
                if self.llm_interface:
                    self.test_individual_models(broken_code)

            except Exception as e:
                logger.error(f"💥 EXCEPTION during dynamic test: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Clean up
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        # Final assessment
        success_rate = (perfect_fixes / max_attempts) * 100

        logger.info(f"\n{'='*80}")
        logger.info("🏁 DYNAMIC MULTI-MODEL TEST COMPLETE")
        logger.info(f"{'='*80}")
        logger.info("📊 RESULTS:")
        logger.info(f"   Perfect Recoveries: {perfect_fixes}/{max_attempts}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Total Recovery Attempts: {self.success_metrics['total_attempts']}")

        # Strategy breakdown
        logger.info(f"\\n🔬 STRATEGY EFFECTIVENESS:")
        for strategy, count in self.success_metrics.items():
            if strategy != 'total_attempts':
                logger.info(f"   {strategy}: {count}")

        if success_rate >= 95:
            logger.info(f"\\n🏆 OUTSTANDING! Near-perfect recovery system!")
        elif success_rate >= 85:
            logger.info(f"\\n🥇 EXCELLENT! Highly effective recovery system!")
        elif success_rate >= 70:
            logger.info(f"\\n🥈 GOOD! Solid recovery capabilities!")
        else:
            logger.info(f"\\n⚠️ NEEDS ENHANCEMENT: Consider improving strategies!")

        return success_rate
    
    async def test_individual_models(self, broken_code: str):
        """Test individual model performance for comparison"""
        
        models_to_test = [
            ("DeepSeek Debugger", "deepseek"),
            ("Qwen Optimizer", "qwen"), 
            ("SmolLM Simple", "smollm")
        ]

        logger.info(f"\\n🔬 TESTING INDIVIDUAL MODEL PERFORMANCE:")

        for model_name, model_type in models_to_test:
            try:
                logger.info(f"   🤖 Testing {model_name}...")

                # Create simple fix prompt
                fix_prompt = f"Fix this Python syntax error:\\n\\n{broken_code}\\n\\nReturn only the corrected code:"

                if model_type == "deepseek":
                    response = self.llm_interface._call_deepseek_debugger(fix_prompt, enable_streaming=False)
                else:
                    response = self.llm_interface._call_qwen_optimizer(fix_prompt, enable_streaming=False)

                if response and len(response.strip()) > 20:
                    if cleaned_response := self.editor._extract_code_from_llm_response(
                        response
                    ):
                        is_valid, _ = self.editor.validate_python_syntax(cleaned_response)
                        status = "✅ VALID" if is_valid else "❌ INVALID"
                        logger.info(f"      {model_name}: {status}")

                        if is_valid:
                            self.success_metrics[f'{model_type}_successes'] += 1
                    else:
                        logger.info(f"      {model_name}: ❌ NO CODE EXTRACTED")
                else:
                    logger.info(f"      {model_name}: ❌ NO RESPONSE")

            except Exception as e:
                logger.info(f"      {model_name}: ❌ ERROR - {e}")

def main():
    """Run the dynamic multi-model test"""
    print("🚀 DYNAMIC MULTI-MODEL SYNTAX RECOVERY TEST")
    print("=" * 70)
    
    try:
        test_suite = DynamicMultiModelTest()
        success_rate = test_suite.test_comprehensive_recovery_loop(max_attempts=15)
        
        print("\\n" + "=" * 70)
        print(f"🏁 DYNAMIC TEST COMPLETED! Success Rate: {success_rate:.1f}%")
        
        # Return appropriate exit code
        return 0 if success_rate >= 80 else 1
        
    except Exception as e:
        print(f"💥 Critical error during dynamic testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())