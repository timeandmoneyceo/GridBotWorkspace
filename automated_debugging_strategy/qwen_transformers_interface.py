"""
Qwen Transformers Interface

This module provides an interface to communicate with Qwen3-1.7B model
using transformers library directly instead of Ollama.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
from datetime import datetime
import threading

from .prompt_optimization_utils import PromptOptimizer

@dataclass
class LLMResponse:
    """Container for LLM response data"""
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class QwenTransformersInterface:
    """Interface for communicating with Qwen3-1.7B using transformers directly"""

    def __init__(self,
                 model_name: str = "Qwen/Qwen3-1.7B",
                 max_new_tokens: int = 2048,
                 temperature: float = 0.1,
                 max_prompt_size: int = 8000,
                 enable_prompt_optimization: bool = True,
                 device: str = "auto",
                 torch_dtype: str = "auto"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_prompt_size = max_prompt_size
        self.enable_prompt_optimization = enable_prompt_optimization

        # Device configuration
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Torch dtype configuration
        if torch_dtype == "auto":
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = getattr(torch, torch_dtype)

        # Initialize prompt optimizer
        if enable_prompt_optimization:
            self.prompt_optimizer = PromptOptimizer(
                max_prompt_size=max_prompt_size,
                target_section_size=max_prompt_size // 10
            )
        else:
            self.prompt_optimizer = None

        self.setup_logging()

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

        # Initialize status tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._health_status = 'Unknown'

    def _load_model(self):
        """Load the Qwen model and tokenizer"""
        try:
            self.logger.info(f"Loading Qwen model: {self.model_name}")
            self.logger.info(f"Device: {self.device}, dtype: {self.torch_dtype}")

            start_time = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Move to device if not using device_map
            if self.device != "cuda" or not torch.cuda.is_available():
                self.model.to(self.device)

            load_time = time.time() - start_time
            self.logger.info(".2f")
            self._health_status = 'Model Loaded'

        except Exception as e:
            self.logger.error(f"Failed to load Qwen model: {e}")
            self._health_status = f'Load Error: {type(e).__name__}'
            raise

    def setup_logging(self):
        """Setup comprehensive logging for the Qwen interface"""
        class QwenFormatter(logging.Formatter):
            def format(self, record):
                symbol_map = {
                    'DEBUG': '[DEBUG]',
                    'INFO': '[INFO] ',
                    'WARNING': '[WARN] ',
                    'ERROR': '[ERROR]',
                    'CRITICAL': '[CRIT] '
                }
                symbol = symbol_map.get(record.levelname, '[LOG]  ')

                timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
                return f"{symbol} [{timestamp}] Qwen: {record.getMessage()}"

        # Setup file logging
        file_handler = logging.FileHandler('qwen_interface.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        # Setup console logging
        console_formatter = QwenFormatter()
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)

        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def test_connection(self) -> bool:
        """Test if the model is loaded and working"""
        try:
            if self.model is None or self.tokenizer is None:
                self.logger.error("Qwen model or tokenizer not loaded")
                self._health_status = 'Not Loaded'
                return False

            # Quick test inference
            test_prompt = "Hello"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            if response := self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            ):
                self._health_status = 'Healthy'
                self.logger.info(f"Qwen model test successful - response: '{response}'")
                return True
            else:
                self._health_status = 'No Response'
                self.logger.error("Qwen model test failed - no response")
                return False

        except Exception as e:
            self._health_status = f'Error: {type(e).__name__}'
            self.logger.error(f"Qwen model test failed: {e}")
            return False

    def _make_request(self, prompt: str, system_prompt: str = None,
                     temperature: float = None, max_tokens: int = None) -> LLMResponse:
        """Make a request to the Qwen model"""
        request_start = time.time()
        self._total_requests += 1

        # Use instance defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_new_tokens

        # Apply prompt optimization if enabled
        original_prompt = prompt
        original_system = system_prompt
        optimization_metadata = {}

        if self.prompt_optimizer and self.enable_prompt_optimization:
            try:
                optimized_prompt, optimized_system, metadata = self.prompt_optimizer.optimize_prompt_for_llm(
                    prompt, system_prompt, context=f"request_{self._total_requests}"
                )
                prompt = optimized_prompt
                system_prompt = optimized_system or system_prompt
                optimization_metadata = metadata

                if metadata.get('optimization_applied'):
                    self.logger.info("PROMPT OPTIMIZATION APPLIED:")
                    self.logger.info(f"   [SIZE] {metadata['original_size']} -> {metadata['final_size']} chars "
                                   f"(reduced by {metadata.get('size_reduction', 0)})")
            except Exception as e:
                self.logger.warning(f"Prompt optimization failed: {e}, using original prompt")
                optimization_metadata = {'optimization_error': str(e)}

        # Log request information
        self.logger.info("=" * 80)
        self.logger.info("QWEN REQUEST STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Temperature: {temperature}")
        self.logger.info(f"Max Tokens: {max_tokens}")

        if optimization_metadata.get('optimization_applied'):
            self.logger.info(f"Prompt Size: {len(prompt)} chars (optimized from {optimization_metadata['original_size']})")
        else:
            self.logger.info(f"Prompt Size: {len(prompt)} chars")

        # Prepare the full prompt with chat template
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            # Apply chat template
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            self.logger.info("PROMPT:")
            self.logger.info("-" * 60)
            if len(full_prompt) > 1000:
                prompt_preview = full_prompt[:500] + "\n\n... [MIDDLE TRUNCATED] ...\n\n" + full_prompt[-300:]
                self.logger.info(prompt_preview)
            else:
                self.logger.info(full_prompt)
            self.logger.info("-" * 60)

            # Tokenize
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            # Generate response
            self.logger.info("Generating response...")
            generate_start = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generate_time = time.time() - generate_start

            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )

            total_time = time.time() - request_start
            self._successful_requests += 1

            # Log successful response
            self.logger.info("=" * 80)
            self.logger.info("QWEN RESPONSE COMPLETED")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Time: {total_time:.2f}s")
            self.logger.info(f"Generation Time: {generate_time:.2f}s")
            self.logger.info(f"Response Length: {len(response_text)} chars")
            self.logger.info(f"Success Rate: {(self._successful_requests / self._total_requests) * 100:.1f}%")

            self.logger.info("FULL RESPONSE:")
            self.logger.info("-" * 80)
            self.logger.info(response_text)
            self.logger.info("-" * 80)

            metadata = {
                'model': self.model_name,
                'device': str(self.device),
                'total_time': total_time,
                'generation_time': generate_time,
                'tokens_generated': len(self.tokenizer.encode(response_text))
            }

            return LLMResponse(
                content=response_text,
                success=True,
                metadata=metadata
            )

        except Exception as e:
            total_time = time.time() - request_start
            error_msg = f"Qwen generation failed after {total_time:.2f}s: {e}"
            self.logger.error(f"QWEN ERROR: {error_msg}")

            return LLMResponse(
                content="",
                success=False,
                error=error_msg
            )

    def generate_error_fix(self, error_info: Dict, source_code: str,
                          file_path: str) -> LLMResponse:
        """Generate a fix for a specific error"""
        error_type = error_info.get('error_type', 'Unknown')

        system_prompt = f"""You are an expert Python debugging assistant. Your task is to analyze the provided error and source code, then generate a precise fix.

IMPORTANT RULES:
1. Provide ONLY the corrected code section, not the entire file
2. Include enough context lines around the fix for accurate replacement
3. Maintain the original indentation and formatting
4. Focus on fixing the specific error mentioned
5. Do not add extra features or optimizations unless necessary for the fix
6. Do NOT create duplicate function definitions or nested functions unless specifically needed
7. Keep response under 200 lines - provide complete fix with context
8. NEVER use numbered lines (1:, 2:, etc.) - provide clean, unnumbered code
9. Do NOT include line numbers or prefixes in your code output

ERROR-SPECIFIC GUIDANCE for {error_type}:
- Analyze the specific error and provide an appropriate fix
- Maintain code functionality while fixing the error
- Include proper error handling where appropriate"""

        prompt = f"""Error Information:
- Type: {error_info.get('error_type', 'Unknown')}
- Message: {error_info.get('error_message', 'No message')}
- File: {file_path}
- Line: {error_info.get('line_number', 'Unknown')}
- Function: {error_info.get('function_name', 'Unknown')}

Source Code Context:
```python
{source_code}
```

Please analyze this error and provide a fix. Focus specifically on line {error_info.get('line_number', 'Unknown')} and the surrounding context."""

        return self._make_request(prompt, system_prompt, temperature=0.1)

    def generate_optimization(self, code_snippet: str, performance_context: str = "") -> LLMResponse:
        """Generate code optimizations"""
        system_prompt = """You are a Python performance optimization expert. Analyze the provided code and suggest optimizations.

FOCUS ON:
1. Performance improvements
2. Memory efficiency
3. Better algorithms or data structures
4. Code readability and maintainability
5. Best practices

CRITICAL RULES:
1. NEVER return the same code unchanged - this is FORBIDDEN
2. If no meaningful optimizations are possible, respond with "NO_OPTIMIZATIONS_NEEDED"
3. Provide the optimized code in the same format as input
4. Include comments explaining the optimizations
5. Maintain functionality - do not change the API
6. Only suggest improvements that provide measurable benefits
7. Keep response under 200 lines - provide comprehensive optimizations

RESPONSE FORMAT:
```python
# Optimized code with key improvements highlighted
[optimized code here - must be different from input]
```

OR (if no optimizations possible):
NO_OPTIMIZATIONS_NEEDED

Explain the optimizations briefly before the code (max 3 sentences)."""

        prompt = f"""Code to optimize:
```python
{code_snippet}
```

Performance context: {performance_context}

Please provide optimized version of this code with explanations.

CRITICAL: If no meaningful optimizations are possible, respond with "NO_OPTIMIZATIONS_NEEDED" only."""

        return self._make_request(prompt, system_prompt, temperature=0.2)

    def generate_targeted_optimization(self, code_section: str, performance_issues: List[str]) -> LLMResponse:
        """Generate optimization for a specific code section"""
        system_prompt = """You are a Python performance optimization expert.

YOUR ONLY JOB: Optimize code for performance issues.

RESPONSE FORMAT:
- If optimizations possible: Return ONLY optimized code in ```python block
- If no optimizations needed: Return ONLY "NO_OPTIMIZATIONS_NEEDED"

NEVER return the same code. NEVER explain. NEVER give examples. Just optimize or say no."""

        issues_text = ", ".join(performance_issues) if performance_issues else "general performance optimization"

        prompt = f"""PERFORMANCE ISSUES TO ADDRESS: {issues_text}

CODE SECTION TO OPTIMIZE:
```python
{code_section}
```

YOUR TASK: Optimize this code to address the performance issues listed above.

RESPONSE RULES:
- If you can make REAL improvements, return ONLY the optimized code in a ```python code block
- If NO improvements are possible, return ONLY the text "NO_OPTIMIZATIONS_NEEDED"
- NEVER return the same code as input
- NEVER provide examples or explanations - just the code or "NO_OPTIMIZATIONS_NEEDED"

Make the code faster, more efficient, or more maintainable while keeping the same functionality."""

        return self._make_request(prompt, system_prompt, temperature=0.2)

    def extract_code_from_response(self, response_content: str) -> str:
        """Extract code blocks from LLM response"""
        self.logger.info("Extracting code from Qwen response...")

        # Look for code blocks marked with ```python or ```
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'```python(.*?)```',
            r'```(.*?)```'
        ]

        for pattern in code_patterns:
            if matches := re.findall(pattern, response_content, re.DOTALL):
                extracted = self._clean_extracted_code(matches[0].strip())
                self.logger.info(f"Extracted code from markdown block: {len(extracted)} chars")
                return extracted

        # If no code blocks found, try to extract lines that look like code
        lines = response_content.split('\n')
        code_lines = []
        in_code_section = False

        for line in lines:
            if (line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'import ', 'from ')) or
                line.strip().endswith(':') or
                ('=' in line and not line.strip().startswith('#'))):
                in_code_section = True

            if in_code_section:
                code_lines.append(line)

                if (
                    line.strip()
                    and not line.startswith((' ', '\t'))
                    and not line.strip().startswith('#')
                    and all(
                        keyword not in line
                        for keyword in [
                            'def ',
                            'class ',
                            'if ',
                            'for ',
                            'while ',
                            'try:',
                            'except:',
                            'finally:',
                            'with ',
                            'import ',
                            'from ',
                        ]
                    )
                ):
                    break

        extracted = self._clean_extracted_code('\n'.join(code_lines).strip()) if code_lines else response_content
        self.logger.info(f"Extracted code from response text: {len(extracted)} chars")
        return extracted

    def extract_optimization_code_from_response(self, response_content: str) -> str:
        """Extract code from optimization responses"""
        self.logger.info(f"[CODE_EXTRACTION] Starting extraction from {len(response_content)} char response")

        # Check for "no optimizations needed" response
        if "NO_OPTIMIZATIONS_NEEDED" in response_content.upper():
            self.logger.info("[CODE_EXTRACTION] Qwen indicated no optimizations needed")
            return ""

        raw_code = self.extract_code_from_response(response_content)

        if not raw_code:
            self.logger.warning("[CODE_EXTRACTION] No code extracted by extract_code_from_response")
            return raw_code

        self.logger.info(f"[CODE_EXTRACTION] Raw code extracted: {len(raw_code)} chars")

        lines = raw_code.split('\n')
        cleaned_lines = []
        skip_imports = True  # Skip imports at the beginning

        for line in lines:
            # Skip comment headers and import lines at the beginning
            if skip_imports:
                if (line.strip().startswith('#') or
                    line.strip().startswith('import ') or
                    line.strip().startswith('from ') or
                    not line.strip()):
                    continue
                else:
                    skip_imports = False

            # Keep the rest of the code
            cleaned_lines.append(line)

        final_code = '\n'.join(cleaned_lines)
        self.logger.info(f"[CODE_EXTRACTION] Final cleaned code: {len(final_code)} chars")

        return final_code

    def _clean_extracted_code(self, code: str) -> str:
        """Clean extracted code by removing line numbers and formatting artifacts"""
        if not code:
            return code

        lines = code.split('\n')
        cleaned_lines = []
        code_lines_with_indent = []

        # First pass: extract all code lines and their indentation
        for line in lines:
            # Remove prefixes like "# Fixed code section", "1:", "2:", etc.
            if line.strip().startswith('# Fixed code section'):
                continue

            if match := re.match(r'^(\s*(?:>>>\s*)?\d+\s*:\s*)(.*)', line):
                prefix, code_part = match.groups()
                if code_part.strip():
                    code_lines_with_indent.append(code_part)
            else:
                if line.strip() and not line.strip().startswith('Explanation:'):
                    code_lines_with_indent.append(line)

        if not code_lines_with_indent:
            return code

        # Find minimum indentation to normalize
        min_indent = float('inf')
        for line in code_lines_with_indent:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        if min_indent == float('inf'):
            min_indent = 0

        # Second pass: normalize indentation
        for line in code_lines_with_indent:
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                new_indent = max(0, current_indent - min_indent)
                normalized_line = ' ' * new_indent + line.lstrip()
                cleaned_lines.append(normalized_line)
            else:
                cleaned_lines.append('')

        return '\n'.join(cleaned_lines)

    def get_status_summary(self) -> Dict[str, Union[str, int, float]]:
        """Get current Qwen interface status summary"""
        success_rate = (self._successful_requests / self._total_requests * 100) if self._total_requests > 0 else 0

        return {
            'health_status': self._health_status,
            'model': self.model_name,
            'device': str(self.device),
            'torch_dtype': str(self.torch_dtype),
            'total_requests': self._total_requests,
            'successful_requests': self._successful_requests,
            'success_rate': round(success_rate, 1)
        }

    def log_status_summary(self):
        """Log a detailed status summary"""
        status = self.get_status_summary()

        self.logger.info("[STATUS] Qwen Status Summary")
        self.logger.info(f"   [MODEL] {status['model']} on {status['device']} ({status['torch_dtype']})")
        self.logger.info(f"   [HEALTH] {status['health_status']}")
        self.logger.info(f"   [STATS] Success Rate: {status['success_rate']}% ({status['successful_requests']}/{status['total_requests']})")