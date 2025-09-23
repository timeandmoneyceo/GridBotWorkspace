"""
LLM Interface Module

This module provides an interface to communicate with the local smollm2:1.7b model
for generating code diffs and optimizations.
"""

import requests
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
from datetime import datetime
import threading

try:
    # Try relative import first (when run as module)
    from .prompt_optimization_utils import PromptOptimizer
except ImportError:
    # Fall back to absolute import (when run as script)
    from prompt_optimization_utils import PromptOptimizer

@dataclass
class LLMResponse:
    """Container for LLM response data with code extraction capability"""
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    _extracted_code: Optional[str] = None
    
    @property
    def extracted_code(self) -> Optional[str]:
        """Extract code from markdown blocks in the response"""
        if self._extracted_code is None and self.content:
            self._extracted_code = self._extract_code_from_response()
        return self._extracted_code
    
    def _extract_code_from_response(self) -> str:
        """Extract code from markdown blocks or detect Python code patterns"""
        content = self.content
        
        # Method 1: Look for ```python code blocks
        if '```python' in content:
            start = content.find('```python') + 9
            end = content.find('```', start)
            if end != -1:
                extracted = content[start:end].strip()
                return self._clean_extracted_code(extracted)
        
        # Method 2: Look for generic ``` code blocks
        elif '```' in content:
            start = content.find('```') + 3
            end = content.find('```', start)
            if end != -1:
                extracted = content[start:end].strip()
                return self._clean_extracted_code(extracted)
        
        # Method 3: Detect Python-like code patterns
        lines = content.split('\n')
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
        
        if code_lines:
            extracted = '\n'.join(code_lines).strip()
            return self._clean_extracted_code(extracted)
        
        # Method 4: If no clear code patterns found, return original content
        return self._clean_extracted_code(content.strip())
    
    def _clean_extracted_code(self, code: str) -> str:
        """Clean extracted code to fix common issues like unterminated strings"""
        try:
            # Fix unterminated triple-quoted strings
            lines = code.split('\n')
            cleaned_lines = []
            in_triple_quote = False
            triple_quote_type = None
            
            for line in lines:
                # Check for triple quotes
                if '"""' in line or "'''" in line:
                    # Count triple quotes in the line
                    triple_double_count = line.count('"""')
                    triple_single_count = line.count("'''")
                    
                    # Handle triple double quotes
                    if triple_double_count % 2 == 1:  # Odd number means toggle state
                        if not in_triple_quote:
                            in_triple_quote = True
                            triple_quote_type = '"""'
                        elif triple_quote_type == '"""':
                            in_triple_quote = False
                            triple_quote_type = None
                    
                    # Handle triple single quotes
                    if triple_single_count % 2 == 1:  # Odd number means toggle state
                        if not in_triple_quote:
                            in_triple_quote = True
                            triple_quote_type = "'''"
                        elif triple_quote_type == "'''":
                            in_triple_quote = False
                            triple_quote_type = None
                
                cleaned_lines.append(line)
            
            # If we end with an unterminated triple quote, close it
            if in_triple_quote and triple_quote_type:
                cleaned_lines.append(triple_quote_type)
            
            cleaned_code = '\n'.join(cleaned_lines)
            
            # Validate syntax
            try:
                compile(cleaned_code, '<cleaned_code>', 'exec')
                return cleaned_code
            except SyntaxError:
                # If still has syntax errors, return original
                return code
                
        except Exception:
            # If cleaning fails, return original
            return code

class SmollLLMInterface:
    """Interface for communicating with the local smollm2:1.7b and qwen3:1.7b models"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "smollm2:1.7b",
                 optimization_model: str = "qwen3:1.7b",
                 timeout: int = 300,
                 max_retries: int = 3,
                 max_prompt_size: int = 8000,
                 enable_prompt_optimization: bool = True):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name  # Default model for debugging
        self.optimization_model = optimization_model  # Model for optimization operations
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_prompt_size = max_prompt_size
        self.enable_prompt_optimization = enable_prompt_optimization
        
        # Initialize prompt optimizer
        if enable_prompt_optimization:
            self.prompt_optimizer = PromptOptimizer(
                max_prompt_size=max_prompt_size,
                target_section_size=max_prompt_size // 10  # 10% sections
            )
        else:
            self.prompt_optimizer = None
            
        self.setup_logging()
        
        # Test connection on initialization
        self.test_connection()
    
    def _get_model_for_operation(self, operation_type: str = "debugging") -> str:
        """Get the appropriate model for the operation type
        
        Args:
            operation_type: "optimization" for optimization tasks, "debugging" for error fixing
            
        Returns:
            Model name to use for the operation
        """
        if operation_type == "optimization":
            return self.optimization_model
        else:
            return self.model_name
    
    def setup_logging(self):
        """Setup comprehensive logging for the LLM interface with enhanced terminal visibility"""
        # Create custom formatter for better readability
        class LLMFormatter(logging.Formatter):
            def format(self, record):
                # Add ASCII symbols for different log levels (Windows compatible)
                symbol_map = {
                    'DEBUG': '[DEBUG]',
                    'INFO': '[INFO] ',
                    'WARNING': '[WARN] ',
                    'ERROR': '[ERROR]',
                    'CRITICAL': '[CRIT] '
                }
                symbol = symbol_map.get(record.levelname, '[LOG]  ')
                
                # Format timestamp
                timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
                
                # Create structured message
                if 'LLM_STATUS' in str(record.getMessage()):
                    return f"{symbol} [{timestamp}] {record.getMessage()}"
                else:
                    return f"{symbol} [{timestamp}] LLM: {record.getMessage()}"
        
        # Setup file logging with detailed format
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler = logging.FileHandler('llm_interface.log', encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        
        # Setup console logging with enhanced format (only if root has no handler)
        console_formatter = LLMFormatter()
        console_handler = None
        root = logging.getLogger()
        if not root.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.logger.addHandler(file_handler)
        if console_handler is not None:
            self.logger.addHandler(console_handler)
        
        # Allow propagation to root so master pipeline can control console output
        self.logger.propagate = True
        
        # Initialize status tracking
        self._last_health_check = 0
        self._consecutive_failures = 0
        self._total_requests = 0
        self._successful_requests = 0
        self._health_status = 'Unknown'
    
    def test_connection(self) -> bool:
        """Test connection to the LLM server with comprehensive status logging"""
        start_time = time.time()
        self.logger.info(f"LLM_STATUS: Testing connection to {self.base_url}...")

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            connection_time = time.time() - start_time

            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]

                # Check for both models
                has_debug_model = self.model_name in available_models
                has_opt_model = self.optimization_model in available_models

                if has_debug_model and has_opt_model:
                    self._health_status = 'Healthy (Both Models)'
                    self._last_health_check = time.time()
                    self.logger.info(f"LLM_STATUS: [OK] Connected to both models ({connection_time:.2f}s)")
                    self.logger.info(f"LLM_STATUS: [DEBUG] {self.model_name} - {'Available' if has_debug_model else 'Not Found'}")
                    self.logger.info(f"LLM_STATUS: [OPTIMIZE] {self.optimization_model} - {'Available' if has_opt_model else 'Not Found'}")
                    self.logger.info(f"LLM_STATUS: [MODELS] Available: {len(available_models)} [{', '.join(available_models[:3])}{'...' if len(available_models) > 3 else ''}]")
                    return True
                elif has_debug_model:
                    self._health_status = f'Debug Model Only ({self.model_name})'
                    self._last_health_check = time.time()
                    self.logger.warning(
                        "LLM_STATUS: [PARTIAL] Debug model available, optimization model missing"
                    )
                    self.logger.warning(f"LLM_STATUS: [DEBUG] {self.model_name} - Available")
                    self.logger.warning(f"LLM_STATUS: [OPTIMIZE] {self.optimization_model} - Not Found")
                    return True
                elif has_opt_model:
                    self._health_status = f'Optimization Model Only ({self.optimization_model})'
                    self._last_health_check = time.time()
                    self.logger.warning(
                        "LLM_STATUS: [PARTIAL] Optimization model available, debug model missing"
                    )
                    self.logger.warning(f"LLM_STATUS: [DEBUG] {self.model_name} - Not Found")
                    self.logger.warning(f"LLM_STATUS: [OPTIMIZE] {self.optimization_model} - Available")
                    return True
                else:
                    self._health_status = 'No Models Found'
                    self.logger.error("LLM_STATUS: [ERROR] Neither model found")
                    self.logger.error(f"LLM_STATUS: [DEBUG] {self.model_name} - Not Found")
                    self.logger.error(f"LLM_STATUS: [OPTIMIZE] {self.optimization_model} - Not Found")
                    self.logger.error(f"LLM_STATUS: [AVAILABLE] {available_models}")
                    return False
            else:
                self._health_status = f'HTTP {response.status_code}'
                self.logger.error(f"LLM_STATUS: [ERROR] Server error: HTTP {response.status_code} ({connection_time:.2f}s)")
                return False
        except requests.exceptions.ConnectTimeout:
            self._health_status = 'Connection Timeout'
            self.logger.error(f"LLM_STATUS: [TIMEOUT] Connection timeout after {time.time() - start_time:.2f}s")
            return False
        except requests.exceptions.ConnectionError:
            self._health_status = 'Connection Refused'
            self.logger.error(
                "LLM_STATUS: [REFUSED] Connection refused - is Ollama running?"
            )
            return False
        except Exception as e:
            self._health_status = f'Error: {type(e).__name__}'
            connection_time = time.time() - start_time
            self.logger.error(f"LLM_STATUS: [FAIL] Connection test failed: {e} ({connection_time:.2f}s)")
            return False
    
    def _make_request(self, prompt: str, system_prompt: Optional[str] = None, 
                     temperature: float = 0.1, max_tokens: int = -1,
                     operation_type: str = "debugging") -> LLMResponse:
        """Make a streaming request to the LLM with live feedback and activity-based timeout"""
        url = f"{self.base_url}/api/generate"
        request_start = time.time()
        self._total_requests += 1

        # Select the appropriate model for this operation
        selected_model = self._get_model_for_operation(operation_type)

        # Apply prompt optimization if enabled
        original_prompt = prompt
        original_system = system_prompt
        optimization_metadata = {}

        if self.prompt_optimizer and self.enable_prompt_optimization:
            try:
                optimized_prompt, optimized_system, metadata = self.prompt_optimizer.optimize_prompt_for_llm(
                    prompt, system_prompt or "", context=f"request_{self._total_requests}"
                )
                prompt = optimized_prompt
                system_prompt = optimized_system or system_prompt
                optimization_metadata = metadata

                if metadata.get('optimization_applied'):
                    self.logger.info("PROMPT OPTIMIZATION APPLIED:")
                    self.logger.info(f"   [SIZE] {metadata['original_size']} -> {metadata['final_size']} chars "
                                   f"(reduced by {metadata.get('size_reduction', 0)})")
                    if metadata.get('sections_created', 0) > 1:
                        self.logger.info(f"   [SECTIONS] Created {metadata['sections_created']} sections, "
                                       f"selected section {metadata.get('selected_section', 'N/A')} "
                                       f"(priority {metadata.get('selected_priority', 'N/A')})")
                    if metadata.get('truncation_applied'):
                        self.logger.info("   [TRUNCATION] Applied intelligent truncation")

            except Exception as e:
                self.logger.warning(f"   [OPTIMIZER_ERROR] Prompt optimization failed: {e}, using original prompt")
                optimization_metadata = {'optimization_error': str(e)}

        # Log detailed request information
        self.logger.info("=" * 80)
        self.logger.info("LLM REQUEST STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"URL: {url}")
        self.logger.info(f"Model: {selected_model} (operation: {operation_type})")
        self.logger.info(f"Temperature: {temperature}")
        self.logger.info(f"Max Tokens: {max_tokens}")
        self.logger.info(f"Timeout: {self.timeout}s")

        if optimization_metadata.get('optimization_applied'):
            self.logger.info(f"Prompt Size: {len(prompt)} chars (optimized from {optimization_metadata['original_size']})")
        else:
            self.logger.info(f"Prompt Size: {len(prompt)} chars")

        if system_prompt:
            if optimization_metadata.get('system_truncated'):
                self.logger.info(f"System Prompt: {len(system_prompt)} chars (truncated from {optimization_metadata.get('original_system_size', 'unknown')})")
            else:
                self.logger.info(f"System Prompt: {len(system_prompt)} chars")
            # Show truncated system prompt for logging
            system_preview = (
                f"{system_prompt[:200]}..."
                if len(system_prompt) > 200
                else system_prompt
            )
            self.logger.info(f"System Content: {system_preview}")

        self.logger.info("PROMPT:")
        self.logger.info("-" * 60)
        # Apply intelligent logging truncation even for optimized prompts
        if len(prompt) > 1000:
            # For very long prompts, show beginning and end
            prompt_preview = prompt[:500] + "\n\n... [MIDDLE CONTENT TRUNCATED FOR LOGGING] ...\n\n" + prompt[-300:]
            self.logger.info(prompt_preview)
        else:
            self.logger.info(prompt)
        self.logger.info("-" * 60)

        # Prepare the prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        # Log request details (truncated for readability)
        prompt_preview = f"{prompt[:100]}..." if len(prompt) > 100 else prompt
        self.logger.info(f"[REQUEST] #{self._total_requests}: Starting STREAMING LLM request")
        self.logger.info(f"   [PROMPT] {prompt_preview}")
        self.logger.info(f"   [CONFIG] temp={temperature}, max_tokens={max_tokens}, activity_timeout={self.timeout}s")

        payload = {
            "model": selected_model,
            "prompt": full_prompt,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": temperature,
                "num_predict": -1  # Unlimited for local model
            }
        }

        for attempt in range(self.max_retries):
            attempt_start = time.time()
            try:
                self.logger.info(f"   [ATTEMPT] {attempt + 1}/{self.max_retries} - Starting stream to {self.model_name}...")

                # Make streaming request with longer initial timeout for slow model startup
                response = requests.post(url, json=payload, stream=True, timeout=600)  # 10 minute connection timeout

                if response.status_code == 200:
                    # Process streaming response with activity-based timeout
                    return self._process_streaming_response(response, request_start, attempt + 1)

                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    self.logger.error(f"   [FAIL] Request failed: {error_msg} (attempt {attempt + 1})")

                    if attempt == self.max_retries - 1:
                        self._consecutive_failures += 1
                        total_time = time.time() - request_start
                        self.logger.error(f"   [FINAL] FAILURE after {total_time:.2f}s and {self.max_retries} attempts")
                        return LLMResponse(content="", success=False, error=error_msg)

                    backoff_time = 2 ** attempt
                    self.logger.warning(f"   [RETRY] Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)

            except requests.exceptions.Timeout:
                attempt_time = time.time() - attempt_start
                self.logger.error(f"   [TIMEOUT] Connection timeout after {attempt_time:.2f}s (attempt {attempt + 1})")

                if attempt == self.max_retries - 1:
                    self._consecutive_failures += 1
                    total_time = time.time() - request_start
                    error_msg = f"Connection timeout after {total_time:.2f}s"
                    self.logger.error(f"   [FINAL] TIMEOUT after {total_time:.2f}s")
                    return LLMResponse(content="", success=False, error=error_msg)

                backoff_time = 2 ** attempt
                self.logger.warning(f"   [RETRY] Retrying after timeout in {backoff_time}s...")
                time.sleep(backoff_time)

            except requests.exceptions.ConnectionError:
                attempt_time = time.time() - attempt_start
                self.logger.error(f"   [CONN_ERR] Connection error after {attempt_time:.2f}s (attempt {attempt + 1})")

                if attempt == self.max_retries - 1:
                    self._consecutive_failures += 1
                    total_time = time.time() - request_start
                    error_msg = "Connection error - is Ollama running?"
                    self.logger.error(f"   [FINAL] CONNECTION ERROR after {total_time:.2f}s")
                    return LLMResponse(content="", success=False, error=error_msg)

                backoff_time = 2 ** attempt
                self.logger.warning(f"   [RETRY] Retrying after connection error in {backoff_time}s...")
                time.sleep(backoff_time)

            except Exception as e:
                attempt_time = time.time() - attempt_start
                error_msg = f"{type(e).__name__}: {e}"
                self.logger.error(f"   [EXCEPTION] {error_msg} (attempt {attempt + 1}, {attempt_time:.2f}s)")

                if attempt == self.max_retries - 1:
                    self._consecutive_failures += 1
                    total_time = time.time() - request_start
                    self.logger.error(f"   [FINAL] EXCEPTION after {total_time:.2f}s")
                    return LLMResponse(content="", success=False, error=error_msg)

                backoff_time = 2 ** attempt
                self.logger.warning(f"   [RETRY] Retrying after exception in {backoff_time}s...")
                time.sleep(backoff_time)

        # This should never be reached, but just in case
        self._consecutive_failures += 1
        total_time = time.time() - request_start
        self.logger.error(f"   [FINAL] MAX RETRIES EXCEEDED after {total_time:.2f}s")
        return LLMResponse(content="", success=False, error="Max retries exceeded")
    
    def _process_streaming_response(self, response, request_start: float, attempt_num: int) -> LLMResponse:
        """Process streaming response with activity-based timeout and live feedback"""
        import sys

        collected_content = ""
        last_activity_time = time.time()
        activity_timeout = 120  # 2 minutes of inactivity before timeout
        total_chunks = 0
        total_tokens = 0
        metadata = {}

        self.logger.info("   [STREAM] Starting to receive response stream...")
        self.logger.info(f"   [STREAM] Activity timeout: {activity_timeout}s (resets on each chunk)")
        print("   [LIVE] Generating response", end="", flush=True)

        try:
            first_chunk_received = False

            for line in response.iter_lines(decode_unicode=True):
                current_time = time.time()

                # Check for activity timeout (no new data for activity_timeout seconds)
                if current_time - last_activity_time > activity_timeout:
                    elapsed = current_time - request_start
                    print("  [TIMEOUT]", flush=True)
                    self.logger.error("=" * 80)
                    self.logger.error("❌ LLM RESPONSE TIMEOUT")
                    self.logger.error("=" * 80)
                    self.logger.error(f"⏱️  No activity for {activity_timeout}s (total: {elapsed:.1f}s)")
                    self.logger.error(f"📦 Chunks received: {total_chunks}")
                    self.logger.error(f"📋 Partial content: {len(collected_content)} chars")
                    if collected_content:
                        preview = (
                            f"{collected_content[:500]}..."
                            if len(collected_content) > 500
                            else collected_content
                        )
                        self.logger.error(f"🕰️ Partial response: {preview}")
                    self.logger.error("=" * 80)
                    self._consecutive_failures += 1
                    return LLMResponse(
                        content=collected_content, 
                        success=False, 
                        error=f"Activity timeout after {activity_timeout}s of inactivity (received {total_chunks} chunks)"
                    )

                if line:  # Non-empty line received
                    if not first_chunk_received:
                        first_chunk_time = current_time - request_start
                        self.logger.info(f"   [STREAM] First chunk received after {first_chunk_time:.2f}s")
                        first_chunk_received = True

                    last_activity_time = current_time  # Reset activity timer
                    total_chunks += 1

                    try:
                        chunk_data = json.loads(line)

                        # Extract response content
                        if 'response' in chunk_data:
                            chunk_content = chunk_data['response']
                            collected_content += chunk_content
                            total_tokens += len(chunk_content.split())

                            # Check for massive response that indicates extended response bug
                            response_lines = collected_content.count('\n') + 1
                            char_count = len(collected_content)

                            # More aggressive detection of extended response bug
                            if response_lines > 200 or char_count > 10000:  # Reduced from 200/6000 to 120/4000
                                print("\n  [OVERSIZED RESPONSE DETECTED]", flush=True)
                                self.logger.error("=" * 80)
                                self.logger.error("EXTENDED RESPONSE BUG DETECTED")
                                self.logger.error("=" * 80)
                                self.logger.error(f"Response lines: {response_lines} (limit: 120)")
                                self.logger.error(f"Response chars: {char_count:,} (limit: 4000)")
                                self.logger.error("This appears to be the 'entire function dump' bug")
                                self.logger.error("Rejecting oversized response to prevent system overload")
                                self.logger.error("=" * 80)
                                return LLMResponse(
                                    content="", 
                                    success=False, 
                                    error=f"Extended response bug detected - response too large ({response_lines} lines, {char_count} chars)"
                                )

                            # Show real-time response content instead of dots
                            print(chunk_content, end="", flush=True)

                        # Extract metadata from final chunk
                        if chunk_data.get('done', False):
                            metadata = {
                                'model': chunk_data.get('model'),
                                'created_at': chunk_data.get('created_at'),
                                'total_duration': chunk_data.get('total_duration'),
                                'load_duration': chunk_data.get('load_duration'),
                                'prompt_eval_count': chunk_data.get('prompt_eval_count'),
                                'eval_count': chunk_data.get('eval_count')
                            }
                            break

                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

            # Successful completion
            print("\\n   [STREAMING COMPLETE]", flush=True)  # Complete the progress line
            total_time = time.time() - request_start

            # Log successful response with metrics
            self._successful_requests += 1
            self._consecutive_failures = 0
            success_rate = (self._successful_requests / self._total_requests) * 100

            response_length = len(collected_content)

            self.logger.info("=" * 80)
            self.logger.info("LLM RESPONSE COMPLETED")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Time: {total_time:.2f}s")
            self.logger.info(f"Response Length: {response_length:,} characters")
            self.logger.info(f"Tokens: {metadata.get('eval_count', total_tokens):,}")
            self.logger.info(f"Chunks: {total_chunks}")
            self.logger.info(f"Success Rate: {success_rate:.1f}% ({self._successful_requests}/{self._total_requests})")

            # Calculate timing metrics if available
            if metadata.get('total_duration'):
                total_duration_ms = metadata.get('total_duration', 0) / 1_000_000
                load_duration_ms = metadata.get('load_duration', 0) / 1_000_000
                self.logger.info(f"Timing: load={load_duration_ms:.0f}ms, generate={total_duration_ms:.0f}ms")

            self.logger.info("FULL RESPONSE:")
            self.logger.info("-" * 80)
            # Show the complete response, not just a preview
            self.logger.info(collected_content)
            self.logger.info("-" * 80)
            self.logger.info("LLM RESPONSE END")
            self.logger.info("=" * 80)

            # Save full response for optimization requests
            if len(collected_content) > 1000:  # Likely an optimization response
                self._save_optimization_response(collected_content)

            return LLMResponse(
                content=collected_content,
                success=True,
                metadata=metadata
            )

        except Exception as e:
            print("  [ERROR]", flush=True)  # Complete the progress line
            elapsed = time.time() - request_start
            error_msg = f"Streaming error after {elapsed:.1f}s: {e}"
            self.logger.error(f"   [STREAM_ERROR] {error_msg}")
            self._consecutive_failures += 1
            return LLMResponse(
                content=collected_content, 
                success=False, 
                error=error_msg
            )
    
    def generate_error_fix(self, error_info: Dict, source_code: str, 
                          file_path: str) -> LLMResponse:
        """Generate a fix for a specific error with intelligent prompt handling"""
        error_type = error_info.get('error_type', 'Unknown')

        # Create error-specific guidance
        error_guidance = {
            'NameError': "For NameError, either: 1) Define the missing variable with an appropriate value, 2) Import the missing module/function, or 3) Fix the variable name if it's a typo. Do NOT create new function definitions.",
            'ZeroDivisionError': "For ZeroDivisionError, add a check to prevent division by zero, such as: if denominator != 0: result = x / denominator; else: result = 0",
            'IndentationError': "For IndentationError, fix the indentation to match Python's requirements. Function/class bodies must be indented.",
            'SyntaxError': "For SyntaxError, fix the syntax according to Python grammar rules. Common issues: missing colons, unmatched brackets, invalid characters.",
            'AttributeError': "For AttributeError, either: 1) Check if the object has the attribute using hasattr(), 2) Use getattr() with a default, or 3) Import the correct module.",
            'ImportError': "For ImportError, either: 1) Install the missing package, 2) Fix the import path, or 3) Use an alternative import.",
            'TypeError': "For TypeError, check argument types and ensure they match the expected types for the operation."
        }

        specific_guidance = error_guidance.get(error_type, "Analyze the error and provide an appropriate fix.")

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
{specific_guidance}

Format your response as:
```python
# Fixed code section (without line numbers)
[corrected code here]
```

Explain the fix briefly before the code block (max 2 sentences)."""

        # Apply intelligent prompt handling for large source code
        if self.prompt_optimizer and len(source_code) > 5000:
            self.logger.info(f"Large source code for error fix: {len(source_code)} chars")

            # Use targeted approach for large code
            if self.prompt_optimizer.should_optimize_prompt(source_code):
                truncated_source = self.prompt_optimizer.truncate_prompt_intelligently(source_code, max_size=4000)
                prompt = f"""Error Information:
- Type: {error_info.get('error_type', 'Unknown')}
- Message: {error_info.get('error_message', 'No message')}
- File: {file_path}
- Line: {error_info.get('line_number', 'Unknown')}
- Function: {error_info.get('function_name', 'Unknown')}

Source Code Context (truncated for processing):
```python
{truncated_source}
```

Note: Code was truncated to focus on the most relevant sections for error fixing.

Please analyze this error and provide a fix. Focus specifically on line {error_info.get('line_number', 'Unknown')} and the surrounding context."""
            else:
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
        else:
            # Normal size code - use standard approach
            enhanced_info = []
            if error_info.get('function_signature'):
                enhanced_info.append(f"- Function Signature: {error_info['function_signature']}")
            if error_info.get('code_block_range'):
                enhanced_info.append(f"- Code Block Range: {error_info['code_block_range']}")
            if error_info.get('is_websocket_error'):
                enhanced_info.append("- Context: WebSocket Server Error")
            if error_info.get('enhanced_context'):
                enhanced_info.append(f"- Enhanced Context: {error_info['enhanced_context']}")

            enhanced_info_str = '\n'.join(enhanced_info) if enhanced_info else ""

            prompt = f"""Error Information:
- Type: {error_info.get('error_type', 'Unknown')}
- Message: {error_info.get('error_message', 'No message')}
- File: {file_path}
- Line: {error_info.get('line_number', 'Unknown')}
- Function: {error_info.get('function_name', 'Unknown')}
{enhanced_info_str}

Source Code Context:
```python
{source_code}
```

Please analyze this error and provide a fix. Focus specifically on line {error_info.get('line_number', 'Unknown')} and the surrounding context."""

        return self._make_request(prompt, system_prompt, temperature=0.1, max_tokens=-1)
    
    def generate_function_fix(self, error_info: Dict, function_info, file_path: str) -> 'LLMResponse':
        """
        Generate a fix for an error by providing the complete function context.
        Enhanced with intelligent handling of massive functions.
        
        Args:
            error_info: Dictionary containing error details
            function_info: FunctionInfo object with complete function source
            file_path: Path to the file being fixed
            
        Returns:
            LLMResponse with the suggested function replacement
        """
        error_type = error_info.get('error_type', 'Unknown')
        function_source = function_info.source_code

        # Check if function is too large for direct processing
        if self.prompt_optimizer and len(function_source) > 6000:
            self.logger.warning(f"Large function detected for fix: {function_info.name} ({len(function_source)} chars)")

            if error_line := error_info.get('line_number'):
                # Extract targeted code around the error
                return self._generate_targeted_function_fix(error_info, function_info, file_path)
            else:
                # Apply intelligent truncation to function
                return self._generate_truncated_function_fix(error_info, function_info, file_path)

        # Normal function size - use standard approach
        return self._generate_standard_function_fix(error_info, function_info, file_path)
    
    def _generate_standard_function_fix(self, error_info: Dict, function_info, file_path: str) -> 'LLMResponse':
        """Generate fix for normal-sized functions"""
        error_type = error_info.get('error_type', 'Unknown')
        
        # Create error-specific guidance for function-level fixes
        error_guidance = {
            'NameError': "For NameError in a function, either: 1) Define the missing variable within the function, 2) Add the variable as a parameter, 3) Import needed modules at the top of the function, or 4) Fix variable name typos. Return the complete corrected function.",
            'ZeroDivisionError': "For ZeroDivisionError, add proper error handling or input validation within the function. Return the complete corrected function with safety checks.",
            'IndentationError': "For IndentationError, fix the indentation throughout the entire function. Ensure all function body lines are properly indented. Return the complete corrected function.",
            'SyntaxError': "For SyntaxError, fix the syntax issues throughout the function. Check for missing colons, unmatched brackets, etc. Return the complete corrected function.",
            'AttributeError': "For AttributeError, add proper attribute checks or imports within the function. Return the complete corrected function.",
            'ConnectionRefusedError': "For ConnectionRefusedError, add proper connection error handling and fallback logic. Return the complete corrected function with try-except blocks.",
            'ImportError': "For ImportError, either add proper imports or handle missing dependencies. Return the complete corrected function.",
            'TypeError': "For TypeError, add type checking and proper argument validation within the function. Return the complete corrected function."
        }
        
        specific_guidance = error_guidance.get(error_type, "Analyze the error and fix the function accordingly.")
        
        system_prompt = f"""You are an expert Python debugging assistant. You will be provided with a complete function that contains an error, and you must return the entire corrected function.

CRITICAL REQUIREMENTS:
1. Return the COMPLETE function definition - from 'def' to the end
2. Maintain exact original indentation and formatting style
3. Fix ONLY the specific error mentioned - do not add extra features
4. Preserve all original functionality while fixing the error
5. Include the complete function signature and all original logic
6. Do not create nested functions or duplicate function definitions

ERROR-SPECIFIC GUIDANCE for {error_type}:
{specific_guidance}

Format your response as:
```python
def function_name(parameters):
    # Complete corrected function body here
    # Include all original logic with the fix applied
```

Provide a brief explanation of the fix before the code block."""

        function_context = ""
        if function_info.class_name:
            function_context = f"This is a method of class '{function_info.class_name}'"
        else:
            function_context = "This is a standalone function"
            
        prompt = f"""Error Information:
- Type: {error_info.get('error_type', 'Unknown')}
- Message: {error_info.get('error_message', 'No message')}
- File: {file_path}
- Error Line: {error_info.get('line_number', 'Unknown')}
- Function: {function_info.name} (lines {function_info.start_line}-{function_info.end_line})

Function Context: {function_context}

Complete Function Source:
```python
{function_info.source_code}
```

Please analyze this function and provide a complete corrected version that fixes the {error_type} error."""
        
        return self._make_request(prompt, system_prompt, temperature=0.1, max_tokens=-1)
    
    def _generate_targeted_function_fix(self, error_info: Dict, function_info, file_path: str) -> 'LLMResponse':
        """Generate fix for large functions by targeting the error area"""
        error_line = error_info.get('line_number')
        function_lines = function_info.source_code.split('\n')
        function_start_line = function_info.start_line
        
        # Calculate relative error line within function
        relative_error_line = error_line - function_start_line
        
        # Extract context around the error (±10 lines)
        context_size = 15
        start_idx = max(0, relative_error_line - context_size)
        end_idx = min(len(function_lines), relative_error_line + context_size)
        
        targeted_lines = function_lines[start_idx:end_idx]
        targeted_code = '\n'.join(targeted_lines)
        
        # Create enhanced error info for targeted fix
        enhanced_error_info = error_info.copy()
        enhanced_error_info['targeted_code'] = targeted_code
        enhanced_error_info['block_start_line'] = function_start_line + start_idx
        enhanced_error_info['block_end_line'] = function_start_line + end_idx
        enhanced_error_info['massive_function'] = True
        enhanced_error_info['function_name'] = function_info.name
        
        # Add function signature and key context
        function_signature = function_lines[0] if function_lines else "def unknown():"
        enhanced_error_info['code_context'] = f"""Function Signature: {function_signature}
Function Total Size: {len(function_info.source_code)} characters
Extracted Context: Lines {function_start_line + start_idx} to {function_start_line + end_idx}"""
        
        self.logger.info(f"Using targeted fix for large function {function_info.name}: "
                        f"extracted {len(targeted_code)} chars around error line {error_line}")
        
        return self.generate_targeted_code_fix(enhanced_error_info, file_path)
    
    def _generate_truncated_function_fix(self, error_info: Dict, function_info, file_path: str) -> 'LLMResponse':
        """Generate fix for large functions using intelligent truncation"""
        if not self.prompt_optimizer:
            # Fallback to standard approach if no optimizer
            return self._generate_standard_function_fix(error_info, function_info, file_path)
        
        # Apply intelligent truncation to function source
        truncated_source = self.prompt_optimizer.truncate_prompt_intelligently(
            function_info.source_code, max_size=4000
        )
        
        # Create a modified function_info object for the truncated version
        class TruncatedFunctionInfo:
            def __init__(self, original_info, truncated_source):
                self.name = original_info.name
                self.class_name = getattr(original_info, 'class_name', None)
                self.start_line = original_info.start_line
                self.end_line = original_info.end_line
                self.source_code = truncated_source
        
        truncated_function_info = TruncatedFunctionInfo(function_info, truncated_source)
        
        self.logger.info(f"Using truncated fix for large function {function_info.name}: "
                        f"{len(function_info.source_code)} -> {len(truncated_source)} chars")
        
        # Add truncation notice to error info
        enhanced_error_info = error_info.copy()
        enhanced_error_info['truncation_notice'] = f"Function was truncated from {len(function_info.source_code)} to {len(truncated_source)} characters for processing"
        
        return self._generate_standard_function_fix(enhanced_error_info, truncated_function_info, file_path)
    
    def generate_targeted_code_fix(self, error_info: Dict, file_path: str) -> 'LLMResponse':
        """
        Generate a fix for an error using targeted code block extraction.
        This is more efficient than sending entire massive functions to the LLM.
        Enhanced with section-aware and debugger integration.
        """
        error_type = error_info.get('error_type', 'Unknown')

        # Enhanced system prompt for different fix types
        base_system_prompt = f"""You are an expert Python debugging assistant. You will be provided with a targeted code block containing an error, and you must return the corrected code block.

CRITICAL REQUIREMENTS:
1. Return only the CORRECTED code block that was provided (NEVER return entire functions)
2. Maintain exact original indentation and formatting style  
3. Fix ONLY the specific error mentioned - do not add extra features
4. Preserve all original functionality while fixing the error
5. Do not include the entire function - only the provided code block
6. Ensure the fix integrates properly with surrounding code
7. NEVER repeat the same code multiple times - provide ONE clean corrected version
8. Keep response focused and targeted - avoid unnecessary context
9. STRICT SIZE LIMIT: Maximum 60 lines of actual code in your response (reduced from 80)
10. Include sufficient context to ensure the fix is complete and correct

CRITICAL RESPONSE SIZE LIMITS:
- ABSOLUTE Maximum response: 150 lines TOTAL (reduced from 200)
- ABSOLUTE Maximum explanation: 3 sentences (reduced from 5)
- MAXIMUM characters: 4000 (NEW strict limit)
- Focus ONLY on the specific error - no additional context
- NEVER duplicate or repeat code blocks in your response
- If fix requires more than 60 lines, provide targeted changes only

ERROR-SPECIFIC GUIDANCE for {error_type}:
- Focus on the specific error location and immediate context
- Ensure variable names, imports, and logic flow are correct
- Add proper error handling only if directly related to the fix
- Maintain consistency with the existing codebase style"""

        # Enhanced prompts for different debugging modes
        if error_info.get('section_aware'):
            system_prompt = f"""{base_system_prompt}

SECTION-AWARE DEBUGGING MODE:
- You are working with a specific section of a massive function: {error_info.get('function_name')}
- Current section: {error_info.get('section_name', 'Unknown')}
- Section purpose: {error_info.get('section_description', 'Unknown')}
- Consider section context and dependencies with other sections
- Ensure the fix doesn't break the section's integration with the overall function flow"""

        elif error_info.get('debugger_enhanced'):
            variables_context = ""
            if error_info.get('debugger_variables'):
                variables_context = "\\nDebugger Variables at Error:\\n"
                for var, value in error_info.get('debugger_variables', {}).items():
                    variables_context += f"  {var} = {value}\\n"

            system_prompt = f"""{base_system_prompt}

DEBUGGER-ENHANCED MODE:
- This error was captured by VS Code debugger at the exact failure point
- You have access to runtime variable values and call stack
- Use this precise information to create an accurate fix{variables_context}
- Consider the runtime state when determining the appropriate fix"""

        elif error_info.get('massive_function'):
            system_prompt = f"""{base_system_prompt}

MASSIVE FUNCTION MODE:
- You are working with a section of a very large function ({error_info.get('function_name')})
- The provided code block is a focused extraction around the error
- Consider that there may be variables and state from other parts of the function
- Ensure the fix works within the context of the larger function"""

        else:
            system_prompt = base_system_prompt

        system_prompt += """

IMPORTANT: Keep your response FOCUSED and COMPLETE:
- Maximum 5 sentences of explanation
- Maximum 300 lines of code
- Provide sufficient context to ensure the fix is correct
- Include necessary surrounding code for proper integration

Format your response as:
```python
# Corrected code block here (same lines as input)
# Include all provided lines with the fix applied
```

Provide a brief explanation of the fix before the code block."""

        # Build enhanced prompt with available context
        prompt_parts = [f"""Error Information:
- Type: {error_info.get('error_type', 'Unknown')}
- Message: {error_info.get('error_message', 'No message')}
- File: {file_path}
- Error Line: {error_info.get('line_number', 'Unknown')}
- Function: {error_info.get('function_name', 'Unknown')}"""]

        # Add enhanced function metadata if available
        if error_info.get('function_signature'):
            prompt_parts.append(f"""- Function Signature: {error_info['function_signature']}""")

        if error_info.get('code_block_range'):
            prompt_parts.append(f"""- Code Block Range: {error_info['code_block_range']}""")

        if error_info.get('is_websocket_error'):
            prompt_parts.append(
                """- Context: WebSocket Server Error (requires special attention to async/network handling)"""
            )

        if error_info.get('enhanced_context'):
            prompt_parts.append(f"""- Enhanced Context: {error_info['enhanced_context']}""")

        # Add section-specific context
        if error_info.get('section_aware'):
            prompt_parts.append(f"""- Section: {error_info.get('section_name', 'Unknown')}
- Section Description: {error_info.get('section_description', 'Unknown')}""")

        # Add debugger context
        if error_info.get('debugger_enhanced'):
            if error_info.get('debugger_variables'):
                prompt_parts.append("- Runtime Variables:")
                prompt_parts.extend(
                    f"  {var} = {value}"
                    for var, value in error_info.get(
                        'debugger_variables', {}
                    ).items()
                )
            if error_info.get('call_stack'):
                prompt_parts.append("- Call Stack:")
                for i, frame in enumerate(error_info.get('call_stack', [])[:5]):  # Limit to 5 frames
                    prompt_parts.append(f"  {i+1}. {frame}")

        prompt_parts.append(f"""- Code Block Lines: {error_info.get('block_start_line', 'Unknown')}-{error_info.get('block_end_line', 'Unknown')}

Targeted Code Block to Fix:
```python
{error_info.get('targeted_code', 'No code provided')}
```

CRITICAL: The error is specifically on line {error_info.get('line_number', 'Unknown')} with the message: "{error_info.get('error_message', 'No message')}"

YOU MUST FIX THIS EXACT ERROR. Do not just copy the code - ACTUALLY FIX THE PROBLEM:
- For SyntaxError 'unmatched ')': Remove the extra closing parenthesis
- For SyntaxError "'(' was never closed": Add the missing closing parenthesis  
- For NameError: Define the missing variable or import
- For IndentationError: Fix the indentation
- For other errors: Apply the appropriate fix

Return the CORRECTED code with the error FIXED, not a copy of the broken code.""")

        if error_info.get('code_context'):
            prompt_parts.append(f"""
Additional Context:
{error_info.get('code_context')}""")

        prompt = "\\n".join(prompt_parts)

        return self._make_request(prompt, system_prompt, temperature=0.1, max_tokens=-1)
    
    def generate_diff_fix(self, error_info: Dict, original_code: str, 
                         context_lines: int = 5) -> LLMResponse:
        """Generate a diff-style fix for an error with intelligent prompt handling"""
        system_prompt = """You are a code fixing assistant. Generate a precise diff to fix the provided error.

RULES:
1. Keep response under 200 lines - provide comprehensive changes
2. Include minimal context for clear understanding
3. Use precise line references

Format your response as a unified diff:
```diff
--- original
+++ fixed
@@ -line_start,line_count +line_start,line_count @@
 unchanged_line
-removed_line
+added_line
 unchanged_line
```

Include a brief explanation before the diff (max 2 sentences)."""
        
        # Apply intelligent prompt handling for large code
        if self.prompt_optimizer and len(original_code) > 3000:
            self.logger.info(f"Large code for diff fix: {len(original_code)} chars")
            truncated_code = self.prompt_optimizer.truncate_prompt_intelligently(original_code, max_size=2500)
            
            prompt = f"""Error to fix:
Type: {error_info.get('error_type')}
Message: {error_info.get('error_message')}
Location: Line {error_info.get('line_number')} in {error_info.get('function_name')}

Original code (truncated for processing):
```python
{truncated_code}
```

Note: Code was truncated to focus on relevant sections.

Generate a minimal diff to fix this error."""
        else:
            prompt = f"""Error to fix:
Type: {error_info.get('error_type')}
Message: {error_info.get('error_message')}
Location: Line {error_info.get('line_number')} in {error_info.get('function_name')}

Original code:
```python
{original_code}
```

Generate a minimal diff to fix this error."""
        
        return self._make_request(prompt, system_prompt, temperature=0.1, max_tokens=-1)
    
    def generate_optimization(self, code_snippet: str, performance_context: str = "") -> LLMResponse:
        """Generate code optimizations with intelligent prompt handling"""
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

        # Log code size for monitoring (but send FULL code to LLM)
        self.logger.info(f"Optimization request: {len(code_snippet)} chars")

        # For optimization, ALWAYS send the full code to ensure complete analysis
        # Unlike regular prompts, optimization requires full context
        prompt = f"""Code to optimize:
```python
{code_snippet}
```

Performance context: {performance_context}

Please provide optimized version of this code with explanations.

CRITICAL: If no meaningful optimizations are possible, respond with "NO_OPTIMIZATIONS_NEEDED" only."""

        return self._make_request(prompt, system_prompt, temperature=0.2, operation_type="optimization")
    
    def generate_targeted_optimization(self, code_section: str, performance_issues: List[str]) -> LLMResponse:
        """Generate optimization for a specific code section with known performance issues"""
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
        
        return self._make_request(prompt, system_prompt, temperature=0.2, operation_type="optimization")
    
    def analyze_code_quality(self, code: str) -> LLMResponse:
        """Analyze code quality and suggest improvements with intelligent prompt handling"""
        system_prompt = """You are a code quality analyst. Review the code and provide suggestions for improvement.

ANALYSIS FOCUS:
1. Code structure and organization
2. Error handling
3. Documentation and comments
4. Variable naming and conventions
5. Potential bugs or issues
6. Performance considerations

RULES:
1. Provide specific, actionable recommendations
2. Keep analysis under 200 lines - provide thorough issue analysis
3. Prioritize critical issues over minor style preferences
4. Include code examples for important fixes

Provide a structured analysis with specific recommendations."""
        
        # Apply intelligent prompt handling for large code
        if self.prompt_optimizer and len(code) > 4000:
            self.logger.info(f"Large code for quality analysis: {len(code)} chars")
            
            if self.prompt_optimizer.should_optimize_prompt(code):
                # Use sectioning for very large code
                sections = self.prompt_optimizer.create_prompt_sections(code)
                if len(sections) > 1:
                    # Analyze the highest priority section
                    best_section = max(sections, key=lambda x: x.priority)
                    analyzed_code = best_section.content
                    
                    prompt = f"""Please analyze this code section for quality and provide improvement suggestions:

Code Section (selected from larger codebase):
```python
{analyzed_code}
```

Section Info: Priority {best_section.priority}, Type: {best_section.context_type}
Original code size: {len(code)} chars

Focus on practical improvements that enhance maintainability, reliability, and performance."""
                else:
                    # Single large section - apply truncation
                    truncated_code = self.prompt_optimizer.truncate_prompt_intelligently(code, max_size=3500)
                    prompt = f"""Please analyze this code for quality and provide improvement suggestions:

```python
{truncated_code}
```

Note: Code was truncated for analysis - focus on the most critical quality improvements.

Focus on practical improvements that enhance maintainability, reliability, and performance."""
            else:
                prompt = f"""Please analyze this code for quality and provide improvement suggestions:

```python
{code}
```

Focus on practical improvements that enhance maintainability, reliability, and performance."""
        else:
            # Normal size code
            prompt = f"""Please analyze this code for quality and provide improvement suggestions:

```python
{code}
```

Focus on practical improvements that enhance maintainability, reliability, and performance."""
        
        return self._make_request(prompt, system_prompt, temperature=0.3)
    
    def extract_code_from_response(self, response_content: str) -> str:
        """Extract code blocks from LLM response and clean formatting"""
        self.logger.info("Extracting code from LLM response...")
        self.logger.info(f"Response length: {len(response_content)} chars")

        # Check for "No optimization possible" responses
        if "No optimization possible:" in response_content:
            self.logger.warning("LLM indicated no optimization possible")
            return ""

        # Look for code blocks marked with ```python or ```
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'```python(.*?)```',
            r'```(.*?)```'
        ]

        for pattern in code_patterns:
            if matches := re.findall(pattern, response_content, re.DOTALL):
                # Return the first (and usually only) code block, cleaned
                extracted = self._clean_extracted_code(matches[0].strip())
                self.logger.info(f"Extracted code from markdown block: {len(extracted)} chars")

                if self._is_meaningful_code_change(extracted, response_content):
                    return extracted
                self.logger.warning("Extracted code appears to be meaningless repetition")
                return ""

        # If no code blocks found, try to extract lines that look like code
        lines = response_content.split('\n')
        code_lines = []
        in_code_section = False

        for line in lines:
            # Start code section if line looks like Python code
            if (line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'import ', 'from ')) or
                line.strip().endswith(':') or
                ('=' in line and not line.strip().startswith('#'))):
                in_code_section = True

            if in_code_section:
                code_lines.append(line)

                # Stop if we hit a line that doesn't look like code
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

        if extracted and self._is_meaningful_code_change(extracted, response_content):
            self.logger.info(f"Extracted code from response text: {len(extracted)} chars")
            return extracted
        else:
            self.logger.warning("No meaningful code changes detected in response")
            return ""
    
    def _is_meaningful_code_change(self, extracted_code: str, full_response: str) -> bool:
        """Check if extracted code represents a meaningful change vs just repetition"""
        # If the code is very short, it might be meaningful
        if len(extracted_code) < 100:
            return True
        
        # Check for optimization indicators
        optimization_indicators = [
            'optimized', 'improved', 'efficient', 'faster', 'better',
            'reduced', 'cached', 'vectorized', 'refactored'
        ]
        
        if any(indicator in full_response.lower() for indicator in optimization_indicators):
            return True
        
        # Check if response contains actual explanation of changes
        explanation_indicators = [
            'changed', 'modified', 'replaced', 'added', 'removed',
            'instead of', 'rather than', 'improvement', 'fix'
        ]
        
        if any(indicator in full_response.lower() for indicator in explanation_indicators):
            return True
        
        # If it's very long (likely full function dump), it's probably not meaningful
        if len(extracted_code) > 4000:
            self.logger.warning(f"Extracted code is very long ({len(extracted_code)} chars) - likely function dump")
            return False
        
        return True
    
    def extract_optimization_code_from_response(self, response_content: str) -> str:
        """Extract code from optimization responses, removing inappropriate imports"""
        self.logger.info(f"[CODE_EXTRACTION] Starting extraction from {len(response_content)} char response")
        
        # Check for "no optimizations needed" response
        if "NO_OPTIMIZATIONS_NEEDED" in response_content.upper():
            self.logger.info("[CODE_EXTRACTION] LLM indicated no optimizations needed")
            return ""
        
        # Check for "No optimization possible" responses
        if "no optimization possible:" in response_content.lower():
            self.logger.warning("[CODE_EXTRACTION] LLM indicated no optimization possible")
            return ""
        
        raw_code = self.extract_code_from_response(response_content)
        
        if not raw_code:
            self.logger.warning("[CODE_EXTRACTION] No code extracted by extract_code_from_response")
            return raw_code
        
        self.logger.info(f"[CODE_EXTRACTION] Raw code extracted: {len(raw_code)} chars")
        self.logger.info(f"[CODE_EXTRACTION] Raw code preview: {raw_code[:200]}...")
        
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
        self.logger.info(f"[CODE_EXTRACTION] Final code preview: {final_code[:200]}...")
        
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
                if code_part.strip():  # Only process non-empty lines
                    code_lines_with_indent.append(code_part)
            else:
                # No line number found, clean and keep the line
                cleaned_line = line
                # Remove simple marker prefixes but preserve indentation
                if cleaned_line.strip() and not cleaned_line.strip().startswith('Explanation:'):
                    code_lines_with_indent.append(cleaned_line)

        if not code_lines_with_indent:
            return code

        # Find minimum indentation to normalize
        min_indent = float('inf')
        for line in code_lines_with_indent:
            if line.strip():  # Only check non-empty lines
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # If no indentation found, set to 0
        if min_indent == float('inf'):
            min_indent = 0

        # Second pass: normalize indentation
        for line in code_lines_with_indent:
            if line.strip():
                # Remove the minimum indentation from all lines
                current_indent = len(line) - len(line.lstrip())
                new_indent = max(0, current_indent - min_indent)
                normalized_line = ' ' * int(new_indent) + line.lstrip()
                cleaned_lines.append(normalized_line)
            else:
                cleaned_lines.append('')  # Preserve empty lines

        return '\n'.join(cleaned_lines)
    
    def extract_diff_from_response(self, response_content: str) -> str:
        """Extract diff blocks from LLM response"""
        # Look for diff blocks
        diff_pattern = r'```diff\n(.*?)\n```'
        if matches := re.findall(diff_pattern, response_content, re.DOTALL):
            return matches[0].strip()

        # Look for unified diff format in the response
        lines = response_content.split('\n')
        diff_lines = []
        in_diff = False

        for line in lines:
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                in_diff = True

            if in_diff:
                diff_lines.append(line)

                # Stop at next section or end
                if line.strip() and not line.startswith(('---', '+++', '@@', ' ', '-', '+')):
                    break

        return '\n'.join(diff_lines).strip() if diff_lines else ""
    
    def _save_optimization_response(self, content: str):
        """Save optimization response to a file for later review"""
        try:
            from datetime import datetime
            import os
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"llm_optimization_response_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"LLM Optimization Response - {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("END OF RESPONSE\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"   [SAVED] Full optimization response saved to {filename}")
            
        except Exception as e:
            self.logger.warning(f"   [SAVE_ERROR] Failed to save response: {e}")
    
    def get_status_summary(self) -> Dict[str, Union[str, int, float]]:
        """Get current LLM interface status summary"""
        current_time = time.time()
        time_since_last_check = current_time - self._last_health_check if self._last_health_check > 0 else 0
        success_rate = (self._successful_requests / self._total_requests * 100) if self._total_requests > 0 else 0
        
        return {
            'health_status': self._health_status,
            'model': self.model_name,
            'base_url': self.base_url,
            'total_requests': self._total_requests,
            'successful_requests': self._successful_requests,
            'consecutive_failures': self._consecutive_failures,
            'success_rate': round(success_rate, 1),
            'time_since_last_check': round(time_since_last_check, 1),
            'timeout_setting': self.timeout,
            'max_retries': self.max_retries
        }
    
    def log_status_summary(self):
        """Log a comprehensive status summary"""
        status = self.get_status_summary()

        self.logger.info("[STATUS] LLM Status Summary")
        self.logger.info(f"   [MODEL] {status['model']} @ {status['base_url']}")
        self.logger.info(f"   [HEALTH] {status['health_status']}")
        self.logger.info(f"   [STATS] Success Rate: {status['success_rate']}% ({status['successful_requests']}/{status['total_requests']})")
        self.logger.info(f"   [CONFIG] timeout={status['timeout_setting']}s, retries={status['max_retries']}")

        if int(status['consecutive_failures']) > 0:
            self.logger.warning(f"   [WARN] Consecutive failures: {status['consecutive_failures']}")

        if float(status['time_since_last_check']) > 60:
            self.logger.warning(f"   [WARN] Last health check: {status['time_since_last_check']:.0f}s ago")