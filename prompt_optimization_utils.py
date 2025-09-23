"""
Prompt Optimization Utilities

This module provides utilities for handling large prompts sent to LLMs,
including code block detection, intelligent truncation, and sectioning.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CodeBlock:
    """Represents a detected code block in text"""
    start_index: int
    end_index: int
    content: str
    language: str
    line_count: int
    block_type: str  # 'function', 'class', 'module', 'snippet'

@dataclass
class PromptSection:
    """Represents a section of a large prompt"""
    section_id: int
    content: str
    code_blocks: List[CodeBlock]
    priority: int  # 1-10, 10 being highest priority
    context_type: str  # 'system', 'error_info', 'code', 'instructions'

class PromptOptimizer:
    """Handles optimization of large prompts for LLM processing"""
    
    def __init__(self, max_prompt_size: int = 8000, target_section_size: int = 800):
        """
        Initialize the prompt optimizer
        
        Args:
            max_prompt_size: Maximum size before splitting prompts
            target_section_size: Target size for each prompt section (~10% of max)
        """
        self.max_prompt_size = max_prompt_size
        self.target_section_size = target_section_size
        self.logger = logging.getLogger(__name__)
        
    def detect_code_blocks(self, text: str) -> List[CodeBlock]:
        """
        Detect and parse code blocks in text
        
        Returns list of CodeBlock objects with metadata
        """
        code_blocks = []
        
        # Pattern for markdown code blocks with language
        markdown_pattern = r'```(\w*)\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            language = match.group(1) or 'unknown'
            content = match.group(2)
            line_count = content.count('\n') + 1
            
            # Determine block type based on content
            block_type = self._classify_code_block(content)
            
            code_blocks.append(CodeBlock(
                start_index=match.start(),
                end_index=match.end(),
                content=content,
                language=language,
                line_count=line_count,
                block_type=block_type
            ))
        
        # Pattern for inline code sections (indented blocks)
        if not code_blocks:  # Only check for indented blocks if no markdown blocks found
            indented_blocks = self._detect_indented_code_blocks(text)
            code_blocks.extend(indented_blocks)
        
        return code_blocks
    
    def _classify_code_block(self, code: str) -> str:
        """Classify the type of code block based on content"""
        lines = code.strip().split('\n')
        
        # Check for function definitions
        if any(line.strip().startswith('def ') for line in lines):
            return 'function'
        
        # Check for class definitions
        if any(line.strip().startswith('class ') for line in lines):
            return 'class'
        
        # Check for imports (likely module-level)
        if any(line.strip().startswith(('import ', 'from ')) for line in lines):
            return 'module'
        
        # Default to snippet
        return 'snippet'
    
    def _detect_indented_code_blocks(self, text: str) -> List[CodeBlock]:
        """Detect indented code blocks (when no markdown formatting)"""
        lines = text.split('\n')
        code_blocks = []
        current_block = []
        current_start = -1
        in_code_block = False

        for i, line in enumerate(lines):
            # Check if line looks like code (indented and contains code patterns)
            is_code_line = (
                line.startswith(('    ', '\t')) or  # Indented
                line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:')) or
                ('=' in line and not line.strip().startswith('#')) or
                line.strip().endswith(':')
            )

            if is_code_line and not in_code_block:
                # Start of new code block
                in_code_block = True
                current_start = i
                current_block = [line]
            elif is_code_line:
                # Continue code block
                current_block.append(line)
            elif in_code_block:
                # End of code block
                if current_block:
                    content = '\n'.join(current_block)
                    block_type = self._classify_code_block(content)

                    # Calculate character positions (approximate)
                    start_char = sum(len(lines[j]) + 1 for j in range(current_start))
                    end_char = start_char + len(content)

                    code_blocks.append(CodeBlock(
                        start_index=start_char,
                        end_index=end_char,
                        content=content,
                        language='python',
                        line_count=len(current_block),
                        block_type=block_type
                    ))

                in_code_block = False
                current_block = []

        # Handle case where text ends with a code block
        if in_code_block and current_block:
            content = '\n'.join(current_block)
            block_type = self._classify_code_block(content)
            start_char = sum(len(lines[j]) + 1 for j in range(current_start))
            end_char = start_char + len(content)

            code_blocks.append(CodeBlock(
                start_index=start_char,
                end_index=end_char,
                content=content,
                language='python',
                line_count=len(current_block),
                block_type=block_type
            ))

        return code_blocks
    
    def should_optimize_prompt(self, prompt: str, system_prompt: str = None) -> bool:
        """Check if prompt should be optimized due to size"""
        total_size = len(prompt)
        if system_prompt:
            total_size += len(system_prompt)
        
        return total_size > self.max_prompt_size
    
    def create_prompt_sections(self, prompt: str, system_prompt: str = None) -> List[PromptSection]:
        """
        Break a large prompt into manageable sections based on code blocks
        
        Returns list of PromptSection objects, each targeting ~10% of max size
        """
        if not self.should_optimize_prompt(prompt, system_prompt):
            # Return single section if optimization not needed
            code_blocks = self.detect_code_blocks(prompt)
            return [PromptSection(
                section_id=1,
                content=prompt,
                code_blocks=code_blocks,
                priority=10,
                context_type='complete'
            )]

        self.logger.info(f"Optimizing large prompt: {len(prompt)} chars -> target sections of {self.target_section_size} chars")

        if code_blocks := self.detect_code_blocks(prompt):
            # Create sections based on code blocks
            return self._create_code_aware_sections(prompt, code_blocks, system_prompt)
        else:
            # No code blocks found, use simple text splitting
            return self._create_text_sections(prompt, system_prompt)
    
    def _create_text_sections(self, prompt: str, system_prompt: str = None) -> List[PromptSection]:
        """Create sections when no code blocks are detected"""
        sections = []
        
        # Split by logical boundaries (paragraphs, sentences)
        paragraphs = prompt.split('\n\n')
        current_section = ""
        section_id = 1
        
        for paragraph in paragraphs:
            if len(current_section + paragraph) > self.target_section_size and current_section:
                # Create section from current content
                sections.append(PromptSection(
                    section_id=section_id,
                    content=current_section.strip(),
                    code_blocks=[],
                    priority=5,  # Medium priority for text-only sections
                    context_type='text'
                ))
                section_id += 1
                current_section = paragraph
            else:
                current_section += "\n\n" + paragraph if current_section else paragraph
        
        # Add final section
        if current_section.strip():
            sections.append(PromptSection(
                section_id=section_id,
                content=current_section.strip(),
                code_blocks=[],
                priority=5,
                context_type='text'
            ))
        
        return sections
    
    def _create_code_aware_sections(self, prompt: str, code_blocks: List[CodeBlock], 
                                   system_prompt: str = None) -> List[PromptSection]:
        """Create sections intelligently based on detected code blocks"""
        sections = []
        section_id = 1
        
        # Sort code blocks by position
        code_blocks_sorted = sorted(code_blocks, key=lambda x: x.start_index)
        
        last_end = 0
        current_section_content = ""
        current_section_blocks = []
        
        for block in code_blocks_sorted:
            # Add text before this code block
            text_before = prompt[last_end:block.start_index]
            
            # Check if adding this block would exceed target size
            block_content = prompt[block.start_index:block.end_index]
            potential_content = current_section_content + text_before + block_content
            
            if len(potential_content) > self.target_section_size and current_section_content:
                # Create section with current content
                priority = self._calculate_section_priority(current_section_blocks)
                context_type = self._determine_context_type(current_section_content, current_section_blocks)
                
                sections.append(PromptSection(
                    section_id=section_id,
                    content=current_section_content.strip(),
                    code_blocks=current_section_blocks,
                    priority=priority,
                    context_type=context_type
                ))
                
                section_id += 1
                current_section_content = text_before + block_content
                current_section_blocks = [block]
            else:
                # Add to current section
                current_section_content += text_before + block_content
                current_section_blocks.append(block)
            
            last_end = block.end_index
        
        # Add any remaining text after the last code block
        remaining_text = prompt[last_end:]
        if remaining_text.strip():
            current_section_content += remaining_text
        
        # Create final section
        if current_section_content.strip():
            priority = self._calculate_section_priority(current_section_blocks)
            context_type = self._determine_context_type(current_section_content, current_section_blocks)
            
            sections.append(PromptSection(
                section_id=section_id,
                content=current_section_content.strip(),
                code_blocks=current_section_blocks,
                priority=priority,
                context_type=context_type
            ))
        
        return sections
    
    def _calculate_section_priority(self, code_blocks: List[CodeBlock]) -> int:
        """Calculate priority for a section based on its code blocks"""
        if not code_blocks:
            return 3  # Low priority for text-only sections
        
        # Higher priority for functions and classes
        block_types = [block.block_type for block in code_blocks]
        
        if 'function' in block_types:
            return 9  # High priority for functions
        elif 'class' in block_types:
            return 8  # High priority for classes
        elif 'module' in block_types:
            return 6  # Medium-high priority for module-level code
        else:
            return 5  # Medium priority for snippets
    
    def _determine_context_type(self, content: str, code_blocks: List[CodeBlock]) -> str:
        """Determine the type of context this section represents"""
        content_lower = content.lower()
        
        # Check for error-related content
        if any(term in content_lower for term in ['error', 'exception', 'traceback', 'failed']):
            return 'error_info'
        
        # Check for system prompts or instructions
        if any(term in content_lower for term in ['system:', 'instruction', 'rules:', 'requirements']):
            return 'instructions'
        
        # Check if primarily code
        if code_blocks and len(''.join(block.content for block in code_blocks)) > len(content) * 0.7:
            return 'code'
        
        return 'mixed'
    
    def truncate_prompt_intelligently(self, prompt: str, max_size: int = None) -> str:
        """
        Intelligently truncate a prompt while preserving important context
        
        Prioritizes:
        1. Error information
        2. Function/class definitions
        3. System instructions
        4. Code snippets
        """
        if max_size is None:
            max_size = self.max_prompt_size
        
        if len(prompt) <= max_size:
            return prompt
        
        self.logger.info(f"Truncating prompt from {len(prompt)} to {max_size} characters")
        
        # Create sections and prioritize
        sections = self.create_prompt_sections(prompt)
        sections_sorted = sorted(sections, key=lambda x: x.priority, reverse=True)
        
        # Build truncated prompt by adding highest priority sections first
        truncated_content = ""
        included_sections = []
        
        for section in sections_sorted:
            potential_size = len(truncated_content) + len(section.content) + 100  # Buffer
            
            if potential_size <= max_size:
                included_sections.append(section)
                truncated_content += section.content + "\n\n"
            else:
                # Try to fit partial content from high-priority sections
                remaining_space = max_size - len(truncated_content) - 200  # Buffer for truncation message
                if remaining_space > 500 and section.priority >= 7:  # High priority sections
                    partial_content = section.content[:remaining_space] + "\n... [TRUNCATED]"
                    truncated_content += partial_content
                break
        
        # Add truncation notice
        if len(included_sections) < len(sections):
            omitted_count = len(sections) - len(included_sections)
            truncated_content += f"\n\n[NOTE: {omitted_count} lower-priority sections omitted due to size limits]"
        
        self.logger.info(f"Truncation complete: {len(sections)} sections -> {len(included_sections)} included")
        
        return truncated_content.strip()
    
    def optimize_prompt_for_llm(self, prompt: str, system_prompt: str = None, 
                               context: str = "") -> Tuple[str, str, Dict]:
        """
        Main optimization method that handles prompt sizing and optimization
        
        Returns:
            (optimized_prompt, optimized_system_prompt, metadata)
        """
        metadata = {
            'original_size': len(prompt),
            'optimization_applied': False,
            'sections_created': 0,
            'truncation_applied': False,
            'context': context
        }
        
        # Add system prompt size if provided
        if system_prompt:
            metadata['original_system_size'] = len(system_prompt)
            metadata['total_original_size'] = len(prompt) + len(system_prompt)
        
        # Check if optimization is needed
        if not self.should_optimize_prompt(prompt, system_prompt):
            self.logger.debug(f"Prompt size OK: {len(prompt)} chars (max: {self.max_prompt_size})")
            return prompt, system_prompt or "", metadata
        
        self.logger.warning(f"Large prompt detected ({len(prompt)} chars) - applying optimization")
        metadata['optimization_applied'] = True
        
        # Create optimized sections
        sections = self.create_prompt_sections(prompt, system_prompt)
        metadata['sections_created'] = len(sections)
        
        if len(sections) == 1:
            # Single large section - apply intelligent truncation
            optimized_prompt = self.truncate_prompt_intelligently(prompt)
            metadata['truncation_applied'] = True
            self.logger.info(f"Applied intelligent truncation: {len(prompt)} -> {len(optimized_prompt)} chars")
        else:
            # Multiple sections - use highest priority section
            best_section = max(sections, key=lambda x: x.priority)
            optimized_prompt = best_section.content
            metadata['selected_section'] = best_section.section_id
            metadata['selected_priority'] = best_section.priority
            self.logger.info(f"Selected section {best_section.section_id} (priority {best_section.priority})")
        
        # Optimize system prompt if provided and too large
        optimized_system = system_prompt or ""
        if system_prompt and len(system_prompt) > 2000:  # System prompt size limit
            optimized_system = system_prompt[:1800] + "\n... [SYSTEM PROMPT TRUNCATED]"
            metadata['system_truncated'] = True
        
        metadata['final_size'] = len(optimized_prompt)
        metadata['final_system_size'] = len(optimized_system)
        metadata['size_reduction'] = metadata['original_size'] - metadata['final_size']
        
        self.logger.info(f"Prompt optimization complete: {metadata['original_size']} -> {metadata['final_size']} chars "
                        f"(reduced by {metadata['size_reduction']} chars)")
        
        return optimized_prompt, optimized_system, metadata