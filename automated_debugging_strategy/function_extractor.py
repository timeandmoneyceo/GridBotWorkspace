"""
Function Extractor Module

This module provides functionality to detect and extract complete function definitions
from Python source code based on line numbers where errors occur.
"""

import ast
import re
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class FunctionInfo:
    """Container for function information"""
    name: str
    start_line: int
    end_line: int
    source_code: str
    class_name: Optional[str] = None
    is_method: bool = False

@dataclass
class CodeBlockInfo:
    """Container for code block information (try/except, if/else, etc.)"""
    block_type: str  # 'try', 'if', 'for', 'while', 'with', etc.
    start_line: int
    end_line: int
    source_code: str
    parent_function: Optional[str] = None
    indentation_level: int = 0

@dataclass  
class CodeBlockInfo:
    """Container for any code block information (function, try/except, if/else, etc.)"""
    block_type: str  # 'function', 'method', 'try', 'except', 'if', 'else', 'elif', 'finally', 'for', 'while'
    name: str
    start_line: int
    end_line: int
    source_code: str
    parent_function: Optional[str] = None
    class_name: Optional[str] = None
    is_method: bool = False
    indentation_level: int = 0

class FunctionExtractor:
    """Extract complete function definitions from Python source code"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_function_info(self, file_path: str, line_number: int) -> Optional[FunctionInfo]:
        """Extract function information for a given file and line number"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            return self.find_function_containing_line(source_code, line_number)
            
        except Exception as e:
            self.logger.error(f"Failed to extract function info from {file_path}:{line_number}: {e}")
            return None
    
    def find_function_containing_line(self, source_code: str, target_line: int) -> Optional[FunctionInfo]:
        """
        Find the function that contains the given line number.
        
        Args:
            source_code: The complete source code as string
            target_line: Line number to search for (1-indexed)
            
        Returns:
            FunctionInfo object if found, None otherwise
        """
        # Try to find the most specific containing function
        return self._find_most_specific_function(source_code, target_line)
    
    def find_code_block_containing_line(self, source_code: str, target_line: int) -> Optional[CodeBlockInfo]:
        """
        Find the most specific code block (try/except, if/else, etc.) containing the line.
        
        Args:
            source_code: The complete source code as string
            target_line: Line number to search for (1-indexed)
            
        Returns:
            CodeBlockInfo object if found, None otherwise
        """
        try:
            tree = ast.parse(source_code)
            source_lines = source_code.split('\n')
            
            # Find the most specific code block
            return self._find_most_specific_code_block(tree, source_lines, target_line)
            
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            return self._regex_find_code_block(source_code, target_line)
        except Exception as e:
            self.logger.error(f"Error finding code block: {e}")
            return None
    
    def _find_most_specific_function(self, source_code: str, target_line: int) -> Optional[FunctionInfo]:
        """Find the most specific (innermost) function containing the target line"""
        try:
            # Parse the source code into an AST
            tree = ast.parse(source_code)

            # Find all function definitions
            functions = self._extract_all_functions(tree, source_code)

            containing_functions = [
                func_info
                for func_info in functions
                if func_info.start_line <= target_line <= func_info.end_line
            ]
            if not containing_functions:
                return None

            # Return the most specific (smallest range) function
            return min(containing_functions, key=lambda f: f.end_line - f.start_line)

        except SyntaxError as e:
            self.logger.warning(f"Syntax error in source code: {e}")
            # Fall back to regex-based extraction
            return self._regex_fallback_extraction(source_code, target_line)
        except Exception as e:
            self.logger.error(f"Error parsing source code: {e}")
            return None
    
    def _find_most_specific_code_block(self, tree: ast.AST, source_lines: List[str], target_line: int) -> Optional[CodeBlockInfo]:
        """Find the most specific code block containing the target line"""
        code_blocks = []
        
        class CodeBlockVisitor(ast.NodeVisitor):
            def __init__(self, extractor):
                self.extractor = extractor
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function
                
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_Try(self, node):
                self._add_block_if_contains_line(node, 'try')
                self.generic_visit(node)
            
            def visit_If(self, node):
                self._add_block_if_contains_line(node, 'if')
                self.generic_visit(node)
            
            def visit_For(self, node):
                self._add_block_if_contains_line(node, 'for')
                self.generic_visit(node)
            
            def visit_While(self, node):
                self._add_block_if_contains_line(node, 'while')
                self.generic_visit(node)
            
            def visit_With(self, node):
                self._add_block_if_contains_line(node, 'with')
                self.generic_visit(node)
            
            def _add_block_if_contains_line(self, node, block_type):
                start_line = node.lineno
                end_line = self.extractor._find_node_end_line(node, source_lines)
                
                if start_line <= target_line <= end_line:
                    # Get indentation level
                    if start_line <= len(source_lines):
                        line_content = source_lines[start_line - 1]
                        indent_level = len(line_content) - len(line_content.lstrip())
                    else:
                        indent_level = 0
                    
                    # Extract source code for this block
                    block_source = '\\n'.join(source_lines[start_line-1:end_line])
                    
                    block_info = CodeBlockInfo(
                        block_type=block_type,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=block_source,
                        parent_function=self.current_function,
                        indentation_level=indent_level
                    )
                    code_blocks.append(block_info)
        
        visitor = CodeBlockVisitor(self)
        visitor.visit(tree)
        
        if not code_blocks:
            return None
        
        # Return the most specific (smallest range) code block
        return min(code_blocks, key=lambda b: b.end_line - b.start_line)
    
    def _find_node_end_line(self, node: ast.AST, source_lines: List[str]) -> int:
        """Find the end line of an AST node by analyzing indentation"""
        start_line = node.lineno - 1  # Convert to 0-indexed
        
        if start_line >= len(source_lines):
            return len(source_lines)
        
        # Find the indentation level of the node
        node_line = source_lines[start_line]
        node_indent = len(node_line) - len(node_line.lstrip())
        
        # Look for the next line with equal or lesser indentation
        for i in range(start_line + 1, len(source_lines)):
            line = source_lines[i]
            
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check indentation
            line_indent = len(line) - len(line.lstrip())
            
            if line_indent <= node_indent:
                return i  # Found end of block
        
        return len(source_lines)  # End of file
    
    def _regex_find_code_block(self, source_code: str, target_line: int) -> Optional[CodeBlockInfo]:
        """Fallback regex-based code block detection"""
        lines = source_code.split('\\n')
        
        if target_line > len(lines):
            return None
        
        # Look backwards for block keywords
        for i in range(target_line - 1, -1, -1):
            line = lines[i].strip()
            
            # Check for various block types
            block_patterns = {
                'try': r'^try\\s*:',
                'if': r'^if\\s+.*:',
                'elif': r'^elif\\s+.*:',
                'else': r'^else\\s*:',
                'for': r'^for\\s+.*:',
                'while': r'^while\\s+.*:',
                'with': r'^with\\s+.*:'
            }
            
            for block_type, pattern in block_patterns.items():
                if re.match(pattern, line):
                    start_line = i + 1
                    indent_level = len(lines[i]) - len(lines[i].lstrip())
                    
                    # Find end of block
                    end_line = len(lines)
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip():
                            line_indent = len(lines[j]) - len(lines[j].lstrip())
                            if line_indent <= indent_level:
                                end_line = j
                                break
                    
                    block_source = '\\n'.join(lines[start_line-1:end_line])
                    
                    return CodeBlockInfo(
                        block_type=block_type,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=block_source,
                        indentation_level=indent_level
                    )
        
        return None
    
    def _extract_all_functions(self, tree: ast.AST, source_code: str) -> List[FunctionInfo]:
        """Extract all function definitions from the AST"""
        functions = []
        source_lines = source_code.split('\n')
        
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, extractor):
                self.current_class = None
                self.extractor = extractor
                
            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class
                
            def visit_FunctionDef(self, node):
                # Extract the complete function source
                start_line = node.lineno
                end_line = self.extractor._find_function_end_line(node, source_lines)
                
                if end_line is not None:
                    function_source = '\n'.join(source_lines[start_line-1:end_line])
                    
                    func_info = FunctionInfo(
                        name=node.name,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=function_source,
                        class_name=self.current_class,
                        is_method=self.current_class is not None
                    )
                    functions.append(func_info)
                
                self.generic_visit(node)
                
            def visit_AsyncFunctionDef(self, node):
                # Handle async functions the same way
                self.visit_FunctionDef(node)
        
        visitor = FunctionVisitor(self)
        visitor.visit(tree)
        return functions
    
    def _find_function_end_line(self, func_node: ast.FunctionDef, source_lines: List[str]) -> Optional[int]:
        """
        Find the last line of a function by analyzing indentation.
        
        Args:
            func_node: AST node for the function
            source_lines: List of source code lines
            
        Returns:
            Line number (1-indexed) of the function's end, or None if not found
        """
        start_line = func_node.lineno - 1  # Convert to 0-indexed

        if start_line >= len(source_lines):
            return None

        # Find the indentation level of the function definition
        func_line = source_lines[start_line]
        func_indent = len(func_line) - len(func_line.lstrip())

        # Look for the next line with equal or lesser indentation
        for i in range(start_line + 1, len(source_lines)):
            line = source_lines[i]

            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Check indentation
            line_indent = len(line) - len(line.lstrip())

            # If we find a line with equal or lesser indentation, 
            # the function ends at the previous non-empty line
            if line_indent <= func_indent:
                return next(
                    (
                        j + 1
                        for j in range(i - 1, start_line, -1)
                        if source_lines[j].strip()
                    ),
                    i,
                )
        # If we reach the end of file, function ends at the last line
        return len(source_lines)
    
    def _regex_fallback_extraction(self, source_code: str, target_line: int) -> Optional[FunctionInfo]:
        """
        Fallback function extraction using regex when AST parsing fails.
        This is less reliable but works with syntax errors.
        """
        lines = source_code.split('\n')

        if target_line > len(lines):
            return None

        # Find all function definitions and their proper boundaries
        functions = []
        current_function = None

        for i, line in enumerate(lines):
            if func_match := re.match(
                r'^(async\s+)?def\s+(\w+)\s*\(.*\).*:', line
            ):
                # End previous function if one exists
                if current_function:
                    current_function['end_line'] = i
                    functions.append(current_function)

                # Start new function
                current_function = {
                    'name': func_match[2],
                    'start_line': i + 1,
                    'end_line': len(lines),
                }

            elif current_function and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                # This is top-level code, so the function ended
                if not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                    current_function['end_line'] = i
                    functions.append(current_function)
                    current_function = None

        # Add the last function if it exists
        if current_function:
            functions.append(current_function)

        # Find the function that contains the target line
        for func in functions:
            if func['start_line'] <= target_line <= func['end_line']:
                # Extract function source
                function_source = '\n'.join(lines[func['start_line']-1:func['end_line']])

                return FunctionInfo(
                    name=func['name'],
                    start_line=func['start_line'],
                    end_line=func['end_line'],
                    source_code=function_source
                )

        return None
    
    def find_code_block_containing_line(self, source_code: str, target_line: int) -> Optional[CodeBlockInfo]:
        """
        Find the most specific code block that contains the given line number.
        This could be a function, try/except block, if/else block, etc.
        
        Args:
            source_code: The complete source code as string
            target_line: Line number to search for (1-indexed)
            
        Returns:
            CodeBlockInfo object if found, None otherwise
        """
        try:
            # First try to find the containing function
            function_info = self.find_function_containing_line(source_code, target_line)
            
            if not function_info:
                return None
            
            # Now look for more specific blocks within the function
            lines = source_code.split('\n')
            
            # Find all code blocks within the function
            blocks = self._find_all_code_blocks_in_range(
                lines, function_info.start_line, function_info.end_line, 
                function_info.name, function_info.class_name
            )
            
            # Find the most specific block containing the target line
            most_specific_block = None
            for block in blocks:
                if block.start_line <= target_line <= block.end_line and (most_specific_block is None or 
                                        (block.end_line - block.start_line) < (most_specific_block.end_line - most_specific_block.start_line)):
                    most_specific_block = block
            
            return most_specific_block or self._function_info_to_code_block_info(function_info)
            
        except Exception as e:
            self.logger.error(f"Error finding code block: {e}")
            return None
    
    def _find_all_code_blocks_in_range(self, lines: List[str], start_line: int, end_line: int, 
                                     parent_function: str, class_name: Optional[str]) -> List[CodeBlockInfo]:
        """Find all code blocks (try, if, for, etc.) within a given line range"""
        blocks = []
        
        i = start_line - 1  # Convert to 0-indexed
        while i < min(end_line, len(lines)):
            line = lines[i].strip()
            original_line = lines[i]
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Detect different types of code blocks
            block_info = None
            
            # Try blocks
            if line.startswith('try:'):
                block_info = self._extract_try_block(lines, i + 1, parent_function, class_name)
            
            # If/elif/else blocks
            elif line.startswith('if ') and line.endswith(':'):
                block_info = self._extract_if_block(lines, i + 1, parent_function, class_name)
            
            # For loops
            elif line.startswith('for ') and ' in ' in line and line.endswith(':'):
                block_info = self._extract_for_block(lines, i + 1, parent_function, class_name)
            
            # While loops  
            elif line.startswith('while ') and line.endswith(':'):
                block_info = self._extract_while_block(lines, i + 1, parent_function, class_name)
            
            # With statements
            elif line.startswith('with ') and line.endswith(':'):
                block_info = self._extract_with_block(lines, i + 1, parent_function, class_name)
            
            if block_info:
                blocks.append(block_info)
                i = block_info.end_line  # Skip to end of this block
            else:
                i += 1
        
        return blocks
    
    def _extract_try_block(self, lines: List[str], start_line: int, parent_function: str, 
                          class_name: Optional[str]) -> Optional[CodeBlockInfo]:
        """Extract a complete try/except/finally block"""
        try:
            start_idx = start_line - 1  # Convert to 0-indexed
            if start_idx >= len(lines):
                return None
            
            # Find base indentation
            base_indent = len(lines[start_idx - 1]) - len(lines[start_idx - 1].lstrip())
            
            # Find the end of the complete try/except/finally structure
            end_line = self._find_try_block_end(lines, start_idx, base_indent)
            
            if end_line is None:
                return None
            
            # Extract the complete try block source
            block_source = '\n'.join(lines[start_idx - 1:end_line])
            
            return CodeBlockInfo(
                block_type='try',
                name='try_block',
                start_line=start_line,
                end_line=end_line,
                source_code=block_source,
                parent_function=parent_function,
                class_name=class_name,
                is_method=class_name is not None,
                indentation_level=base_indent
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting try block: {e}")
            return None
    
    def _find_try_block_end(self, lines: List[str], start_idx: int, base_indent: int) -> Optional[int]:
        """Find the end of a complete try/except/finally block"""
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if not stripped:  # Empty line
                i += 1
                continue
                
            line_indent = len(line) - len(line.lstrip())
            
            # If we find a line with equal or lesser indentation that's not part of try/except/finally
            if line_indent <= base_indent and not (stripped.startswith('except') or stripped.startswith('finally') or 
                                   stripped.startswith('else:') or stripped.startswith('elif')):
                return i
            
            i += 1
        
        return len(lines)
    
    def _extract_if_block(self, lines: List[str], start_line: int, parent_function: str,
                         class_name: Optional[str]) -> Optional[CodeBlockInfo]:
        """Extract a complete if/elif/else block"""
        try:
            start_idx = start_line - 1
            if start_idx >= len(lines):
                return None
            
            base_indent = len(lines[start_idx - 1]) - len(lines[start_idx - 1].lstrip())
            end_line = self._find_if_block_end(lines, start_idx, base_indent)
            
            if end_line is None:
                return None
            
            block_source = '\n'.join(lines[start_idx - 1:end_line])
            
            return CodeBlockInfo(
                block_type='if',
                name='if_block',
                start_line=start_line,
                end_line=end_line,
                source_code=block_source,
                parent_function=parent_function,
                class_name=class_name,
                is_method=class_name is not None,
                indentation_level=base_indent
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting if block: {e}")
            return None
    
    def _find_if_block_end(self, lines: List[str], start_idx: int, base_indent: int) -> Optional[int]:
        """Find the end of a complete if/elif/else block"""
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if not stripped:
                i += 1
                continue
                
            line_indent = len(line) - len(line.lstrip())
            
            if line_indent <= base_indent and not (stripped.startswith('elif ') or stripped.startswith('else:')):
                return i
            
            i += 1
        
        return len(lines)
    
    def _extract_for_block(self, lines: List[str], start_line: int, parent_function: str,
                          class_name: Optional[str]) -> Optional[CodeBlockInfo]:
        """Extract a for loop block"""
        return self._extract_simple_block(lines, start_line, 'for', parent_function, class_name)
    
    def _extract_while_block(self, lines: List[str], start_line: int, parent_function: str,
                            class_name: Optional[str]) -> Optional[CodeBlockInfo]:
        """Extract a while loop block"""
        return self._extract_simple_block(lines, start_line, 'while', parent_function, class_name)
    
    def _extract_with_block(self, lines: List[str], start_line: int, parent_function: str,
                           class_name: Optional[str]) -> Optional[CodeBlockInfo]:
        """Extract a with statement block"""
        return self._extract_simple_block(lines, start_line, 'with', parent_function, class_name)
    
    def _extract_simple_block(self, lines: List[str], start_line: int, block_type: str,
                             parent_function: str, class_name: Optional[str]) -> Optional[CodeBlockInfo]:
        """Extract a simple indented block (for, while, with)"""
        try:
            start_idx = start_line - 1
            if start_idx >= len(lines):
                return None
            
            base_indent = len(lines[start_idx - 1]) - len(lines[start_idx - 1].lstrip())
            end_line = self._find_simple_block_end(lines, start_idx, base_indent)
            
            if end_line is None:
                return None
            
            block_source = '\n'.join(lines[start_idx - 1:end_line])
            
            return CodeBlockInfo(
                block_type=block_type,
                name=f'{block_type}_block',
                start_line=start_line,
                end_line=end_line,
                source_code=block_source,
                parent_function=parent_function,
                class_name=class_name,
                is_method=class_name is not None,
                indentation_level=base_indent
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting {block_type} block: {e}")
            return None
    
    def _find_simple_block_end(self, lines: List[str], start_idx: int, base_indent: int) -> Optional[int]:
        """Find the end of a simple indented block"""
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= base_indent:
                    return i
            
            i += 1
        
        return len(lines)
    
    def _function_info_to_code_block_info(self, func_info: FunctionInfo) -> CodeBlockInfo:
        """Convert FunctionInfo to CodeBlockInfo"""
        return CodeBlockInfo(
            block_type='method' if func_info.is_method else 'function',
            name=func_info.name,
            start_line=func_info.start_line,
            end_line=func_info.end_line,
            source_code=func_info.source_code,
            parent_function=None,
            class_name=func_info.class_name,
            is_method=func_info.is_method,
            indentation_level=self._get_indentation_level(func_info.source_code)
        )
    
    def _get_indentation_level(self, source_code: str) -> int:
        """Get the base indentation level of source code"""
        lines = source_code.split('\n')
        return next(
            (len(line) - len(line.lstrip()) for line in lines if line.strip()), 0
        )
    
    def extract_function_by_name(self, source_code: str, function_name: str) -> Optional[FunctionInfo]:
        """
        Extract a specific function by name.
        
        Args:
            source_code: The complete source code as string
            function_name: Name of the function to extract
            
        Returns:
            FunctionInfo object if found, None otherwise
        """
        try:
            tree = ast.parse(source_code)
            functions = self._extract_all_functions(tree, source_code)

            return next(
                (
                    func_info
                    for func_info in functions
                    if func_info.name == function_name
                ),
                None,
            )
        except Exception as e:
            self.logger.error(f"Error extracting function {function_name}: {e}")
            return None
        """
        Extract a specific function by name.
        
        Args:
            source_code: The complete source code as string
            function_name: Name of the function to extract
            
        Returns:
            FunctionInfo object if found, None otherwise
        """
        try:
            tree = ast.parse(source_code)
            functions = self._extract_all_functions(tree, source_code)

            return next(
                (
                    func_info
                    for func_info in functions
                    if func_info.name == function_name
                ),
                None,
            )
        except Exception as e:
            self.logger.error(f"Error extracting function {function_name}: {e}")
            return None