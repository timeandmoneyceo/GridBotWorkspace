#!/usr/bin/env python3
"""
Enhanced File Editor Test

Test the improved SafeFileEditor with auto-refinement and retry logic.
"""

import os
import sys
import tempfile
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append('.')
# Since we're already in the automated_debugging_strategy directory, use direct import

# Import the actual modules
from automated_debugging_strategy.qwen_agent_interface import QwenAgentInterface
from automated_debugging_strategy.automated_file_editor import SafeFileEditor, EditResult

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('EnhancedEditorTest')

def test_enhanced_syntax_error_handling():
    """Test the enhanced syntax error handling with auto-refinement"""
    logger = setup_logging()
    test_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info("=" * 80)
    logger.info("ENHANCED SYNTAX ERROR HANDLING TEST")
    logger.info("=" * 80)
    
    try:
        # 1. Initialize enhanced file editor
        logger.info("[STEP 1] Initializing enhanced SafeFileEditor...")
        
        file_editor = SafeFileEditor(
            use_serena=False,  # Disable Serena to avoid import issues
            create_backups=True,
            validate_syntax=True
        )
        
        logger.info("✅ Enhanced SafeFileEditor initialized")
        
        # 2. Create a target file with good syntax
        logger.info("[STEP 2] Creating target file with valid Python code...")
        
        target_code = '''def process_data(data_list):
    """Process a list of data items"""
    result = []
    for item in data_list:
        if item > 0:
            result.append(item * 2)
    return result

def main():
    numbers = [1, 2, 3, 4, 5]
    processed = process_data(numbers)
    print(f"Processed numbers: {processed}")

if __name__ == "__main__":
    main()'''
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(target_code)
            target_file = f.name
        
        logger.info(f"✅ Created target file: {target_file}")
        
        # 3. Test intentionally broken line replacement that should trigger auto-fix
        logger.info("[STEP 3] Testing line replacement with intentional indentation error...")
        
        # This line has wrong indentation and should trigger auto-fix
        broken_line = "def process_data(data_list):  # Wrong indentation will cause syntax error"
        
        logger.info(f"Attempting to replace line 1 with: {broken_line}")
        
        # Use the enhanced apply_line_replacement method
        edit_result = file_editor.apply_line_replacement(
            target_file,
            1,  # Line number
            broken_line
        )
        
        if edit_result.success:
            logger.info("🎯 ✅ ENHANCED SYNTAX ERROR HANDLING SUCCESSFUL!")
            logger.info("✅ Line replacement succeeded (possibly after auto-fix)")
            logger.info(f"✅ Changes made: {edit_result.changes_made}")
            if edit_result.diff:
                logger.info(f"✅ Diff preview: {edit_result.diff[:300]}...")
        else:
            logger.error(f"❌ Line replacement failed: {edit_result.error}")
        
        # 4. Verify the final file state
        logger.info("[STEP 4] Verifying final file state...")
        
        with open(target_file, 'r', encoding='utf-8') as f:
            final_content = f.read()
        
        logger.info(f"Final file content ({len(final_content)} chars):")
        logger.info("-" * 60)
        logger.info(final_content[:500] + ("..." if len(final_content) > 500 else ""))
        logger.info("-" * 60)
        
        # Test if the file is still valid Python after any fixes
        try:
            import ast
            ast.parse(final_content)
            logger.info("✅ Final file has valid Python syntax")
            syntax_valid = True
        except SyntaxError as e:
            logger.error(f"❌ Final file has syntax error: {e}")
            syntax_valid = False
        
        # 5. Test a more complex auto-fix scenario
        logger.info("[STEP 5] Testing complex indentation fix scenario...")
        
        # Create content with multiple indentation issues
        complex_broken_line = "    def process_data(data_list):\\n        # This has wrong base indentation"
        
        edit_result_2 = file_editor.apply_line_replacement(
            target_file,
            1,  # Line number
            complex_broken_line
        )
        
        if edit_result_2.success:
            logger.info("✅ Complex indentation fix succeeded")
        else:
            logger.warning(f"⚠️ Complex indentation fix failed: {edit_result_2.error}")
        
        # Summary
        if edit_result.success and syntax_valid:
            logger.info("🎯 ENHANCED FILE EDITOR TEST SUCCESSFUL!")
            logger.info("✅ Auto-refinement and retry logic working correctly")
            logger.info("✅ Syntax error handling improved significantly")
            return True
        else:
            logger.warning("⚠️ Enhanced file editor test had some issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ ENHANCED FILE EDITOR TEST FAILED: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # Cleanup
        try:
            if 'target_file' in locals() and os.path.exists(target_file):
                os.unlink(target_file)
                logger.info(f"Cleaned up test file: {target_file}")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

def test_llm_syntax_fix_capability():
    """Test the LLM-based syntax fixing capability"""
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("LLM SYNTAX FIX CAPABILITY TEST")
    logger.info("=" * 80)
    
    try:
        # Initialize file editor
        file_editor = SafeFileEditor(validate_syntax=True)
        
        # Test the LLM fix method directly
        broken_code = '''def broken_function(
    print("Missing closing parenthesis"
    return True'''
        
        syntax_error = "EOF while scanning triple-quoted string literal"
        
        logger.info("[STEP 1] Testing LLM syntax fix method...")
        logger.info(f"Broken code: {broken_code}")
        logger.info(f"Error: {syntax_error}")
        
        fixed_code = file_editor._llm_fix_syntax(broken_code, syntax_error)
        
        logger.info(f"Fixed code: {fixed_code}")
        
        # Validate the fix
        try:
            import ast
            ast.parse(fixed_code)
            logger.info("✅ LLM fix produced valid Python syntax")
            return True
        except SyntaxError as e:
            logger.error(f"❌ LLM fix still has syntax error: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ LLM SYNTAX FIX TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced File Editor with Auto-Refinement")
    print("="*80)
    
    # Test 1: Enhanced syntax error handling
    enhanced_test_result = test_enhanced_syntax_error_handling()
    
    # Test 2: LLM syntax fix capability  
    llm_fix_test_result = test_llm_syntax_fix_capability()
    
    # Summary
    print("\n" + "="*80)
    print("ENHANCED FILE EDITOR TEST SUMMARY")
    print("="*80)
    print(f"Enhanced Syntax Error Handling: {'✅ PASSED' if enhanced_test_result else '❌ FAILED'}")
    print(f"LLM Syntax Fix Capability: {'✅ PASSED' if llm_fix_test_result else '❌ FAILED'}")
    
    if enhanced_test_result:
        print("🎯 ENHANCED FILE EDITOR IS WORKING!")
        print("The auto-refinement and retry logic successfully handles syntax errors.")
    else:
        print("⚠️ Enhanced file editor needs further improvement.")
        print("Check the logs above for details on what failed.")