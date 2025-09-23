#!/usr/bin/env python3
"""
Test Enhanced SafeFileEditor Auto-Refinement
"""

import os
import sys
import tempfile
import logging

# Add path for imports
sys.path.append('.')
# Since we're already in the automated_debugging_strategy directory, use direct import

def test_enhanced_editor():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('EnhancedTest')
    
    logger.info("Testing Enhanced SafeFileEditor Auto-Refinement")
    logger.info("=" * 60)
    
    try:
        from automated_debugging_strategy.automated_file_editor import SafeFileEditor
        
        # Create a test file
        test_code = '''def hello():
    print("Hello World")
    return True

def main():
    hello()
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        logger.info(f"Created test file: {test_file}")
        
        # Initialize enhanced editor
        editor = SafeFileEditor(
            validate_syntax=True,
            create_backups=True,
            use_serena=False
        )
        
        logger.info("Testing enhanced line replacement with auto-refinement...")
        
        # Test with a line that might cause indentation issues
        new_line = "        print('Modified with wrong indentation')"
        
        result = editor.apply_line_replacement(test_file, 2, new_line)
        
        if result.success:
            logger.info("✅ Enhanced line replacement succeeded!")
            logger.info(f"Changes made: {result.changes_made}")
            if result.diff:
                logger.info(f"Diff: {result.diff[:200]}...")
        else:
            logger.error(f"❌ Enhanced line replacement failed: {result.error}")
        
        # Check final file
        with open(test_file, 'r') as f:
            final_content = f.read()
        
        logger.info("Final file content:")
        logger.info("-" * 40)
        logger.info(final_content)
        logger.info("-" * 40)
        
        # Cleanup
        os.unlink(test_file)
        
        return result.success
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_enhanced_editor()
    print(f"Enhanced Editor Test: {'PASSED' if success else 'FAILED'}")