#!/usr/bin/env python3
"""
Test Script for Debugging Operation

This script specifically tests the debugging functionality on GridbotBackup.py
to verify that syntax errors are properly detected and fixed.
"""

import os
import sys
import logging
from datetime import datetime

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the master automation pipeline
from master_automation_pipeline import MasterAutomationPipeline

def setup_test_logging():
    """Set up logging for the test"""
    log_dir = "test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'debug_test_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('DebugTest')
    logger.info(f"Test logging initialized - log file: {log_file}")
    return logger

def verify_syntax_error_exists(file_path):
    """Verify that GridbotBackup.py has the intentional syntax error"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the file
        try:
            compile(content, file_path, 'exec')
            return False, "No syntax error found"
        except SyntaxError as e:
            return True, f"Syntax error found: Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def test_debug_operation():
    """Test the debugging operation on GridbotBackup.py"""
    logger = setup_test_logging()
    
    logger.info("="*80)
    logger.info("STARTING DEBUG OPERATION TEST")
    logger.info("="*80)
    
    # Define target file
    target_file = "GridbotBackup.py"
    target_path = os.path.join(current_dir, target_file)
    
    logger.info(f"Target file: {target_path}")
    
    # Verify the file exists
    if not os.path.exists(target_path):
        logger.error(f"Target file not found: {target_path}")
        return False
    
    # Check if syntax error exists before debugging
    logger.info("Phase 1: Checking for syntax errors before debugging...")
    has_error, error_msg = verify_syntax_error_exists(target_path)
    
    if has_error:
        logger.info(f"[CONFIRMED] Syntax error detected: {error_msg}")
    else:
        logger.warning(f"[WARNING] No syntax error found: {error_msg}")
        logger.warning("Test may not demonstrate debugging capability properly")
    
    # Create pipeline with minimal configuration for debugging only
    logger.info("Phase 2: Initializing automation pipeline...")
    
    test_config = {
        'target_files': [target_file],
        'enable_debugging': True,
        'enable_optimization': False,  # Disable optimization for focused test
        'enable_analysis': False,      # Disable analysis for focused test
        'auto_apply_fixes': True,      # Enable auto-apply for testing
        'max_retries': 3,
        'timeout': 300,
        'backup_files': True
    }
    
    try:
        pipeline = MasterAutomationPipeline(test_config)
        logger.info("[OK] Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"[FAIL] Failed to initialize pipeline: {e}")
        return False
    
    # Run debugging phase specifically
    logger.info("Phase 3: Running debugging operation...")
    logger.info("-" * 60)
    
    try:
        # Run only the debug phase
        debug_results = pipeline.run_debug_phase([target_path])
        
        logger.info("-" * 60)
        logger.info("Phase 4: Analyzing debug results...")
        
        if debug_results:
            successful_sessions = [session for session in debug_results if session.success]
            failed_sessions = [session for session in debug_results if not session.success]
            
            logger.info(f"Debug sessions completed: {len(debug_results)}")
            logger.info(f"Successful sessions: {len(successful_sessions)}")
            logger.info(f"Failed sessions: {len(failed_sessions)}")
            
            # Detailed analysis of each session
            for i, session in enumerate(debug_results, 1):
                logger.info(f"\nSession {i} Results:")
                logger.info(f"  Target file: {session.target_file}")
                logger.info(f"  Success: {session.success}")
                logger.info(f"  Fixes applied: {len(session.fixes_applied)}")
                
                if session.fixes_applied:
                    logger.info("  Fix details:")
                    for j, fix in enumerate(session.fixes_applied, 1):
                        logger.info(f"    {j}. Type: {fix.get('type', 'unknown')}")
                        logger.info(f"       Success: {fix.get('success', False)}")
                        if 'error' in fix:
                            logger.info(f"       Error: {fix['error']}")
                
                if not session.success and session.error_message:
                    logger.error(f"  Error: {session.error_message}")
        
        else:
            logger.error("[FAIL] No debug results returned")
            return False
    
    except Exception as e:
        logger.error(f"[FAIL] Debug operation failed: {e}")
        import traceback
        logger.error(f"Exception details: {traceback.format_exc()}")
        return False
    
    # Verify syntax error is fixed after debugging
    logger.info("Phase 5: Verifying syntax fix...")
    has_error_after, error_msg_after = verify_syntax_error_exists(target_path)
    
    if not has_error_after:
        logger.info("[SUCCESS] Syntax error has been fixed!")
        logger.info("File now compiles without syntax errors")
        test_success = True
    else:
        logger.error(f"[FAIL] Syntax error still exists: {error_msg_after}")
        test_success = False
    
    # Final summary
    logger.info("="*80)
    logger.info("DEBUG OPERATION TEST COMPLETE")
    logger.info("="*80)
    
    if test_success:
        logger.info("[SUCCESS] Debug operation test PASSED")
        logger.info("- Syntax error was detected")
        logger.info("- Debugging operation was executed")
        logger.info("- Syntax error was successfully fixed")
    else:
        logger.error("[FAIL] Debug operation test FAILED")
        logger.error("- Debugging operation did not fix the syntax error")
    
    return test_success

def main():
    """Main test execution"""
    print("GridBot Debug Operation Test")
    print("="*50)

    if success := test_debug_operation():
        print("\n✅ TEST PASSED: Debug operation working correctly")
        return 0
    else:
        print("\n❌ TEST FAILED: Debug operation needs improvement")
        return 1

if __name__ == "__main__":
    exit(main())