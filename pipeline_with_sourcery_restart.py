#!/usr/bin/env python3
"""
Pipeline Wrapper with AI Workspace Doctor Restart Support

This wrapper script handles the complete AI Workspace Doctor + Pipeline integration:
1. Runs the master automation pipeline
2. If AI Workspace Doctor applies changes (exit code 42), restarts the pipeline
3. Continues until no more AI Workspace Doctor changes are needed
4. Prevents infinite restart loops

Usage:
    python pipeline_with_ai_doctor_restart.py [pipeline args...]
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def run_pipeline_with_restart(args):
    """Run the pipeline with automatic restart support for AI Workspace Doctor changes"""
    script_dir = Path(__file__).parent
    pipeline_script = script_dir / "master_automation_pipeline.py"

    max_restarts = 3  # Prevent infinite loops
    restart_count = 0

    print("=" * 80)
    print("PIPELINE WITH AI WORKSPACE DOCTOR RESTART SUPPORT")
    print("=" * 80)
    print(f"Max restarts: {max_restarts}")
    print(f"Pipeline script: {pipeline_script}")
    print(f"Args: {args}")
    print("=" * 80)

    while restart_count < max_restarts:
        if restart_count > 0:
            print(f"\n🔄 RESTART #{restart_count} - AI Workspace Doctor applied changes")
            print("=" * 60)
            # Small delay to ensure file system changes are settled
            time.sleep(2)

        # Prepare command
        cmd = [sys.executable, str(pipeline_script)] + args

        print(f"Running: {' '.join(cmd)}")
        print(f"Restart count: {restart_count}/{max_restarts}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 40)

        try:
            # Run the pipeline
            result = subprocess.run(cmd, check=False)
            exit_code = result.returncode

            print(f"\nPipeline exited with code: {exit_code}")

            if exit_code == 42:
                # AI Workspace Doctor applied changes - restart needed
                restart_count += 1
                print(f"🔄 AI Workspace Doctor restart requested (attempt {restart_count}/{max_restarts})")

                # Check for restart marker file
                restart_marker = Path.cwd() / '.ai_doctor_restart_pending'
                if restart_marker.exists():
                    print(f"Found restart marker: {restart_marker}")
                    try:
                        print("Marker content:")
                        print(restart_marker.read_text())
                        restart_marker.unlink()  # Remove marker
                    except Exception as e:
                        print(f"Warning: Could not process restart marker: {e}")

                if restart_count < max_restarts:
                    print("Restarting pipeline with fresh AI Workspace Doctor changes...")
                    continue
                else:
                    print("⚠️  Maximum restarts reached - stopping to prevent infinite loop")
                    print("This usually means AI Workspace Doctor keeps finding new issues to fix.")
                    print("Consider reviewing the changes manually or adjusting AI doctor config.")
                    return 1

            elif exit_code == 0:
                # Successful completion
                print("✅ Pipeline completed successfully!")
                return 0

            else:
                # Other error
                print(f"❌ Pipeline failed with exit code {exit_code}")
                return exit_code

        except KeyboardInterrupt:
            print("\n🛑 Pipeline interrupted by user")
            return 130

        except Exception as e:
            print(f"❌ Error running pipeline: {e}")
            return 1

    # Should not reach here
    print("❌ Unexpected end of restart loop")
    return 1

def main():
    """Main entry point"""
    args = sys.argv[1:]  # Pass through all arguments to the pipeline

    # If first arg is --help, show help for both this script and the pipeline
    if args and args[0] in ('--help', '-h'):
        print(__doc__)
        print("\nUnderlying pipeline options:")
        print("-" * 40)
        # Run pipeline help
        try:
            subprocess.run([sys.executable, str(Path(__file__).parent / "master_automation_pipeline.py"), "--help"])
        except Exception:
            print("Could not display pipeline help")
        return 0

    return run_pipeline_with_restart(args)

if __name__ == "__main__":
    sys.exit(main())