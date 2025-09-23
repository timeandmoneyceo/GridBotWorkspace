import os
import time
from automated_debugging_strategy.master_automation_pipeline import MasterAutomationPipeline


def main():
    # Configurable defaults; edit as needed
    config = {
        "test_mode": False,
        "asm_interval_sec": 30,
        # Optionally list explicit targets; otherwise defaults are used
        # "autonomous_targets": [
        #     os.path.join(os.getcwd(), "GridbotBackup.py"),
        #     os.path.join(os.getcwd(), "gridbot_websocket_server.py"),
        # ],
    }

    pipeline = MasterAutomationPipeline(config)
    pipeline.start_autonomous_mode()
    print("[RUNNER] Autonomous Strategy Manager started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[RUNNER] Stopping...")
        pipeline.stop_autonomous_mode()
        print("[RUNNER] Stopped.")


if __name__ == "__main__":
    main()
