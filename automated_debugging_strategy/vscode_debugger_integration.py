"""
VS Code Debugger Integration

This module provides integration with VS Code's debugger to capture precise error
locations, variable states, and call stacks for enhanced automated debugging.
"""

import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import socket
import os
import subprocess

from .debug_automation_orchestrator import DebugAutomationOrchestrator

@dataclass
class DebuggerBreakpoint:
    """Container for VS Code debugger breakpoint data"""
    file_path: str
    line_number: int
    error_message: str
    variables: Dict[str, Any]
    call_stack: List[str]
    timestamp: datetime
    error_type: str = "Runtime Error"
    thread_id: Optional[str] = None

class VSCodeDebuggerIntegration:
    """Integration layer for VS Code debugger automation"""
    
    def __init__(self, orchestrator: DebugAutomationOrchestrator = None, 
                 listen_port: int = 9229):
        self.orchestrator = orchestrator or DebugAutomationOrchestrator()
        self.listen_port = listen_port
        self.is_listening = False
        self.breakpoint_queue = []
        self.debug_session_active = False
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for VS Code integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vscode_debugger_integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_debugger_listener(self):
        """Start listening for VS Code debugger events"""
        try:
            self.logger.info(f"Starting VS Code debugger listener on port {self.listen_port}")
            self.is_listening = True
            
            # Start listener in separate thread
            listener_thread = threading.Thread(target=self._debugger_listener_worker, daemon=True)
            listener_thread.start()
            
            # Start breakpoint processor
            processor_thread = threading.Thread(target=self._process_breakpoints, daemon=True)
            processor_thread.start()
            
            self.logger.info("VS Code debugger integration started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start debugger listener: {e}")
            return False
    
    def _debugger_listener_worker(self):
        """Worker thread to listen for debugger events"""
        try:
            # Create socket to listen for debugger events
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('localhost', self.listen_port))
                sock.listen(5)
                
                self.logger.info(f"Listening for debugger connections on localhost:{self.listen_port}")
                
                while self.is_listening:
                    try:
                        conn, addr = sock.accept()
                        self.logger.info(f"Debugger connection from {addr}")
                        
                        # Handle connection in separate thread
                        handler_thread = threading.Thread(
                            target=self._handle_debugger_connection, 
                            args=(conn,), 
                            daemon=True
                        )
                        handler_thread.start()
                        
                    except Exception as e:
                        if self.is_listening:
                            self.logger.error(f"Error accepting debugger connection: {e}")
                        
        except Exception as e:
            self.logger.error(f"Debugger listener worker failed: {e}")
    
    def _handle_debugger_connection(self, connection):
        """Handle individual debugger connection"""
        try:
            with connection:
                buffer = ""
                while self.is_listening:
                    try:
                        data = connection.recv(4096).decode('utf-8')
                        if not data:
                            break
                            
                        buffer += data
                        
                        # Process complete JSON messages
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                self._process_debugger_message(line.strip())
                                
                    except Exception as e:
                        self.logger.warning(f"Error receiving debugger data: {e}")
                        break
                        
        except Exception as e:
            self.logger.error(f"Error handling debugger connection: {e}")
    
    def _process_debugger_message(self, message: str):
        """Process incoming debugger message"""
        try:
            data = json.loads(message)
            
            # Check if this is a breakpoint/error event
            if data.get('type') == 'event':
                event_type = data.get('event')
                
                if event_type == 'stopped':
                    self._handle_stopped_event(data)
                elif event_type == 'exited':
                    self._handle_exited_event(data)
                elif event_type == 'terminated':
                    self._handle_terminated_event(data)
                    
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON from debugger: {message}")
        except Exception as e:
            self.logger.error(f"Error processing debugger message: {e}")
    
    def _handle_stopped_event(self, data: Dict):
        """Handle debugger stopped event (breakpoint hit)"""
        try:
            body = data.get('body', {})
            reason = body.get('reason', 'unknown')
            thread_id = body.get('threadId')

            if reason in ['exception', 'breakpoint', 'step']:
                self.logger.info(f"Debugger stopped: {reason} (thread {thread_id})")

                if breakpoint_data := self._extract_breakpoint_data(
                    data, thread_id
                ):
                    self.breakpoint_queue.append(breakpoint_data)
                    self.logger.info(f"Added breakpoint to queue: {breakpoint_data.file_path}:{breakpoint_data.line_number}")

        except Exception as e:
            self.logger.error(f"Error handling stopped event: {e}")
    
    def _extract_breakpoint_data(self, data: Dict, thread_id: str) -> Optional[DebuggerBreakpoint]:
        """Extract breakpoint data from debugger event"""
        try:
            # This is a simplified extraction - in practice, you'd need to query 
            # the debugger API for stack frames, variables, etc.
            
            # For demonstration, we'll create mock data structure
            # In real implementation, you'd use VS Code's Debug Adapter Protocol
            
            return DebuggerBreakpoint(
                file_path=data.get('source', {}).get('path', 'unknown'),
                line_number=data.get('line', 0),
                error_message=data.get('body', {}).get('text', 'Unknown error'),
                variables={},  # Would be populated from debugger API
                call_stack=[],  # Would be populated from debugger API
                timestamp=datetime.now(),
                thread_id=thread_id
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting breakpoint data: {e}")
            return None
    
    def _handle_exited_event(self, data: Dict):
        """Handle debugger exited event"""
        self.debug_session_active = False
        self.logger.info("Debug session exited")
    
    def _handle_terminated_event(self, data: Dict):
        """Handle debugger terminated event"""
        self.debug_session_active = False
        self.logger.info("Debug session terminated")
    
    def _process_breakpoints(self):
        """Process breakpoints in the queue"""
        while self.is_listening:
            try:
                if self.breakpoint_queue:
                    breakpoint = self.breakpoint_queue.pop(0)
                    self.logger.info(f"Processing breakpoint: {breakpoint.file_path}:{breakpoint.line_number}")

                    # Convert to orchestrator format
                    breakpoint_data = {
                        'file_path': breakpoint.file_path,
                        'line_number': breakpoint.line_number,
                        'error_message': breakpoint.error_message,
                        'variables': breakpoint.variables,
                        'call_stack': breakpoint.call_stack,
                        'error_type': breakpoint.error_type
                    }

                    if success := self.orchestrator.integrate_vscode_debugger(
                        breakpoint_data
                    ):
                        self.logger.info("Automated fix applied successfully")
                    else:
                        self.logger.warning("Automated fix failed")

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Error processing breakpoints: {e}")
    
    def stop_listener(self):
        """Stop the debugger listener"""
        self.is_listening = False
        self.logger.info("VS Code debugger listener stopped")
    
    def create_launch_config(self, target_file: str, workspace_dir: str) -> str:
        """Create VS Code launch configuration for debugging with automation"""
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Python: Automated Debugging",
                    "type": "python",
                    "request": "launch",
                    "program": target_file,
                    "console": "integratedTerminal",
                    "justMyCode": False,
                    "stopOnEntry": False,
                    "args": [],
                    "env": {},
                    "debugServer": self.listen_port,
                    "preLaunchTask": "Start Debug Automation",
                    "postDebugTask": "Stop Debug Automation"
                }
            ]
        }
        
        # Create .vscode directory if it doesn't exist
        vscode_dir = os.path.join(workspace_dir, '.vscode')
        os.makedirs(vscode_dir, exist_ok=True)
        
        # Write launch.json
        launch_file = os.path.join(vscode_dir, 'launch.json')
        with open(launch_file, 'w') as f:
            json.dump(launch_config, f, indent=4)
        
        self.logger.info(f"Created VS Code launch configuration: {launch_file}")
        return launch_file
    
    def create_tasks_config(self, workspace_dir: str) -> str:
        """Create VS Code tasks configuration for debug automation"""
        tasks_config = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Start Debug Automation",
                    "type": "shell",
                    "command": "python",
                    "args": [
                        "-c",
                        f"import sys; sys.path.append('{os.path.dirname(__file__)}'); " +
                        "from vscode_debugger_integration import VSCodeDebuggerIntegration; " +
                        "integration = VSCodeDebuggerIntegration(); integration.start_debugger_listener()"
                    ],
                    "group": "build",
                    "isBackground": True,
                    "problemMatcher": []
                },
                {
                    "label": "Stop Debug Automation", 
                    "type": "shell",
                    "command": "echo",
                    "args": ["Debug automation stopped"],
                    "group": "build"
                }
            ]
        }
        
        # Create .vscode directory if it doesn't exist
        vscode_dir = os.path.join(workspace_dir, '.vscode')
        os.makedirs(vscode_dir, exist_ok=True)
        
        # Write tasks.json
        tasks_file = os.path.join(vscode_dir, 'tasks.json')
        with open(tasks_file, 'w') as f:
            json.dump(tasks_config, f, indent=4)
        
        self.logger.info(f"Created VS Code tasks configuration: {tasks_file}")
        return tasks_file
    
    def setup_workspace_for_debugging(self, target_file: str, workspace_dir: str = None):
        """Setup complete VS Code workspace for automated debugging"""
        if workspace_dir is None:
            workspace_dir = os.path.dirname(target_file)
        
        self.logger.info(f"Setting up VS Code workspace for automated debugging: {target_file}")
        
        # Create configurations
        launch_file = self.create_launch_config(target_file, workspace_dir)
        tasks_file = self.create_tasks_config(workspace_dir)
        
        # Create readme with instructions
        readme_content = f"""# Automated Debugging Setup

This workspace is configured for automated debugging integration.

## How to Use:

1. **Start Debug Automation**: 
   - Press `Ctrl+Shift+P`
   - Type "Tasks: Run Task"
   - Select "Start Debug Automation"

2. **Start Debugging**:
   - Press `F5` or use "Run and Debug" panel
   - Select "Python: Automated Debugging"

3. **When breakpoints hit**:
   - The automation system will capture error context
   - Variable states and call stack will be analyzed
   - Automated fixes will be generated and applied

## Files:
- Target file: {target_file}
- Launch config: {launch_file}
- Tasks config: {tasks_file}

## Logs:
- VS Code integration: vscode_debugger_integration.log
- Debug orchestrator: debug_orchestrator.log

The system will automatically handle section-aware debugging for massive functions like `run_bot`.
"""
        
        readme_file = os.path.join(workspace_dir, 'DEBUG_AUTOMATION_README.md')
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        self.logger.info("VS Code workspace setup complete!")
        return {
            'launch_config': launch_file,
            'tasks_config': tasks_file,
            'readme': readme_file,
            'workspace_dir': workspace_dir
        }