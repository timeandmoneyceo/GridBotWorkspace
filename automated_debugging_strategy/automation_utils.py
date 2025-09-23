"""
Utility Functions for Automation System

This module provides common utility functions used across the automation system.
"""

import os
import sys
import json
import logging
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import tempfile
import importlib.util

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}") from e

def save_config(config: Dict, config_path: str):
    """Save configuration to JSON file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save configuration: {e}") from e

def check_python_executable(python_path: str) -> bool:
    """Check if a Python executable is valid"""
    try:
        result = subprocess.run(
            [python_path, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists and is readable"""
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)

def create_backup_directory(backup_dir: str) -> bool:
    """Create backup directory if it doesn't exist"""
    try:
        os.makedirs(backup_dir, exist_ok=True)
        return True
    except Exception:
        return False

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def get_file_modification_time(file_path: str) -> datetime:
    """Get file modification time"""
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp)
    except Exception:
        return datetime.min

def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Validate Python syntax using compile()"""
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Compilation error: {e}"

def extract_imports_from_file(file_path: str) -> List[str]:
    """Extract import statements from a Python file"""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
    except Exception:
        pass
    return imports

def check_module_availability(module_name: str) -> bool:
    """Check if a Python module is available"""
    try:
        importlib.util.find_spec(module_name)
        return True
    except ImportError:
        return False

def run_python_code_safely(code: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """Run Python code safely in a subprocess"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    except subprocess.TimeoutExpired:
        return False, "", "Code execution timed out"
    except Exception as e:
        return False, "", f"Execution error: {e}"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string"""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        kb = size_bytes / 1024
        return f"{kb:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        mb = size_bytes / (1024 * 1024)
        return f"{mb:.2f} MB"
    else:
        gb = size_bytes / (1024 * 1024 * 1024)
        return f"{gb:.2f} GB"

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe filesystem usage"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x80-\x9f]', '', filename)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename

def create_timestamped_filename(base_name: str, extension: str = '') -> str:
    """Create a filename with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if extension and not extension.startswith('.'):
        extension = f'.{extension}'
    return f"{base_name}_{timestamp}{extension}"

def copy_file_with_backup(source: str, destination: str, backup_dir: str = None) -> bool:
    """Copy file with automatic backup of destination if it exists"""
    try:
        # Create backup if destination exists
        if os.path.exists(destination) and backup_dir:
            create_backup_directory(backup_dir)
            backup_name = os.path.basename(destination)
            backup_path = os.path.join(backup_dir, create_timestamped_filename(backup_name))
            shutil.copy2(destination, backup_path)
        
        # Copy the file
        shutil.copy2(source, destination)
        return True
        
    except Exception:
        return False

def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'python_executable': sys.executable,
        'working_directory': os.getcwd(),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        import psutil
        info['memory_total'] = psutil.virtual_memory().total
        info['memory_available'] = psutil.virtual_memory().available
        info['cpu_count'] = psutil.cpu_count()
    except ImportError:
        pass
    
    return info

def cleanup_temporary_files(file_patterns: List[str], max_age_hours: int = 24):
    """Clean up temporary files matching patterns"""
    import glob
    import time
    
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600)
    
    for pattern in file_patterns:
        try:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
        except Exception:
            pass

def monitor_system_resources() -> Dict[str, float]:
    """Monitor current system resource usage"""
    resources = {}
    
    try:
        import psutil
        
        # CPU usage
        resources['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        resources['memory_percent'] = memory.percent
        resources['memory_used_gb'] = memory.used / (1024**3)
        resources['memory_total_gb'] = memory.total / (1024**3)
        
        # Disk usage
        disk = psutil.disk_usage('.')
        resources['disk_percent'] = (disk.used / disk.total) * 100
        resources['disk_free_gb'] = disk.free / (1024**3)
        
    except ImportError:
        pass
    except Exception:
        pass
    
    return resources

class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1, item_description: str = None):
        """Update progress"""
        self.current_item += increment
        
        if self.total_items > 0:
            percentage = (self.current_item / self.total_items) * 100
            elapsed = datetime.now() - self.start_time
            
            # Estimate remaining time
            if self.current_item > 0:
                time_per_item = elapsed.total_seconds() / self.current_item
                remaining_items = self.total_items - self.current_item
                eta_seconds = time_per_item * remaining_items
                eta = format_duration(eta_seconds)
            else:
                eta = "Unknown"
            
            status = f"{self.description}: {self.current_item}/{self.total_items} ({percentage:.1f}%) - ETA: {eta}"
            
            if item_description:
                status += f" - {item_description}"
            
            print(f"\r{status}", end="", flush=True)
    
    def finish(self):
        """Mark progress as complete"""
        elapsed = datetime.now() - self.start_time
        print(f"\n{self.description} completed in {format_duration(elapsed.total_seconds())}")