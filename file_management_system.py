"""
File Management System Module

This module manages file cleanup to prevent backups, logs, and reports from growing out of control.
It maintains a reasonable number of recent files while removing older ones.
"""

import os
import glob
import time
import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

class FileManagementSystem:
    """Manages file cleanup and organization for automation system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.setup_logging()
        
    def get_default_config(self) -> Dict:
        """Get default file management configuration"""
        return {
            # Backup management
            'max_backup_files': 20,  # Keep last 20 backup files per original file
            'backup_retention_days': 7,  # Keep backups for 7 days
            
            # Log file management
            'max_log_files': 10,  # Keep last 10 log files
            'log_retention_days': 3,  # Keep logs for 3 days
            'max_log_size_mb': 50,  # Archive logs larger than 50MB
            
            # Report and session management
            'max_session_files': 15,  # Keep last 15 session files
            'max_report_files': 15,   # Keep last 15 report files
            'session_retention_days': 5,  # Keep sessions for 5 days
            
            # Summary and optimization files
            'max_summary_files': 10,  # Keep last 10 summary files
            'summary_retention_days': 7,  # Keep summaries for 7 days
            
            # Sourcery workspace doctor files
            'max_sourcery_files': 10,  # Keep last 10 Sourcery reports
            'sourcery_retention_days': 7,  # Keep Sourcery files for 7 days
            
            # Temporary file cleanup
            'temp_file_age_hours': 24,  # Remove temp files older than 24 hours
            
            # Directories to manage
            'managed_directories': [
                'backups',
                '.',  # Current directory for logs
                'reports',
                'sessions',
                'temp'
            ],
            
            # File patterns to manage
            'file_patterns': {
                'backups': ['*.backup.*'],
                'logs': [
                    '*.log', '*.log.*',
                    'master_automation.log',
                    'debug_orchestrator.log',
                    'debug_log_parser.log'
                ],
                'sessions': [
                    'automation_session_*.json',
                    'continuous_automation_report_*.json'
                ],
                'reports': [
                    'automation_report_*.json',
                    'debug_report_*.json',
                    'optimization_report_*.json',
                    'optimization_report_enhanced_*.json'
                ],
                'summaries': [
                    'summary_iteration_*.txt', '*summary*.txt', '*_summary.txt',
                    'ai_strategy_report_*.md',
                    'optimization_analysis_report.md'
                ],
                'sourcery_files': [
                    'sourcery_summary_*.md',
                    'sourcery_changes_*.patch',
                    'sourcery_review_*.txt',
                    'sourcery_review_*.json',
                    'sourcery_diffs_*.patch'
                ],
                'temp': [
                    'temp_*', '*.tmp', '*.temp',
                    '.sourcery_restart_pending',
                    'improvement_tracking.json'
                ]
            }
        }
    
    def setup_logging(self):
        """Setup logging for file management system"""
        class FileManagerFormatter(logging.Formatter):
            def format(self, record):
                timestamp = self.formatTime(record, '%H:%M:%S')
                if record.levelname == 'INFO':
                    return f"[{timestamp}] CLEANUP: {record.getMessage()}"
                elif record.levelname == 'WARNING':
                    return f"[{timestamp}] CLEANUP WARN: {record.getMessage()}"
                elif record.levelname == 'ERROR':
                    return f"[{timestamp}] CLEANUP ERROR: {record.getMessage()}"
                else:
                    return f"[{timestamp}] CLEANUP {record.levelname}: {record.getMessage()}"
        
        # Setup console handler with custom formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(FileManagerFormatter())
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def run_cleanup(self) -> Dict:
        """Run comprehensive file cleanup at start of automation"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING FILE MANAGEMENT CLEANUP")
        self.logger.info("=" * 80)
        
        cleanup_stats = {
            'files_removed': 0,
            'files_archived': 0,
            'directories_cleaned': 0,
            'space_freed_mb': 0,
            'cleanup_time': 0
        }
        
        start_time = time.time()
        
        try:
            # 1. Clean backup files
            backup_stats = self.cleanup_backup_files()
            cleanup_stats['files_removed'] += backup_stats['removed']
            cleanup_stats['space_freed_mb'] += backup_stats['space_freed_mb']
            
            # 2. Clean log files
            log_stats = self.cleanup_log_files()
            cleanup_stats['files_removed'] += log_stats['removed']
            cleanup_stats['files_archived'] += log_stats['archived']
            cleanup_stats['space_freed_mb'] += log_stats['space_freed_mb']
            
            # 3. Clean session and report files
            session_stats = self.cleanup_session_files()
            cleanup_stats['files_removed'] += session_stats['removed']
            cleanup_stats['space_freed_mb'] += session_stats['space_freed_mb']
            
            # 4. Clean Sourcery files
            sourcery_stats = self.cleanup_sourcery_files()
            cleanup_stats['files_removed'] += sourcery_stats['removed']
            cleanup_stats['space_freed_mb'] += sourcery_stats['space_freed_mb']
            
            # 5. Clean summary files
            summary_stats = self.cleanup_summary_files()
            cleanup_stats['files_removed'] += summary_stats['removed']
            cleanup_stats['space_freed_mb'] += summary_stats['space_freed_mb']
            
            # 6. Clean temporary files
            temp_stats = self.cleanup_temp_files()
            cleanup_stats['files_removed'] += temp_stats['removed']
            cleanup_stats['space_freed_mb'] += temp_stats['space_freed_mb']
            
            # 7. Create organized directory structure if needed
            self.ensure_directory_structure()
            
            cleanup_stats['cleanup_time'] = time.time() - start_time
            
            # Log cleanup summary
            self.logger.info("CLEANUP SUMMARY:")
            self.logger.info(f"   Files removed: {cleanup_stats['files_removed']}")
            self.logger.info(f"   Files archived: {cleanup_stats['files_archived']}")
            self.logger.info(f"   Space freed: {cleanup_stats['space_freed_mb']:.1f} MB")
            self.logger.info(f"   Cleanup time: {cleanup_stats['cleanup_time']:.2f}s")
            self.logger.info("=" * 80)
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            cleanup_stats['cleanup_time'] = time.time() - start_time
            return cleanup_stats
    
    def cleanup_backup_files(self) -> Dict:
        """Clean up old backup files"""
        self.logger.info("Cleaning backup files...")

        stats = {'removed': 0, 'space_freed_mb': 0}
        backup_dir = self.config.get('backup_dir', 'backups')

        if not os.path.exists(backup_dir):
            self.logger.info("   No backup directory found")
            return stats

        # Group backups by original file
        backup_groups = {}

        for pattern in self.config['file_patterns']['backups']:
            for backup_file in glob.glob(os.path.join(backup_dir, pattern)):
                # Extract original filename from backup
                basename = os.path.basename(backup_file)
                if '.backup.' in basename:
                    original_name = basename.split('.backup.')[0]
                    if original_name not in backup_groups:
                        backup_groups[original_name] = []
                    backup_groups[original_name].append(backup_file)

        # Clean each group
        for backup_files in backup_groups.values():
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

            max_files = self.config['max_backup_files']
            retention_days = self.config['backup_retention_days']
            cutoff_time = time.time() - (retention_days * 24 * 3600)

            # Remove excess files or old files
            for i, backup_file in enumerate(backup_files):
                should_remove = False
                reason = ""

                if i >= max_files:
                    should_remove = True
                    reason = f"excess (>{max_files})"
                elif os.path.getmtime(backup_file) < cutoff_time:
                    should_remove = True
                    reason = f"old (>{retention_days} days)"

                if should_remove:
                    try:
                        file_size = os.path.getsize(backup_file) / (1024 * 1024)  # MB
                        os.remove(backup_file)
                        stats['removed'] += 1
                        stats['space_freed_mb'] += file_size
                        self.logger.info(f"   Removed {os.path.basename(backup_file)} ({reason})")
                    except Exception as e:
                        self.logger.warning(f"   Failed to remove {backup_file}: {e}")

        self.logger.info(f"   Backup cleanup: {stats['removed']} files removed, {stats['space_freed_mb']:.1f} MB freed")
        return stats
    
    def cleanup_log_files(self) -> Dict:
        """Clean up old log files"""
        self.logger.info("Cleaning log files...")
        
        stats = {'removed': 0, 'archived': 0, 'space_freed_mb': 0}
        retention_days = self.config['log_retention_days']
        max_size_mb = self.config['max_log_size_mb']
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        for pattern in self.config['file_patterns']['logs']:
            for log_file in glob.glob(pattern):
                try:
                    file_size_mb = os.path.getsize(log_file) / (1024 * 1024)
                    file_age = os.path.getmtime(log_file)
                    
                    # Archive large files
                    if file_size_mb > max_size_mb:
                        archived_name = f"{log_file}.archived.{int(time.time())}"
                        os.rename(log_file, archived_name)
                        stats['archived'] += 1
                        self.logger.info(f"   Archived {log_file} (size: {file_size_mb:.1f}MB)")
                        continue
                    
                    # Remove old files
                    if file_age < cutoff_time:
                        os.remove(log_file)
                        stats['removed'] += 1
                        stats['space_freed_mb'] += file_size_mb
                        self.logger.info(f"   Removed {log_file} (age: {(time.time()-file_age)/86400:.1f} days)")
                        
                except Exception as e:
                    self.logger.warning(f"   Failed to process {log_file}: {e}")
        
        self.logger.info(f"   Log cleanup: {stats['removed']} removed, {stats['archived']} archived, {stats['space_freed_mb']:.1f} MB freed")
        return stats
    
    def cleanup_session_files(self) -> Dict:
        """Clean up old session and report files"""
        self.logger.info("Cleaning session and report files...")
        
        stats = {'removed': 0, 'space_freed_mb': 0}
        
        # Clean session files
        session_files = []
        for pattern in self.config['file_patterns']['sessions']:
            session_files.extend(glob.glob(pattern))
        
        # Clean report files
        report_files = []
        for pattern in self.config['file_patterns']['reports']:
            report_files.extend(glob.glob(pattern))
        
        # Process each file type
        file_groups = [
            ('sessions', session_files, self.config['max_session_files']),
            ('reports', report_files, self.config['max_report_files'])
        ]
        
        retention_days = self.config['session_retention_days']
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        for file_type, files, max_files in file_groups:
            # Sort by modification time (newest first)
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            
            for i, file_path in enumerate(files):
                should_remove = False
                reason = ""
                
                if i >= max_files:
                    should_remove = True
                    reason = f"excess (>{max_files})"
                elif os.path.getmtime(file_path) < cutoff_time:
                    should_remove = True
                    reason = f"old (>{retention_days} days)"
                
                if should_remove:
                    try:
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        os.remove(file_path)
                        stats['removed'] += 1
                        stats['space_freed_mb'] += file_size
                        self.logger.info(f"   Removed {file_type}: {os.path.basename(file_path)} ({reason})")
                    except Exception as e:
                        self.logger.warning(f"   Failed to remove {file_path}: {e}")
        
        self.logger.info(f"   Session/Report cleanup: {stats['removed']} files removed, {stats['space_freed_mb']:.1f} MB freed")
        return stats
    
    def cleanup_summary_files(self) -> Dict:
        """Clean up old summary files"""
        self.logger.info("Cleaning summary files...")
        
        stats = {'removed': 0, 'space_freed_mb': 0}
        
        summary_files = []
        for pattern in self.config['file_patterns']['summaries']:
            summary_files.extend(glob.glob(pattern))
        
        # Sort by modification time (newest first)
        summary_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        
        max_files = self.config['max_summary_files']
        retention_days = self.config['summary_retention_days']
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        for i, summary_file in enumerate(summary_files):
            should_remove = False
            reason = ""
            
            if i >= max_files:
                should_remove = True
                reason = f"excess (>{max_files})"
            elif os.path.getmtime(summary_file) < cutoff_time:
                should_remove = True
                reason = f"old (>{retention_days} days)"
            
            if should_remove:
                try:
                    file_size = os.path.getsize(summary_file) / (1024 * 1024)  # MB
                    os.remove(summary_file)
                    stats['removed'] += 1
                    stats['space_freed_mb'] += file_size
                    self.logger.info(f"   Removed summary: {os.path.basename(summary_file)} ({reason})")
                except Exception as e:
                    self.logger.warning(f"   Failed to remove {summary_file}: {e}")
        
        self.logger.info(f"   Summary cleanup: {stats['removed']} files removed, {stats['space_freed_mb']:.1f} MB freed")
        return stats
    
    def cleanup_temp_files(self) -> Dict:
        """Clean up temporary files"""
        self.logger.info("Cleaning temporary files...")
        
        stats = {'removed': 0, 'space_freed_mb': 0}
        temp_age_hours = self.config['temp_file_age_hours']
        cutoff_time = time.time() - (temp_age_hours * 3600)
        
        # Check current directory and temp directory
        search_dirs = ['.', 'temp'] if os.path.exists('temp') else ['.']
        
        for search_dir in search_dirs:
            for pattern in self.config['file_patterns']['temp']:
                search_pattern = os.path.join(search_dir, pattern)
                for temp_file in glob.glob(search_pattern):
                    try:
                        if os.path.getmtime(temp_file) < cutoff_time:
                            file_size = os.path.getsize(temp_file) / (1024 * 1024)  # MB
                            os.remove(temp_file)
                            stats['removed'] += 1
                            stats['space_freed_mb'] += file_size
                            self.logger.info(f"   Removed temp: {os.path.basename(temp_file)} (age: {temp_age_hours}+ hours)")
                    except Exception as e:
                        self.logger.warning(f"   Failed to remove {temp_file}: {e}")
        
        self.logger.info(f"   Temp cleanup: {stats['removed']} files removed, {stats['space_freed_mb']:.1f} MB freed")
        return stats
    
    def cleanup_sourcery_files(self) -> Dict:
        """Clean up Sourcery workspace doctor files"""
        self.logger.info("Cleaning Sourcery workspace doctor files...")
        
        stats = {'removed': 0, 'space_freed_mb': 0}
        max_files = self.config.get('max_sourcery_files', 10)
        retention_days = self.config.get('sourcery_retention_days', 7)
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        # Handle different locations for Sourcery files
        locations = [
            '.',  # Current directory
            'reports',  # Reports directory
            'automation_logs'  # Logs directory
        ]
        
        files = []
        for location in locations:
            for pattern in self.config['file_patterns']['sourcery_files']:
                pattern_path = os.path.join(location, pattern) if location != '.' else pattern
                files.extend(glob.glob(pattern_path))
        
        if files:
            # Sort by modification time (newest first)
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            
            for i, file_path in enumerate(files):
                should_remove = False
                reason = ""
                
                if i >= max_files:
                    should_remove = True
                    reason = f"excess (>{max_files})"
                elif os.path.getmtime(file_path) < cutoff_time:
                    should_remove = True
                    reason = f"old (>{retention_days} days)"
                
                if should_remove:
                    try:
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        os.remove(file_path)
                        stats['removed'] += 1
                        stats['space_freed_mb'] += file_size
                        self.logger.info(f"   Removed Sourcery file: {os.path.basename(file_path)} ({reason})")
                    except Exception as e:
                        self.logger.warning(f"   Failed to remove {file_path}: {e}")
        
        self.logger.info(f"   Sourcery cleanup: {stats['removed']} files removed, {stats['space_freed_mb']:.1f} MB freed")
        return stats
    
    def ensure_directory_structure(self):
        """Ensure organized directory structure exists"""
        self.logger.info("Ensuring directory structure...")
        
        directories_to_create = [
            'backups',
            'reports', 
            'sessions',
            'temp'
        ]
        
        for directory in directories_to_create:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    self.logger.info(f"   Created directory: {directory}")
                except Exception as e:
                    self.logger.warning(f"   Failed to create {directory}: {e}")
    
    def get_disk_usage_summary(self) -> Dict:
        """Get summary of disk usage by file type"""
        summary = {
            'backups': {'count': 0, 'size_mb': 0},
            'logs': {'count': 0, 'size_mb': 0},
            'sessions': {'count': 0, 'size_mb': 0},
            'reports': {'count': 0, 'size_mb': 0},
            'summaries': {'count': 0, 'size_mb': 0},
            'temp': {'count': 0, 'size_mb': 0}
        }

        # Count files by type
        for file_type, patterns in self.config['file_patterns'].items():
            if file_type == 'backups':
                # Special handling for backup directory
                backup_dir = self.config.get('backup_dir', 'backups')
                if os.path.exists(backup_dir):
                    for pattern in patterns:
                        for file_path in glob.glob(os.path.join(backup_dir, pattern)):
                            try:
                                summary[file_type]['count'] += 1
                                summary[file_type]['size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
                            except Exception:
                                pass
            else:
                for pattern in patterns:
                    for file_path in glob.glob(pattern):
                        try:
                            summary[file_type]['count'] += 1
                            summary[file_type]['size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
                        except:
                            pass

        return summary
    
    def log_disk_usage_report(self):
        """Log current disk usage by file type"""
        usage = self.get_disk_usage_summary()
        
        self.logger.info("CURRENT DISK USAGE BY FILE TYPE:")
        self.logger.info("-" * 50)
        
        total_files = 0
        total_size = 0
        
        for file_type, stats in usage.items():
            if stats['count'] > 0:
                self.logger.info(f"   {file_type.capitalize():12}: {stats['count']:3d} files, {stats['size_mb']:6.1f} MB")
                total_files += stats['count']
                total_size += stats['size_mb']
        
        self.logger.info("-" * 50)
        self.logger.info(f"   {'Total':12}: {total_files:3d} files, {total_size:6.1f} MB")
        self.logger.info("-" * 50)

def main():
    """Test file management system"""
    file_manager = FileManagementSystem()
    
    # Show current usage
    file_manager.log_disk_usage_report()
    
    # Run cleanup
    stats = file_manager.run_cleanup()
    
    # Show usage after cleanup
    print()
    file_manager.log_disk_usage_report()
    
    return stats

if __name__ == "__main__":
    main()