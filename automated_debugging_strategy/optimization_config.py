"""
Optimization System Upgrade Configuration
Performance enhancements and efficiency improvements based on analysis of production runs
"""

class OptimizationConfig:
    """Enhanced configuration for optimization system performance"""
    
    # Priority-based processing limits
    MAX_CANDIDATES_PER_RUN = 10  # Process top 10 candidates per file
    HIGH_PRIORITY_THRESHOLD = 6  # Auto-apply optimizations with priority >= 6
    
    # Timeout management (in seconds)
    HIGH_PRIORITY_TIMEOUT = 900   # 15 minutes for critical optimizations
    MEDIUM_PRIORITY_TIMEOUT = 600 # 10 minutes for normal optimizations
    LOW_PRIORITY_TIMEOUT = 300    # 5 minutes for minor optimizations
    
    # Auto-application criteria
    AUTO_APPLY_PATTERNS = [
        "string concatenation",
        "list comprehension", 
        "loop optimization",
        "memory usage",
        "performance improvement"
    ]
    
    # Minimum optimization size for auto-application
    MIN_OPTIMIZATION_SIZE = 500  # characters
    
    # Efficiency thresholds
    TARGET_SUCCESS_RATE = 80.0    # Target 80% success rate
    MIN_EFFICIENCY_RATE = 60.0    # Minimum 60% application rate
    
    # Resource management
    MAX_CONCURRENT_OPTIMIZATIONS = 1  # Process one at a time for stability
    BACKUP_RETENTION_DAYS = 30        # Keep backups for 30 days
    
    # Reporting enhancements
    GENERATE_DETAILED_REPORTS = True
    INCLUDE_PERFORMANCE_METRICS = True
    SAVE_OPTIMIZATION_DIFFS = True

class OptimizationMetrics:
    """Track optimization system performance metrics"""
    
    def __init__(self):
        self.total_runs = 0
        self.total_candidates = 0
        self.total_applied = 0
        self.total_successful = 0
        self.average_response_time = 0.0
        self.efficiency_rate = 0.0
        
    def update_metrics(self, candidates_found: int, applied: int, successful: int, response_time: float):
        """Update metrics with latest run data"""
        self.total_runs += 1
        self.total_candidates += candidates_found
        self.total_applied += applied
        self.total_successful += successful
        
        # Calculate rolling average response time
        self.average_response_time = ((self.average_response_time * (self.total_runs - 1)) + response_time) / self.total_runs
        
        # Calculate efficiency rate
        self.efficiency_rate = (self.total_applied / self.total_candidates * 100) if self.total_candidates > 0 else 0.0
        
    def get_summary(self) -> dict:
        """Get optimization metrics summary"""
        return {
            "total_runs": self.total_runs,
            "total_candidates": self.total_candidates,
            "total_applied": self.total_applied,
            "success_rate": f"{(self.total_successful / self.total_candidates * 100):.1f}%" if self.total_candidates > 0 else "0%",
            "efficiency_rate": f"{self.efficiency_rate:.1f}%",
            "average_response_time": f"{self.average_response_time:.2f}s"
        }

# Optimization quality scoring
OPTIMIZATION_SCORING = {
    "string_concatenation_fix": 10,
    "loop_optimization": 8,
    "memory_improvement": 7,
    "error_handling": 6,
    "code_readability": 5,
    "performance_enhancement": 9,
    "algorithm_improvement": 10
}

# File-specific optimization strategies
FILE_STRATEGIES = {
    "gridbot_websocket_server.py": {
        "focus": ["async_optimization", "websocket_efficiency", "database_operations"],
        "priority_boost": 2,  # Increase priority for server optimizations
        "max_candidates": 5   # Limit candidates for complex files
    },
    "GridbotBackup.py": {
        "focus": ["ml_optimization", "data_processing", "financial_calculations"],
        "priority_boost": 1,
        "max_candidates": 8
    }
}