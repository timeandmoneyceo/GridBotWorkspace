# GridBot Essential Commands and Operations

## Core Development Commands

### Virtual Environment Management
```powershell
# CRITICAL: Always activate virtual environment first
.\.venv\Scripts\activate

# Install dependencies
.\install_deps.bat

# Setup Ollama LLM server
python setup_ollama.py
```

### Main Application Commands
```powershell
# Start GridBot trading system
python GridbotBackup.py

# Start WebSocket server
python gridbot_websocket_server.py

# Run automation pipeline (test mode)
python automated_debugging_strategy\master_automation_pipeline.py --test-mode --single-run --max-iterations 1

# Run infinite automation loop
python llm_agent.py
```

### VS Code Tasks (Ctrl+Shift+P -> Tasks: Run Task)
- **GridBot: WebSocket Server** - Start WebSocket data server
- **GridBot: Core (Backup)** - Launch main trading bot
- **Agent: Health Check** - Verify system health
- **Agent: Quick Test** - Run basic functionality test
- **Agent: Full Simulation** - Complete system simulation
- **Agent: Dry Run** - Test mode without real trades
- **Agent: Performance Test** - System performance evaluation
- **Agent: LLM Test** - Test LLM connectivity
- **Start Ollama Server** - Launch local LLM server
- **Test SmolLM2 Connection** - Verify model availability

### Testing Commands
```powershell
# Run system health check
python agent_harness.py health

# Quick functionality test
python agent_harness.py quick

# Full simulation test
python agent_harness.py simulation

# Performance testing
python agent_harness.py performance

# LLM connectivity test
python agent_harness.py llm-test

# Test Ollama connection
python test_ollama.py

# Test SmolLM2 model
python test_smollm2.py
```

### Database Operations
```powershell
# Check database status
python check_db.py

# Alternative database check
python check_db2.py

# Validate database structure
python -c "import sqlite3; conn=sqlite3.connect('gridbot.db'); print('Tables:', [row[0] for row in conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()]); conn.close()"
```

### Log Analysis
```powershell
# View recent log entries
Get-Content "gridbot_ml.log" | Select-Object -Last 20

# Copy logs for analysis
Copy-Item "gridbot_ml.log" "backup_gridbot_ml_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Count log entries
(Get-Content "gridbot_ml.log").Count

# Monitor automation logs
Get-Content "automated_debugging_strategy\master_automation.log" -Tail 50 -Wait
```

### File Management
```powershell
# Clean temporary files
Remove-Item temp\* -Force -Recurse

# Archive old reports
Move-Item "debug_report_*.json" -Destination "reports\"

# Backup configuration
Copy-Item "config.py" "config.py.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
```

### System Utilities (Windows)
```powershell
# List processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Check port usage
netstat -ano | findstr :11434

# File operations
Get-ChildItem -Recurse -Include "*.py" | Measure-Object
Get-ChildItem -Path . -Name "*.log" | Sort-Object LastWriteTime

# System information
Get-ComputerInfo | Select-Object TotalPhysicalMemory, CsProcessors
```

### Development Workflow
```powershell
# 1. Start development session
.\.venv\Scripts\activate

# 2. Start Ollama server (background)
ollama serve

# 3. Test system health
python agent_harness.py health

# 4. Run automation pipeline
python automated_debugging_strategy\master_automation_pipeline.py --test-mode

# 5. Monitor logs
Get-Content "automated_debugging_strategy\master_automation.log" -Tail 20 -Wait
```

### Emergency Commands
```powershell
# Stop all Python processes
Get-Process python | Stop-Process -Force

# Kill specific process by PID
Stop-Process -Id 12345 -Force

# Restart Ollama server
Stop-Process -Name "ollama" -Force; Start-Sleep 2; ollama serve

# Reset virtual environment
deactivate; Remove-Item .venv -Recurse -Force; python -m venv .venv; .\.venv\Scripts\activate; .\install_deps.bat
```