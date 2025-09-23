# GridBot Task Completion Guidelines

## Post-Development Checklist

### Code Quality Verification
1. **Syntax Validation**
   ```powershell
   python -m py_compile filename.py
   ```

2. **Import Validation**
   ```powershell
   python -c "import filename; print('Imports successful')"
   ```

3. **Unicode Safety Check**
   - Ensure all logging uses ASCII-safe characters
   - No Unicode emojis in log messages (use `[SUCCESS]`, `[ERROR]` tags)
   - UTF-8 encoding with fallback handling for file operations

### Testing Requirements
1. **Health Check** (Required after any changes)
   ```powershell
   python agent_harness.py health
   ```

2. **Quick Test** (For minor changes)
   ```powershell
   python agent_harness.py quick
   ```

3. **Full Simulation** (For major changes)
   ```powershell
   python agent_harness.py simulation
   ```

4. **LLM Connectivity** (If automation pipeline modified)
   ```powershell
   python agent_harness.py llm-test
   ```

### Automation Pipeline Validation
1. **Test Mode Execution**
   ```powershell
   python automated_debugging_strategy\master_automation_pipeline.py --test-mode --single-run --max-iterations 1
   ```

2. **Check Ollama Server Status**
   ```powershell
   curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"qwen3:1.7b\",\"prompt\":\"Hello\",\"stream\":false}"
   ```

3. **Verify Model Availability**
   ```powershell
   ollama list
   ```

### Database Integrity
1. **Database Structure Check**
   ```powershell
   python check_db.py
   ```

2. **Verify Table Schema**
   ```powershell
   python -c "import sqlite3; conn=sqlite3.connect('gridbot.db'); print([row for row in conn.execute('PRAGMA table_info(trades)').fetchall()]); conn.close()"
   ```

### Performance Validation
1. **Memory Usage Check**
   - Monitor for memory leaks in long-running processes
   - Verify proper cleanup of temporary files

2. **Log File Management**
   - Check log file sizes don't exceed reasonable limits
   - Verify log rotation is working properly

3. **Thread Safety**
   - Ensure proper daemon thread management
   - Verify graceful shutdown capabilities

### File Management
1. **Backup Critical Files**
   ```powershell
   Copy-Item "config.py" "config.py.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
   ```

2. **Clean Temporary Files**
   ```powershell
   Remove-Item temp\* -Force -Recurse -ErrorAction SilentlyContinue
   ```

3. **Archive Old Reports**
   ```powershell
   $cutoffDate = (Get-Date).AddDays(-7)
   Get-ChildItem "debug_report_*.json" | Where-Object {$_.LastWriteTime -lt $cutoffDate} | Move-Item -Destination "reports\"
   ```

### Documentation Updates
1. **Update README.md** if new features added
2. **Update task configuration** in `.vscode/tasks.json` if new commands added
3. **Document configuration changes** in commit messages

### Version Control Best Practices
1. **Commit Message Format**
   ```
   [COMPONENT] Brief description
   
   - Detailed change 1
   - Detailed change 2
   - Performance impact: X% improvement
   ```

2. **Ensure No Sensitive Data** in commits
   - API keys should be in environment variables
   - Database files should be in .gitignore

### Final Validation Commands
```powershell
# Complete validation sequence
.\.venv\Scripts\activate
python agent_harness.py health
python test_ollama.py
python automated_debugging_strategy\master_automation_pipeline.py --test-mode --single-run
echo "Task completion validation successful"
```

### Emergency Rollback Plan
1. **Backup Strategy**: Always maintain working configuration backups
2. **Rollback Command**: 
   ```powershell
   Copy-Item "config.py.backup.TIMESTAMP" "config.py" -Force
   ```
3. **Service Restart**:
   ```powershell
   Get-Process python | Stop-Process -Force
   Start-Sleep 2
   python GridbotBackup.py
   ```