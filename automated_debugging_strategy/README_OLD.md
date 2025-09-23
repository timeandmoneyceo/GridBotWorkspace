# GridBot Automated Debugging Strategy

A clean, focused system for automatically debugging and optimizing your GridBot code using a local LLM.

## Quick Start

### Prerequisites
- Local LLM server running at `http://localhost:11434` (Ollama with smollm2:1.7b)
- Python 3.7+
- Target files: `GridbotBackup.py` and `gridbot_websocket_server.py`

### Run Automation
```bash
# Simple run
python master_automation_pipeline.py

# Custom options
python master_automation_pipeline.py --files GridbotBackup.py --max-iterations 5 --verbose
```

## What It Does

1. **🔍 Detects Errors**: Runs your code and parses error output
2. **🤖 Gets Fixes**: Asks local LLM to generate targeted fixes
3. **✏️ Applies Changes**: Safely edits files with automatic backups
4. **🔄 Iterates**: Repeats until errors are fixed or max iterations reached
5. **⚡ Optimizes**: Analyzes and improves code performance

## Core Components

- **`master_automation_pipeline.py`** - Main orchestrator
- **`debug_automation_orchestrator.py`** - Debug cycle management
- **`llm_interface.py`** - Local LLM communication
- **`automated_file_editor.py`** - Safe file editing with backups
- **`debug_log_parser.py`** - Error extraction from logs

## Configuration

Edit `automation_config.json`:
```json
{
  "llm_base_url": "http://localhost:11434",
  "llm_model": "smollm2:1.7b",
  "max_debug_iterations": 10,
  "target_files": ["GridbotBackup.py", "gridbot_websocket_server.py"],
  "run_optimization": true,
  "verbose": true
}
```

## Key Features

- ✅ **Automatic Backups**: Every change is backed up
- ✅ **Syntax Validation**: Checks code before applying changes
- ✅ **Smart Error Detection**: Handles various Python error types
- ✅ **Performance Optimization**: Identifies and improves slow code
- ✅ **Comprehensive Logging**: Detailed operation logs
- ✅ **Safe Rollbacks**: Can restore from backups if needed

## Output

- **Session Reports**: `automation_session_*.json`
- **Debug Logs**: `debug_orchestrator.log`
- **Backups**: `backups/` folder with timestamped file versions
- **Status**: Real-time console output with progress

## Recent Improvements (Sept 2025)

- Fixed file editor validation issues
- Enhanced LLM prompt construction to prevent oversized responses
- Improved error detection with fallback mechanisms
- Better integration between debug parser and orchestrator
- Stricter response size limits to prevent function dumps
- Enhanced code extraction and validation

## Troubleshooting

1. **LLM Connection Failed**: Ensure Ollama is running with `ollama serve`
2. **File Errors**: Check target files exist and have proper permissions
3. **Syntax Issues**: System validates automatically; check logs for details
4. **No Fixes Applied**: Review LLM responses in logs for debugging

## Command Line Options

```bash
--files FILE1 FILE2        # Specify target files
--max-iterations N          # Maximum debug cycles per file
--no-optimization          # Skip optimization phase
--verbose                  # Detailed logging
--llm-url URL              # Custom LLM server URL
--llm-model MODEL          # Custom model name
```

This system automatically maintains and improves your GridBot code, reducing manual debugging time and enhancing performance.