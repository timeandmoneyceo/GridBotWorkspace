# GridBot Project Structure and Architecture

## Root Directory Structure
```
GridBotWorkspace/
├── .venv/                          # Python virtual environment
├── .vscode/                        # VS Code configuration
│   └── tasks.json                  # Build/run/test task definitions
├── automated_debugging_strategy/   # AI-driven automation pipeline
│   ├── master_automation_pipeline.py    # Main orchestrator
│   ├── qwen_agent_interface.py          # LLM interface layer
│   ├── optimization_automation_system.py # Code optimization engine
│   ├── debug_automation_orchestrator.py # Error detection/fixing
│   ├── automated_file_editor.py         # Automated code editing
│   ├── serena_integration.py            # Semantic code editing
│   └── [reports/sessions/temp/]         # Generated data directories
├── backups/                        # Automated backups
├── reports/                        # Analysis reports
├── sessions/                       # Automation session data
└── temp/                          # Temporary files
```

## Core Application Files
```
GridBotWorkspace/
├── GridbotBackup.py               # Main trading bot logic
├── gridbot_websocket_server.py    # Real-time data server
├── config.py                      # Configuration management (200+ parameters)
├── agent_harness.py              # Testing and validation framework
├── llm_agent.py                   # Infinite loop automation controller
└── setup_ollama.py               # LLM server setup and configuration
```

## Configuration and Setup
```
GridBotWorkspace/
├── config.yaml                   # LLM model configuration
├── install_deps.bat             # Dependency installation script
├── mcp.json                      # Model Context Protocol configuration
├── README.md                     # Comprehensive project documentation
└── GridBotWorkspace.code-workspace # VS Code workspace settings
```

## Data and Logging
```
GridBotWorkspace/
├── database.db                   # Main SQLite database
├── gridbot.db                   # Trading data database
├── gridbot_ml.log              # ML trading log (CSV format)
├── feature_trades.csv          # Feature-based trading data
├── agent_heartbeat.json        # System health monitoring
└── [Various .log files]        # Component-specific logs
```

## Machine Learning Models
```
GridBotWorkspace/
├── client_pytorch_model.pth         # PyTorch LSTM model
├── client_pytorch_scaler.pkl        # PyTorch data scaler
├── client_pytorch_target_scaler.pkl # PyTorch target scaler
├── client_meta_model.pkl           # Meta-ensemble model
├── client_meta_scaler.pkl          # Meta-model scaler
├── client_sklearn_rf_scaler.pkl    # Random Forest scaler
├── client_sklearn_sgd_scaler.pkl   # SGD model scaler
└── client_xgb_scaler.pkl          # XGBoost scaler
```

## Automation Pipeline Architecture

### Main Components
1. **Master Automation Pipeline**: Central orchestrator managing the entire automation flow
2. **Queue Processor**: Manages operation sequences (Environment Validation → Debug Phase → Optimization Phase)
3. **LLM Interface Layer**: Coordinates between multiple specialized models:
   - Qwen3:1.7b (Orchestrator with agent capabilities)
   - DeepSeek-coder (Debugging specialist)
   - SmolLM2:1.7b (Optimization specialist)

### Data Flow
```
User Request → Master Pipeline → Operation Queue → Specialized Components → LLM Models → Results Processing → File Modification → Validation
```

### Key Subsystems

#### Debug Automation Orchestrator
- Syntax error detection and fixing
- Runtime error analysis
- Automated code correction
- Validation and testing

#### Optimization Automation System
- Performance bottleneck identification
- Code efficiency improvements
- Function refactoring suggestions
- Targeted optimization with context preservation

#### File Management System
- Automated backups with timestamps
- Log file rotation and archiving
- Temporary file cleanup
- Report generation and storage

### Serena Integration (MCP)
- Semantic code analysis and understanding
- Symbol-based code navigation
- Intelligent code editing capabilities
- Context-aware refactoring support

## Database Schema

### Core Tables
- **trades**: Trading transaction history
- **log_analysis**: Automation analysis results
- **iterations**: Pipeline execution tracking
- **parameters**: Configuration parameter history
- **predictions**: ML model predictions
- **clients**: WebSocket client connections

### Generated Reports
- **debug_report_*.json**: Debugging session results
- **optimization_report_*.json**: Code optimization results
- **automation_report_*.json**: Complete automation summaries
- **continuous_automation_report_*.json**: Long-running session data

## Codebase Patterns

### Error Handling
- Comprehensive try/except blocks with detailed logging
- Graceful degradation with fallback mechanisms
- ASCII-safe error messages for Windows terminal compatibility

### Threading Model
- Daemon threads for background operations
- Dynamic timeout management with extensions
- Thread-safe operation queues
- Proper cleanup and shutdown procedures

### Configuration Management
- Centralized parameter definitions with bounds checking
- Dynamic parameter optimization based on performance
- Environment-specific configuration overrides