# GridBot Tech Stack and Dependencies

## Core Technology Stack

### Programming Language
- **Python 3.13**: Latest Python version with enhanced performance
- **Virtual Environment**: `.venv` with isolated dependencies

### Machine Learning & AI
- **PyTorch**: Deep learning framework for LSTM price prediction models
- **Scikit-learn**: Traditional ML algorithms (Random Forest, SGD)
- **XGBoost**: Gradient boosting for ensemble predictions
- **Meta-Model Ensemble**: Combines multiple model predictions

### LLM Integration
- **Ollama Server**: Local LLM server (localhost:11434)
- **Model Architecture**:
  - Qwen3:1.7b (Orchestrator with agent capabilities)
  - DeepSeek-coder (Debugging specialist)
  - SmolLM2:1.7b (Optimization specialist)
- **Model Context Protocol (MCP)**: Serena integration for semantic code editing

### Data Processing & Storage
- **SQLite**: Primary database for trades, logs, parameters, predictions
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Trading & Exchange Integration
- **CCXT**: Unified API for cryptocurrency exchanges
- **WebSocket Clients**: Real-time market data (websocket-client, websocket-server)
- **Exchange Support**: Coinbase Pro, Alpaca, others via CCXT

### Development & Automation Tools
- **VS Code**: Primary IDE with extensive task configuration
- **Automated Testing**: Unit tests, integration tests, performance tests
- **Logging**: concurrent-log-handler with ASCII-safe Windows compatibility
- **Process Management**: Subprocess handling with PID tracking

### Performance & Monitoring
- **Threading**: Multi-threaded operations with timeout management
- **Profiling**: Performance monitoring and optimization
- **Backoff**: Exponential backoff for API retries
- **Memory Management**: Efficient handling of large datasets

## Dependencies Installation
```bash
# Core packages installed via install_deps.bat:
ccxt websocket-client websocket-server urllib3 
scikit-learn backoff xgboost joblib concurrent-log-handler

# PyTorch with CUDA support:
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional packages in virtual environment:
requests pandas numpy sqlite3 threading subprocess json yaml
```

## System Architecture
- **Windows Environment**: Optimized for Windows PowerShell and terminal
- **Containerization**: Virtual environment isolation
- **Service Architecture**: Background services with health checks
- **API Integration**: RESTful and WebSocket communication
- **Unicode Compatibility**: ASCII-safe logging for Windows cp1252 encoding