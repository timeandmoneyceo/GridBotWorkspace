# GridBot Project Overview

## Project Purpose
GridBot is an advanced automated cryptocurrency trading system with AI-driven optimization capabilities. The project implements:

- **Automated Grid Trading**: Places buy/sell orders in a grid pattern to capture profit from market volatility
- **ML-Powered Predictions**: Uses multiple machine learning models (PyTorch LSTM, Scikit-learn, XGBoost) for price prediction
- **Real-time WebSocket Data**: Processes live market data from exchanges like Coinbase and Alpaca
- **Self-Healing Automation**: LLM-driven debugging and optimization pipeline that continuously improves the codebase
- **Configuration Management**: Dynamic parameter optimization based on market conditions

## Core Components

### Trading Engine (`GridbotBackup.py`)
- Main trading logic with grid-based order placement
- ML model training and prediction (PyTorch, Scikit-learn, XGBoost, Meta-model ensemble)
- Technical analysis indicators (RSI, MACD, Bollinger Bands, ATR, VWAP)
- Risk management and position sizing
- Breakout detection and trailing stop mechanisms

### WebSocket Server (`gridbot_websocket_server.py`)
- Real-time market data processing
- Client connection management
- Database operations for trade logging
- Message translation for ML processing

### Configuration System (`config.py`)
- 200+ configurable parameters for trading strategy
- Min/max bounds for parameter optimization
- ML model hyperparameters
- Risk management settings

### Automation Pipeline (`automated_debugging_strategy/`)
- **Master Automation Pipeline**: Orchestrates debugging and optimization
- **LLM Integration**: Uses Ollama server with Qwen3, DeepSeek-coder, SmolLM2 models
- **Automated File Editor**: Applies code fixes automatically
- **Debug Orchestrator**: Identifies and resolves errors
- **Optimization System**: Improves code performance
- **Serena Integration**: Semantic code editing capabilities

## Architecture
- **Language**: Python 3.13
- **Environment**: Virtual environment (.venv) with comprehensive dependencies
- **Database**: SQLite for trade history and system state
- **ML Framework**: PyTorch, Scikit-learn, XGBoost with ensemble methods
- **Communication**: WebSocket for real-time data
- **Exchange Integration**: CCXT library for multiple exchange support
- **Logging**: Comprehensive logging with ASCII-safe Windows terminal compatibility

## Key Features
- **Continuous Operation**: Infinite loop automation with graceful restart capabilities
- **Multi-Model Ensemble**: Combines multiple ML approaches for robust predictions
- **Dynamic Parameter Tuning**: AI-driven configuration optimization
- **Error Recovery**: Automatic detection and fixing of runtime issues
- **Performance Monitoring**: Real-time profiling and optimization
- **Unicode Safe**: All logging designed for Windows terminal compatibility