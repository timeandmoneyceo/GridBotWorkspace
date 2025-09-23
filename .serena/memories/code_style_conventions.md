# GridBot Code Style and Conventions

## Code Style Guidelines

### Naming Conventions
- **Variables**: `snake_case` throughout the codebase
  - `bot_state`, `trade_count`, `ml_confidence_threshold`
- **Functions**: `snake_case` with descriptive names
  - `place_sell_after_buy()`, `calculate_locked_funds()`, `train_pytorch_predictor()`
- **Classes**: `PascalCase` 
  - `WebSocketManager`, `LSTMPricePredictor`, `OptimizationCandidate`
- **Constants**: `UPPER_SNAKE_CASE`
  - `GRID_SIZE`, `NUM_BUY_GRID_LINES`, `ML_CONFIDENCE_THRESHOLD`
- **Files**: `snake_case.py`
  - `gridbot_websocket_server.py`, `master_automation_pipeline.py`

### Type Hints
- **Current State**: Minimal type hints observed in codebase
- **Functions**: Some functions use return type hints (`-> None`, `-> bool`)
- **Parameters**: Limited parameter type annotations
- **Recommendation**: Type hints not heavily enforced but encouraged for new code

### Documentation Style
- **Docstrings**: Triple quoted strings with descriptive text
```python
def migrate_database() -> None:
    \"\"\"Migrate existing database to ensure all required tables and
    columns exist with proper structure and constraints.\"\"\"
```
- **Comments**: Single line `#` comments for inline explanations
- **Section Headers**: Numbered comments for logical sections
```python
# 1. Fetch Market Data
# 2. Update Balances  
# 3. Refresh Data for ML Predictions
```

### Function Structure
- **Single Responsibility**: Functions generally follow single purpose principle
- **Error Handling**: Extensive try/except blocks with logging
- **Return Patterns**: Consistent early returns for error conditions
- **Parameter Validation**: Input validation with appropriate error messages

### Logging Standards
- **ASCII-Safe**: All logging uses ASCII characters (no Unicode emojis)
- **Structured Tags**: Consistent prefixes like `[SUCCESS]`, `[ERROR]`, `[WORKING]`
- **Timestamp Format**: `[HH:MM:SS.mmm]` for precise timing
- **Log Levels**: INFO, WARNING, ERROR with appropriate usage
- **Context**: Rich context information in log messages

### File Organization
- **Module Structure**: Clear separation of concerns
- **Import Organization**: Standard library first, then third-party, then local
- **Global Variables**: Defined at module level with clear names
- **Configuration**: Centralized in `config.py` with extensive parameters

### Performance Patterns
- **Threading**: Daemon threads for background operations
- **Timeouts**: Dynamic timeout management with extensions
- **Resource Management**: Proper file handle and connection cleanup
- **Memory Efficiency**: Avoiding large string concatenations, using generators

### Windows Compatibility
- **Encoding**: UTF-8 with fallback handling for file operations
- **Path Handling**: OS-agnostic path operations
- **Terminal Output**: ASCII-safe characters only (cp1252 compatible)
- **Process Management**: Windows-specific PID and signal handling