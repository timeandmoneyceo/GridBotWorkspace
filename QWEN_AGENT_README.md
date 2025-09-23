# Qwen Agent Enhanced Automation Pipeline

This automated debugging and optimization system has been enhanced with Qwen3:1.7B agent capabilities for advanced code analysis, debugging, and optimization.

## Features

### 🤖 Qwen Agent Integration
- **Tool Calling**: Qwen agent can use various tools for file operations, code analysis, and system monitoring
- **Advanced Reasoning**: Thinking mode enabled for complex problem-solving
- **Optimized Parameters**: Uses Qwen3 recommended sampling parameters for best performance

### 📊 Summary and Reference System
- **Operation Summaries**: Automatic creation of summary files tracking all operations
- **Reference Files**: Maintains knowledge base of code patterns, error solutions, and optimizations
- **Efficiency Tracking**: Monitors and reports on system performance and improvements

### 🔧 Enhanced Capabilities
- **File System Access**: Read, write, and analyze files across the workspace
- **Code Analysis**: AST parsing, complexity analysis, performance profiling
- **Execution Tools**: Run code snippets and tests with output capture
- **System Monitoring**: Track resource usage and system health

## Configuration

The system uses Qwen3:1.7B with the following optimized parameters:

```json
{
  "llm_model": "Qwen3-1.7B",
  "llm_base_url": "http://localhost:8000/v1",
  "llm_temperature": 0.6,
  "llm_top_p": 0.95,
  "llm_top_k": 20,
  "llm_min_p": 0.0,
  "llm_max_tokens": 32768,
  "enable_thinking": true
}
```

## Usage

### Basic Run
```bash
python master_automation_pipeline.py
```

### Single Cycle (No Continuous Mode)
```bash
python master_automation_pipeline.py --single-run
```

### Custom Configuration
```bash
python master_automation_pipeline.py --config custom_config.json
```

### Specify Target Files
```bash
python master_automation_pipeline.py --files GridbotBackup.py config.py
```

## Qwen Agent Tools

The Qwen agent has access to comprehensive tools:

### File Operations
- `read_file`: Read file contents with line number support
- `list_directory`: List directory contents recursively
- `search_files`: Find files matching patterns
- `grep_search`: Search for text patterns in files

### Code Analysis
- `analyze_code`: Analyze code for issues and optimization opportunities
- `extract_functions`: Extract and analyze function definitions
- `run_code`: Execute Python code and capture output
- `run_tests`: Run unit tests with detailed reporting

### System Monitoring
- `get_system_info`: Get system resource usage and process information

### Summary and Reference
- `create_summary`: Generate operation summaries
- `update_reference`: Update reference knowledge base

## Output Files

The system creates several types of output files:

### Summary Files
- `qwen_agent_summaries.json`: Operation summaries and results
- `automation_session_*.json`: Detailed session data
- `automation_report_*.json`: Final pipeline reports

### Reference Files
- `qwen_agent_references_code_patterns.json`: Code patterns and best practices
- `qwen_agent_references_error_solutions.json`: Error solutions and fixes
- `qwen_agent_references_optimizations.json`: Optimization techniques
- `qwen_agent_references_debugging.json`: Debugging strategies

### Logs
- `master_automation.log`: Main pipeline logging
- `qwen_interface.log`: Qwen agent specific logging
- `debug_orchestrator.log`: Debugging operations

## Requirements

### Qwen Setup
1. Install Qwen agent: `pip install qwen-agent`
2. Set up Qwen3:1.7B server (recommended endpoint: `http://localhost:8000/v1`)
3. Ensure the model is running and accessible

### Python Dependencies
- qwen-agent
- requests
- psutil (for system monitoring)
- Other dependencies as listed in requirements.txt

## Best Practices

### For Thinking Mode (enable_thinking=True)
- Temperature: 0.6
- Top P: 0.95
- Top K: 20
- Min P: 0.0

### For Non-Thinking Mode (enable_thinking=False)
- Temperature: 0.7
- Top P: 0.8
- Top K: 20
- Min P: 0.0

### Output Length
- Standard queries: 32,768 tokens
- Complex problems: 38,912 tokens

## Troubleshooting

### Qwen Connection Issues
- Verify Qwen server is running on the correct endpoint
- Check API key configuration (use "EMPTY" for local models)
- Ensure model name matches server configuration

### Performance Issues
- Adjust max_tokens based on query complexity
- Monitor system resources during operation
- Use summary_save_interval to control file I/O frequency

### File Access Issues
- Ensure workspace_path is correctly set
- Check file permissions for read/write operations
- Verify target files exist and are accessible

## Continuous Automation

The system supports continuous autonomous operation:

- **Infinite Loops**: Set max_cycles to 0 for continuous operation
- **Health Checks**: Automatic system health monitoring
- **Auto Cleanup**: Periodic file cleanup between cycles
- **Failure Tolerance**: Configurable consecutive failure limits

## Integration Notes

This Qwen-enhanced system works on top of the existing automation pipeline without disrupting the main project files. All updates are contained within the `automated_debugging_strategy` folder as requested.

The agent provides advanced reasoning capabilities while maintaining compatibility with existing debugging and optimization workflows.