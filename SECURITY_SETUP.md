# 🔐 Security Setup Instructions

## ⚠️ IMPORTANT: Configuration Setup

Before running this application, you must set up your configuration file with your actual API credentials.

### Setup Steps:

1. **Copy the example configuration:**
   ```bash
   cp automated_debugging_strategy/config.example.py automated_debugging_strategy/config.py
   ```

2. **Edit the configuration file:**
   - Open `automated_debugging_strategy/config.py`
   - Replace `"your-api-key-here"` with your actual API key
   - Replace `"your-private-key-here"` with your actual private key

3. **Verify the setup:**
   - Make sure `config.py` is listed in `.gitignore`
   - Never commit `config.py` to version control
   - Only commit `config.example.py` with placeholder values

### 🛡️ Security Notes:

- The actual `config.py` file is automatically ignored by Git
- Your sensitive credentials will never be committed to the repository
- Always use the example file as a template for new setups
- Keep your API keys and private keys secure and never share them

### 📁 File Structure:
```
automated_debugging_strategy/
├── config.example.py  ✅ (Safe to commit - template only)
├── config.py          ❌ (Ignored by Git - contains real credentials)
└── ...
```

For more information about the trading bot configuration, see the main README.md file.