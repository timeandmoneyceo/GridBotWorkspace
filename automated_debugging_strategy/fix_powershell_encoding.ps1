# Fix PowerShell encoding issues for Unicode characters
# Run this script to set UTF-8 encoding for the current PowerShell session

Write-Host "Setting PowerShell encoding to UTF-8..." -ForegroundColor Green

# Set console output encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Set console input encoding to UTF-8  
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Set PowerShell's default encoding to UTF-8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

Write-Host "PowerShell encoding set to UTF-8" -ForegroundColor Green
Write-Host "This will allow proper display of Unicode characters and emojis" -ForegroundColor Yellow

# Optional: Add to PowerShell profile for persistent setting
$profileExists = Test-Path $PROFILE
if ($profileExists) {
    Write-Host "To make this permanent, add the following lines to your PowerShell profile:" -ForegroundColor Cyan
    Write-Host '$PSDefaultParameterValues["*:Encoding"] = "utf8"' -ForegroundColor White
    Write-Host '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8' -ForegroundColor White
    Write-Host '[Console]::InputEncoding = [System.Text.Encoding]::UTF8' -ForegroundColor White
} else {
    Write-Host "PowerShell profile not found. Consider creating one for persistent settings." -ForegroundColor Yellow
}

Write-Host "`nEncoding fix applied! You can now run the automation pipeline." -ForegroundColor Green