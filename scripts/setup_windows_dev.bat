@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "POWERSHELL_EXE=powershell"

where %POWERSHELL_EXE% >nul 2>nul
if errorlevel 1 (
    echo PowerShell was not found. Open PowerShell and run:
    echo   powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup_windows_dev.ps1"
    exit /b 1
)

%POWERSHELL_EXE% -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup_windows_dev.ps1" %*
exit /b %ERRORLEVEL%
