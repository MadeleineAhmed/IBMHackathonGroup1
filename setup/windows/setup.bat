@echo off
REM Double-click this file to run the Windows team setup.
REM Pass -NoHub to skip the Skore Hub sign-in:  setup.bat -NoHub
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
