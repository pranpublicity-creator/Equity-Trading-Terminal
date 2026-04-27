@echo off
title EQUITY TERMINAL — Port 5005
color 0A

echo.
echo  ============================================
echo   EQUITY TRADING TERMINAL (NSE)
echo   http://localhost:5005
echo  ============================================
echo.

cd /d "%~dp0"

echo  Starting... (browser will open automatically)
echo  Press Ctrl+C to stop the server.
echo.

REM Open browser after 3 seconds
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5005"

REM Start the Flask app
"C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe" app.py

echo.
echo  Server stopped. Press any key to close.
pause >nul
