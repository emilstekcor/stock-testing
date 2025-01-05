@echo off
REM run_app.bat: Simple Batch File to Launch the Streamlit Trading App

echo =========================================
echo Launching your Streamlit Trading App...
echo =========================================

REM If you want to ensure a specific environment or path, you can set it here.
REM For example:
REM call "C:\path\to\your\venv\Scripts\activate.bat"

streamlit run main.py

REM Prevent the window from closing immediately (optional):
echo.
pause
