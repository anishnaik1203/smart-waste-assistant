@echo off
TITLE Smart Waste AI Assistant - Streamlit Launcher
echo ==============================================================
echo       Smart Waste Classification & Recycling Assistant
echo          Artificial Intelligence & Data Science Engineering
echo ==============================================================
echo.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please install dependencies first using:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Activating virtual environment...
CALL .venv\Scripts\activate.bat

echo Starting Streamlit Web Application on http://localhost:8501 ...
echo.
streamlit run app.py

pause
