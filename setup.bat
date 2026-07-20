@echo off
REM Quick setup for Windows — double-click or run from cmd:
REM   setup.bat

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: python not found. Install Python from https://python.org
    pause
    exit /b 1
)

echo Activating and installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To run the UI:
echo   venv\Scripts\activate.bat
echo   streamlit run app.py
echo.
echo To run the CLI test:
echo   venv\Scripts\activate.bat
echo   python run_cli.py
echo.
pause
