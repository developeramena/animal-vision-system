@echo off
setlocal
title Animal Vision System
cd /d "%~dp0"

if not exist "app.py" (
    echo ERROR: app.py was not found.
    echo Keep this file inside the Animal_Vision_Streamlit_Project folder.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python 3.13 environment...
    py -3.13 -m venv .venv
    if errorlevel 1 goto python_error
)

".venv\Scripts\python.exe" -c "import streamlit, tensorflow, cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing the required libraries. This may take several minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto install_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto install_error
)

echo.
echo Starting Animal Vision System...
echo Website: http://localhost:8501
echo Keep this window open while using the website.
echo Press Ctrl+C here when you want to stop the website.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.port 8501
if errorlevel 1 goto app_error
exit /b 0

:python_error
echo.
echo ERROR: Python 3.13 could not create the environment.
echo Confirm it with: py -3.13 --version
pause
exit /b 1

:install_error
echo.
echo ERROR: One or more libraries could not be installed.
echo Check the internet connection and try again.
pause
exit /b 1

:app_error
echo.
echo ERROR: The Streamlit application stopped with an error.
pause
exit /b 1
