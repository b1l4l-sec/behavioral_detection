@echo off

echo ========================================
echo    Behavioral Detection System
echo    Quick Launcher
echo ========================================

cd /d "%~dp0"

if "%1"=="" goto menu
if "%1"=="dashboard" goto dashboard
if "%1"=="detector" goto detector
if "%1"=="all" goto all
if "%1"=="generator" goto generator
if "%1"=="trainer" goto trainer
if "%1"=="tests" goto tests
goto menu

:menu
echo.
echo Available commands:
echo   run.bat dashboard  - Start Streamlit dashboard
echo   run.bat detector   - Start real-time detector
echo   run.bat all        - Start all components
echo   run.bat generator  - Generate training data
echo   run.bat trainer    - Train ML models
echo   run.bat tests      - Run all tests
echo.
goto end

:dashboard
echo Starting Streamlit Dashboard...
.venv\Scripts\streamlit.exe run src\interface\streamlit_app.py --server.port=8501
goto end

:detector
echo Starting Real-time Detector...
.venv\Scripts\python.exe -m src.detector.realtime_detector --duration 0
goto end

:all
echo Starting All Components...
start "Dashboard" .venv\Scripts\streamlit.exe run src\interface\streamlit_app.py --server.port=8501
timeout /t 3 /nobreak >nul
start "Detector" .venv\Scripts\python.exe -m src.detector.realtime_detector --duration 0
echo.
echo All components started!
echo   - Dashboard: http://localhost:8501
echo   - Detector: Running in background
goto end

:generator
echo Starting Dataset Generator...
.venv\Scripts\python.exe -m src.generator.dataset_generator --benign 1000 --malicious 800 --duration 10
goto end

:trainer
echo Starting Model Trainer...
.venv\Scripts\python.exe -m src.models.train_models
goto end

:tests
echo Running Tests...
.venv\Scripts\python.exe -m pytest tests/ -v
goto end

:end
