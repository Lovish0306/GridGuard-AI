@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo  GridGuard AI - setup and launch
echo ============================================
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py -3
) else (
  set PY=python
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto error
if not exist "models\fault_classifier.joblib" (
  echo Training synthetic-data models...
  python train_models.py
  if errorlevel 1 goto error
)
echo Starting GridGuard AI...
python -m streamlit run app.py
goto end
:error
echo.
echo Setup failed. Ensure Python 3.10-3.12 is installed and internet is available.
pause
:end
endlocal
