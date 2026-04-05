@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
title Telco Customer Churn Prediction - Environment Setup
color 0B

echo ============================================================
echo    Telco Customer Churn Prediction - One-Click Setup
echo    BMDS2003 Data Science Project
echo ============================================================
echo.

echo [1/6] Checking conda installation...
where conda >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Conda is not found in your PATH!
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)
echo       [OK] Conda found.
echo.

echo [2/6] Checking conda environment 'telco_churn'...
call conda info --envs | findstr /C:"telco_churn" >nul 2>&1
if errorlevel 1 (
    echo       Creating conda environment 'telco_churn' with Python 3.10...
    call conda create -n telco_churn python=3.10 -y
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        pause
        exit /b 1
    )
    echo       [OK] Environment created successfully.
) else (
    echo       [OK] Environment 'telco_churn' already exists. Skipping creation.
)
echo.

echo [3/6] Activating environment 'telco_churn'...
set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" (
    for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
        set "CONDA_BAT=%%I"
        goto :_conda_bat_found
    )
)
:_conda_bat_found
if not exist "%CONDA_BAT%" (
    echo [ERROR] Could not locate conda.bat for environment activation.
    pause
    exit /b 1
)
call "%CONDA_BAT%" activate telco_churn >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment.
    echo Try running: conda init cmd.exe
    pause
    exit /b 1
)
echo       [OK] Environment activated.
echo.

echo [4/6] Detecting GPU and installing PyTorch...
set "TORCH_OK=0"
nvidia-smi >nul 2>&1
if errorlevel 1 goto :_torch_cpu_install
if not errorlevel 1 (
    echo       [GPU DETECTED] NVIDIA GPU found! Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    if not errorlevel 1 (
        set "TORCH_OK=1"
        goto :_torch_done
    )
)
echo       [WARNING] GPU install failed. Falling back to CPU version...
:_torch_cpu_install
echo       [CPU MODE] Installing PyTorch CPU version...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if not errorlevel 1 set "TORCH_OK=1"
:_torch_done
if "%TORCH_OK%"=="1" goto :_torch_ok
echo [ERROR] Failed to install PyTorch (both GPU and CPU options).
pause
exit /b 1
:_torch_ok
echo       [OK] PyTorch installed.
echo.

echo [5/6] Installing Python dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo       [OK] All dependencies installed.
echo.

echo [6/6] Checking trained models...
if exist "models\model_rf.pkl" (
    echo       [OK] Trained models found. Skipping training.
) else (
    echo       No trained models found. Starting training pipeline...
    echo       This may take 1-3 minutes...
    echo.
    python train.py
    if errorlevel 1 (
        echo [ERROR] Training failed. Please check the error messages above.
        pause
        exit /b 1
    )
    echo.
    echo       [OK] All models trained and saved successfully.
)
echo.

echo ============================================================
echo    Setup Complete! Launching Streamlit App...
echo    Press Ctrl+C in terminal to stop the server.
echo ============================================================
echo.
python -m streamlit run app.py --browser.gatherUsageStats false

pause
