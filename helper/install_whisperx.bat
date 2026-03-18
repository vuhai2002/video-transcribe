
@echo off
echo ===================================================
echo INSTALL WHISPERX & DEPENDENCIES (Windows)
echo ===================================================

echo 1. Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 2. Installing WhisperX from GitHub...
pip install git+https://github.com/m-bain/whisperx.git

echo.
echo 3. Installing RapidFuzz for text matching...
pip install rapidfuzz

echo.
echo 4. Verifying installation...
python -c "import whisperx; print('WhisperX installed successfully!')"
if %errorlevel% neq 0 (
    echo [ERROR] WhisperX installation failed. Check if git is installed and reachable.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Build environment ready!
pause
