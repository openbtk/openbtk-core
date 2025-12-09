@echo off
echo ==========================================
echo OpenBTK GitHub Deployment Script
echo ==========================================
echo.
echo Checking for Git...
git --version
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in your PATH.
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b
)

echo.
echo Initializing Git repository...
git init
git remote add origin https://github.com/openbtk/openbtk-core.git

echo.
echo Adding files...
git add .

echo.
echo Committing files...
git commit -m "Initial release of OpenBTK v0.1.0"

echo.
echo Pushing to GitHub (You may need to sign in)...
git branch -M main
git push -u origin main

echo.
echo Done!
pause
