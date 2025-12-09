@echo off
echo ==========================================
echo OpenBTK PyPI Publishing Script
echo ==========================================
echo.
echo requirements: pip install build twine
echo.

if not exist dist (
    echo [INFO] dist folder not found. Building package...
    python -m build
)

echo.
echo Uploading to PyPI...
echo You will be asked for your PyPI API Token (username: __token__).
python -m twine upload dist/*

echo.
echo Done!
pause
