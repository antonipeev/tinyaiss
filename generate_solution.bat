@echo off
REM Generate Visual Studio 2026 solution for tinyAISS
cd /d "%~dp0"
cd build
cmake .. -G "Visual Studio 18 2026" -A x64
cd ..
echo.
echo ============================================
echo Solution generated in ./build/tinyAISS.slnx
echo ============================================
pause
