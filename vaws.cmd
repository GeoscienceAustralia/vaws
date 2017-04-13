@echo off

set VAWS_DIR=c:\dev\vaws\vaws

set PYTHONPATH=%VAWS_DIR%\src;%PYTHONPATH%
echo %PYTHONPATH%

echo "Starting vaws gui"
python src\gui\main.py

