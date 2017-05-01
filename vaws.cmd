@echo off



setlocal

set VAWS_DIR=%~dp0

set PYTHONPATH=%VAWS_DIR%src;%PYTHONPATH%


echo %PYTHONPATH%



echo "Starting vaws gui"
 
echo %*

python src\gui\main.py

 %*

endlocal
