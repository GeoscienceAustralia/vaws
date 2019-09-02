@echo off
echo Building GUI files...
set PDIR=%~dp0
REM echo %PDIR%

del %PDIR%\*_ui.py
del %PDIR%\vaws_rc.py

echo Building UI
call pyuic5 --from-imports --output %PDIR%\main_ui.py %PDIR%\ui\main.ui
powershell -Command "(gc %PDIR%\main_ui.py) -replace 'from matplotlibwidget','from vaws.gui.matplotlibwidget' | Out-File -encoding ASCII %PDIR%\main_ui.py"
call pyuic5 %PDIR%\ui\house.ui -o %PDIR%\house_ui.py

echo Building resources
call pyrcc5 -o %PDIR%\vaws_rc.py %PDIR%\vaws.qrc 