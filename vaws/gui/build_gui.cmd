@echo off
echo Building GUI files...
set PDIR=%~dp0
echo %PDIR%

del %PDIR%\*_ui.py
del %PDIR%\vaws_rc.py

echo Building UI
where pyuic5
where pyrcc5
pyuic5 --from-imports --output %PDIR%\main_ui.py %PDIR%\ui\main.ui
REM powershell -Command "(gc %PDIR%\main_ui.py) -replace 'from matplotlibwidget','from vaws.gui.matplotlibwidget' | Out-File -encoding ASCII %PDIR%\main_ui.py
pyuic5 %PDIR%\ui\house.ui -o %PDIR%\house_ui.py

echo Building resources
pyrcc5 %PDIR%\vaws.qrc > %PDIR%\vaws_rc.py

echo done
