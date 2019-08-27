mkdir %SP_DIR%\vaws
mkdir %SP_DIR%\vaws\gui
mkdir %SP_DIR%\vaws\model
mkdir %SP_DIR%\vaws\scenarios

xcopy vaws\gui %SP_DIR%\vaws\gui /S /Y
xcopy vaws\model %SP_DIR%\vaws\model /S /Y
xcopy vaws\scenarios %SP_DIR%\vaws\scenarios /S /Y
for /d /r %SP_DIR\vaws %d in (output) do @if exist "%d" rd /s/q "%d"
xcopy vaws\__init__.py %SP_DIR%\vaws /Y
