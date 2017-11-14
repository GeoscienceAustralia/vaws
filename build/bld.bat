mkdir %SP_DIR%\vaws
mkdir %SP_DIR%\vaws\gui
mkdir %SP_DIR%\vaws\model

xcopy vaws\gui %SP_DIR%\vaws\gui /S /Y
xcopy vaws\model %SP_DIR%\vaws\model /S /Y
xcopy vaws\*.py %SP_DIR%\vaws /Y
