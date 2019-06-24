@echo off
echo Building GUI files...

del *_ui.py
del vaws_rc.py

echo Building UI

call pyuic5 --from-imports --output main_ui.py ui\main.ui
call pyuic5 ui\house.ui -o house_ui.py
call pyuic5 ui\connection_type.ui > connection_type_ui.py
call pyuic5 ui\zone.ui > zone_ui.py
call pyuic5 ui\connection.ui > connection_ui.py

echo Building resources
pyrcc5 vaws.qrc > vaws_rc.py

echo fini

