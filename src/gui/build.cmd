@echo off
echo Building GUI files...

del *_ui.py
del windsim_rc.py

echo Building UI

call pyuic4 ui\main.ui -o main_ui.py

call pyuic4 ui\house.ui -o house_ui.py

call pyuic4 ui\connection_type.ui > connection_type_ui.py

call pyuic4 ui\zone.ui > zone_ui.py

call pyuic4 ui\connection.ui > connection_ui.py

echo Building resources

pyrcc4 windsim.qrc > windsim_rc.py

echo fini

