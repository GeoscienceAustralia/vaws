#! /usr/bin/env bash
#
echo Building GUI files...

rm -f *_ui.py
rm -f windsim_rc.py

echo Building UI
pyuic4 ui/main.ui > main_ui.py
pyuic4 ui/house.ui > house_ui.py
pyuic4 ui/connection_type.ui > connection_type_ui.py
pyuic4 ui/zone.ui > zone_ui.py
pyuic4 ui/connection.ui > connection_ui.py

echo Building resources
pyrcc4 windsim.qrc > windsim_rc.py

echo fini

