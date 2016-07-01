#!/bin/bash
#
echo Building GUI files...

rm -f *_ui.py
./clean.sh

echo Building UI
cd ui
pyuic4 main.ui > ../main_ui.py
pyuic4 house.ui > ../house_ui.py
pyuic4 connection_type.ui > ../connection_type_ui.py
pyuic4 zone.ui > ../zone_ui.py
pyuic4 connection.ui > ../connection_ui.py

echo Building resources
cd ..
pyrcc4 windsim.qrc > windsim_rc.py

echo fini

