#! /usr/bin/env bash
#
echo Building GUI files...

rm -f *_ui.py
rm -f vaws_rc.py

echo Building UI
pyuic5 --from-imports --output main_ui.py ui/main.ui
pyuic5 ui/house.ui > house_ui.py
pyuic5 ui/connection_type.ui > connection_type_ui.py
pyuic5 ui/zone.ui > zone_ui.py
pyuic5 ui/connection.ui > connection_ui.py

echo Building resources
pyrcc5 -o vaws_rc.py vaws.qrc

echo fini

