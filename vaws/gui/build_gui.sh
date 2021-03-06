#! /usr/bin/env bash
#
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Building GUI files... $DIR"
rm -f $DIR/*_ui.py
rm -f $DIR/vaws_rc.py

echo Building UI
pyuic5 --import-from="vaws.gui" --output $DIR/main_ui.py $DIR/ui/main.ui
sed -i 's/matplotlibwidget/vaws.gui.matplotlibwidget/g' $DIR/main_ui.py
pyuic5 $DIR/ui/house.ui > $DIR/house_ui.py

echo Building resources
pyrcc5 -o $DIR/vaws_rc.py $DIR/vaws.qrc

echo Done
