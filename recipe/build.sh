mkdir $SP_DIR/vaws
cp -r vaws/gui $SP_DIR/vaws
cp -r vaws/model $SP_DIR/vaws
cp -r vaws/scenarios $SP_DIR/vaws
find $SP_DIR/vaws -name "output" -delete
cp vaws/__init__.py $SP_DIR/vaws
