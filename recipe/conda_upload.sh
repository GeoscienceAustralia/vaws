# Only need to change these two variables
PKG_NAME=vaws
USER=dynaryu

OS=$TRAVIS_OS_NAME-64
#mkdir ~/conda-bld
conda config --set anaconda_upload no
# export CONDA_BLD_PATH=~/conda-bld
#export VERSION=`date +%Y.%m.%d`
PACKAGE_NAME_FULL=$(conda build --output . --python=3.6)
conda build . --numpy=1.16 --python=3.6
if [ -e "$PACKAGE_NAME_FULL" ]; then
    echo $PACKAGE_NAME_FULL
    anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $PACKAGE_NAME_FULL --force
else
    echo "File does not exist"
fi
#anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $CONDA_BLD_PATH/$OS/$PKG_NAME-`date +%Y.%m.%d`-py27_1.tar.bz2 --force
#anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $PACKAGE_NAME_FULL --force
