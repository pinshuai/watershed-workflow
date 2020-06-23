# NOTE: this requires that you have defined CONDA_PREFIX to point to your Anaconda environment's installation.

CC=`which clang`  # `which gcc` for Linux
CXX=`which clang++`  # `which g++` for Linux
FC=`which gfortran`

CONDA_PREFIX=/opt/anaconda3/envs/watershed_workflow
SEACAS_SRC_DIR=/Users/shua784/Dropbox/github/seacas
SEACAS_BUILD_DIR=${SEACAS_SRC_DIR}/build  # these can be changed
SEACAS_DIR=${SEACAS_SRC_DIR}/install  # these can be changed


mkdir -p $SEACAS_BUILD_DIR
mkdir -p $SEACAS_DIR
cd $SEACAS_BUILD_DIR

cmake  \
    -D SEACASProj_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D SEACASProj_ENABLE_SEACASExodus:BOOL=ON \
    -D CMAKE_INSTALL_PREFIX:PATH=${SEACAS_DIR} \
    -D CMAKE_BUILD_TYPE=Debug \
    -D BUILD_SHARED_LIBS:BOOL=ON \
    \
    -D CMAKE_CXX_COMPILER:FILEPATH=${CXX} \
    -D CMAKE_C_COMPILER:FILEPATH=${CC} \
    -D CMAKE_Fortran_COMPILER:FILEPATH=${FC} \
    -D SEACASProj_SKIP_FORTRANCINTERFACE_VERIFY_TEST:BOOL=ON \
    -D TPL_ENABLE_Netcdf:BOOL=ON \
    -D TPL_ENABLE_HDF5:BOOL=ON \
    -D TPL_ENABLE_Matio:BOOL=OFF \
    -D TPL_ENABLE_MPI=OFF \
    -D TPL_ENABLE_CGNS:BOOL=OFF \
    \
    -D Netcdf_LIBRARY_DIRS:PATH=${CONDA_PREFIX}/lib \
    -D Netcdf_INCLUDE_DIRS:PATH=${CONDA_PREFIX}/include \
    -D HDF5_ROOT:PATH=${CONDA_PREFIX} \
    -D HDF5_NO_SYSTEM_PATHS=ON \
${SEACAS_SRC_DIR}


