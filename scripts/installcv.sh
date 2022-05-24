cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

mv  opencv_contrib opencv
echo "Create a virtual environtment for the python binding module"
pip install virtualenv virtualenvwrapper
rm -rf ~/.cache/pip
echo "Edit ~/.bashrc"
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv cv -p python3
pip install numpy

echo "Procced with the installation"
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D WITH_CUDA=OFF \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/cv/lib/python3.6/site-packages \
-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
-D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
-D BUILD_EXAMPLES=OFF ..

nproc
make -j4
make install
ldconfig