sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt-get -y update \
&& sudo apt-get -y autoclean \
&& sudo apt-get -y autoremove

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
apt-utils \
nano \
tcpdump \
pkg-config \
software-properties-common \
g++\
make \
qt5-default \
qtbase5-dev \
autoconf \
alien \
dpkg-dev \
debhelper \
build-essential \
qtbase5-private-dev \
git \
wget

# Install the C++ dependencies
sudo apt-get update && sudo apt-get install -y  clang \
libblas-dev \
liblapack-dev \
libtool \
libboost-all-dev  \
libgoogle-glog-dev  \
libevent-dev\
libssl-dev \
libgtk2.0-dev \
libavcodec-dev \
libavformat-dev \
libswscale-dev\
libtbb2 \
libtbb-dev \
libjpeg-dev \
libpng-dev \
libtiff-dev \
libjasper-dev \
libeigen3-dev \
libatlas-base-dev \
libgomp1 \
libx264-dev

# install curl for the people
sudo apt-get remove curl -y

cd ~ &&\
mkdir opt

# install cmake
cd ~/opt 
hash -r
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar -zxvf cmake-3.16.5.tar.gz 
cd cmake-3.16.5
./bootstrap 
make 
sudo make install

# install cpr
cd ~/opt \
&& rm -rf cpr \
&& git clone https://github.com/whoshuu/cpr.git \
&& cd cpr \
&& git checkout f4622efcb59d84071ae11404ae61bd821c1c344b \
&& git submodule update --init --recursive 
&& mkdir -p build && cd build \
&& cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j \
&& sudo make install \
&& sudo ln -sf ~/opt/cpr/build/lib/libcurl-d.so /usr/local/lib/ \
&& sudo ldconfig

# install libmodbus
MOD_VERSION='3.0.8'
cd ~/opt \
&& sudo rm -rf libmodbus \
&& git clone https://github.com/stephane/libmodbus.git \
&& cd libmodbus \
&& git checkout tags/v-${POCO_VERSION}-release \
&& git submodule update --init --recursive \
&& ./autogen.sh \
&& ./configure --prefix=/usr/local/\
&& sudo make install \
&& sudo ldconfig

# Install poco
POCO_VERSION='1.10.1'
cd ~/opt \
&& git clone -b master https://github.com/pocoproject/poco.git \
&& cd ~/opt/poco \
&& git checkout tags/poco-${POCO_VERSION}-release \
&& mkdir cmake-build \
&& cd ~/opt/poco/cmake-build \
&& cmake -D CMAKE_INSTALL_PREFIX=/usr/local .. \
&& cmake --build . --config Release \
&& sudo cmake --build . --target install \
&& sudo ldconfig


# Install Edge TPU library
## Add Edgetpu Source
sudo echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list \
&& sudo apt-get install curl -y \
&& sudo curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - \
&& sudo apt-get update \
&& sudo apt-get install libedgetpu1-std -y \
&& sudo apt-get install libedgetpu-dev -y 

cd ~/opt \
&& sudo rm -rf tensorflow \
&& sudo rm -rf tensorflow_src \
&& git clone https://github.com/tensorflow/tensorflow tensorflow_src \
&& cd tensorflow_src \
&& git checkout 48c3bae94a8b324525b45f157d638dfd4e8c3be1 \
&& cd tensorflow/lite/tools/make/ 
#&& ./configure
## Run the C++ installation
bash build_lib.sh

## install the flatbuffers
cd ~/opt/tensorflow_src/tensorflow/lite/tools/make/downloads/flatbuffers/ \
&& mkdir build && cd build \
&& cmake  -D CMAKE_INSTALL_PREFIX=/usr/local .. \
&& make -j4 \
&& sudo  make install \
&& sudo  ldconfig

cd ~/opt/tensorflow_src/tensorflow/lite/tools/make/downloads/absl/ \
&& mkdir build && cd build \
&& cmake .. -DABSL_RUN_TESTS=ON -DABSL_USE_GOOGLETEST_HEAD=ON -DCMAKE_CXX_STANDARD=11  -DCMAKE_INSTALL_PREFIX=/usr/local \
&& cmake --build . --target all \
&& sudo make install \
&& sudo ldconfig

OPENCV_VERSION="4.5.2"
OPENCV_CONTRIB_VERSION="4.5.2" 
ARCH_BIN="7.5"
set -ex \
&& sudo apt-get -qq update \
&& sudo apt-get -qq install -y --no-install-recommends \
	libhdf5-dev \
	libopenblas-dev \
	libprotobuf-dev \
	libjpeg8 libjpeg8-dev \
	libpng16-16 libpng-dev \
	libtiff5 libtiff-dev \
	libwebp6 libwebp-dev \
	libopenjp2-7 libopenjp2-7-dev \
	tesseract-ocr tesseract-ocr-por libtesseract-dev sudo libcanberra-gtk-module \
	python3 python3-pip python3-numpy python3-dev
cd ~/opt/ 
git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
cd  ~/opt/opencv_contrib
git checkout $OPENCV_CONTRIB_VERSION
cd ~/opt/opencv
RUN git checkout $OPENCV_VERSION
mkdir ~/opt/opencv/build && cd ~/opt/opencv/build
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opt/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D OPENCV_ENABLE_NONFREE=ON \
-D ENABLE_FAST_MATH=ON \
-D WITH_JPEG=ON \
-D WITH_PNG=ON \
-D WITH_TIFF=ON \
-D WITH_WEBP=ON \
-D WITH_JASPER=ON \
-D WITH_EIGEN=ON \
-D WITH_TBB=ON \
-D WITH_LAPACK=ON \
-D WITH_PROTOBUF=ON \
-D WITH_V4L=OFF \
-D WITH_GSTREAMER=OFF \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D WITH_VTK=OFF \
-D WITH_OPENEXR=OFF \
-D WITH_FFMPEG=OFF \
-D WITH_OPENCL=OFF \
-D WITH_OPENNI=OFF \
-D WITH_XINE=OFF \
-D WITH_GDAL=OFF \
-D WITH_IPP=OFF \
-D BUILD_OPENCV_PYTHON3=OFF \
-D BUILD_OPENCV_PYTHON2=OFF \
-D BUILD_OPENCV_JAVA=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_IPP_IW=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_ANDROID_EXAMPLES=OFF \
-D OPENCV_GENERATE_PKGCONFIG=YES \
-D BUILD_DOCS=OFF \
-D BUILD_ITT=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_TESTS=OFF ..
# -D CUDA_ARCH_BIN=${ARCH_BIN} \
# -D CUDA_ARCH_PTX="" \
# -D WITH_CUDNN=ON \
# -D WITH_CUBLAS=ON \
# -D CUDA_FAST_MATH=ON \
# -D WITH_CUDA=ON \
make -j$(nproc) \
&& sudo make install 

# Install QT
sudo apt-get install qtcreator qt5-default -y

# Set up Git
git config --global user.email "aa@blueridge.ai"
git config --global user.name "Anton Alexander"

sudo apt install swig libjpeg-dev zlib1g-dev python3-dev python3-numpy
pip install numpy pybind11
pip3 install wheel 
sudo apt-get install python3-setuptools
sh tensorflow/lite/tools/make/download_dependencies.sh
sh tensorflow/lite/tools/pip_package/build_pip_package.sh