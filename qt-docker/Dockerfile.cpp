FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils nano tcpdump
RUN apt-get update && apt-get install -y g++ make qt5-default qtbase5-dev 
RUN apt-get update && apt-get install -y autoconf  libblas-dev liblapack-dev libtool 
RUN apt-get update && apt-get install -y libboost-all-dev  \
	libgoogle-glog-dev  \
	libevent-dev	\
	libssl-dev \
	alien \
	dpkg-dev \
	debhelper \
	build-essential \
	qtbase5-private-dev \
	gpg-agent \
	g++-aarch64-linux-gnu \
	software-properties-common

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"

RUN apt-get -y update \
    && apt-get -y autoclean \
    && apt-get -y autoremove

RUN apt-get update && apt-get install -y --no-install-recommends \
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
RUN apt-get update && apt-get install -y  clang \
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

## install cmake
WORKDIR /opt
RUN cd /opt &&\
	hash -r &&\
	wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz &&\
	tar -zxvf cmake-3.16.5.tar.gz &&\
	cd cmake-3.16.5 &&\
	./bootstrap &&\
	make &&\
	make install

# install curl for the people
RUN apt-get remove curl -y
WORKDIR /opt
RUN cd /opt &&\
	git clone https://github.com/whoshuu/cpr.git &&\
	cd cpr  &&\
	git submodule update --init --recursive &&\
	mkdir -p build && cd build &&\
	cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_INSTALL_PREFIX=/usr/local .. &&\
	make -j &&\
	make install &&\
	ln -sf /opt/cpr/build/lib/libcurl-d.so /usr/local/lib/ &&\
	ldconfig

# Install poco
ARG POCO_VERSION='1.10.1'
ENV POCO_VERSION=${POCO_VERSION}
WORKDIR /opt
RUN cd /opt &&\
	git clone -b master https://github.com/pocoproject/poco.git
WORKDIR /opt/poco
RUN git checkout tags/poco-${POCO_VERSION}-release
RUN	mkdir cmake-build
WORKDIR /opt/poco/cmake-build 
RUN cmake -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
	cmake --build . --config Release && \
	cmake --build . --target install && \
	ldconfig

# dlib installation
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
ARG RUNTIME_DEPS='libpng libjpeg-turbo giflib openblas libx11'
ARG BUILD_DEPS='wget unzip cmake build-base linux-headers libpng-dev libjpeg-turbo-dev giflib-dev openblas-dev libx11-dev'
ARG LIB_PREFIX='/usr/local'
ARG DLIB_VERSION='19.22'
ENV DLIB_VERSION=${DLIB_VERSION} \
    LIB_PREFIX=${LIB_PREFIX} \
    DLIB_INCLUDE_DIR='$LIB_PREFIX/include' \
    DLIB_LIB_DIR='$LIB_PREFIX/lib'
WORKDIR /opt
RUN echo "Dlib: ${DLIB_VERSION}" \
RUN rm -rf /usr/local/lib && ln -s /usr/local/lib64 /usr/local/lib
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /opt/dlib
RUN git checkout tags/v${DLIB_VERSION} \
    && mkdir build
WORKDIR /opt/dlib/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=$LIB_PREFIX \
	-D DLIB_NO_GUI_SUPPORT=ON \
	-D DLIB_USE_BLAS=ON \
	-D DLIB_GIF_SUPPORT=ON \
	-D DLIB_PNG_SUPPORT=ON \
	-D DLIB_JPEG_SUPPORT=ON \
	-D DLIB_USE_CUDA=ON ..
RUN make -j $(getconf _NPROCESSORS_ONLN) 
RUN make install
RUN ldconfig 



RUN apt-get -qq autoremove \
&& apt-get -qq clean

RUN apt install curl -y
WORKDIR /


# Install Tensorflow 
RUN cd / && git clone https://github.com/tensorflow/tensorflow.git tensorflow_src

RUN cd tensorflow_src && git checkout 48c3bae94a8b324525b45f157d638dfd4e8c3be1

WORKDIR /tensorflow_src/tensorflow/lite/tools/make/

RUN apt-get install zip unzip -y

RUN bash download_dependencies.sh
RUN bash build_lib.sh
RUN	cp /tensorflow_src/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a /usr/local/lib


# Install Edge TPU library
## Add Edgetpu Source
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
&& apt-get update

## Install the Edge TPU runtime
RUN apt-get install libedgetpu1-std -y 
## Install dev headers
RUN apt-get install libedgetpu-dev -y 

## install the flatbuffers
WORKDIR /tensorflow_src/tensorflow/lite/tools/make/downloads/flatbuffers/build
RUN cmake  -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j4
RUN make install
RUN ldconfig

# # Build absl
# WORKDIR /tensorflow_src/tensorflow/lite/tools/make/downloads/absl/build
# RUN cmake .. -DABSL_RUN_TESTS=ON -DABSL_USE_GOOGLETEST_HEAD=ON -DCMAKE_CXX_STANDARD=11  -DCMAKE_INSTALL_PREFIX=/usr/local
# RUN cmake --build . --target all
# RUN  make install
# RUN ldconfig

## Install abseil dependency.

WORKDIR /usr/src/server/opt
RUN cd /usr/src/server/opt &&\
	git clone https://github.com/abseil/abseil-cpp.git &&\
	cd abseil-cpp &&\
	git checkout 5dd240724366295970c613ed23d0092bcf392f18 &&\
	mkdir build && cd build &&\
	cmake .. &&\
	make && make install && ldconfig


# Install QT
ENV QT_VERSION v6.1.0
ENV QT_CREATOR_VERSION v4.15.0-rc1

# Build prerequisites
RUN apt-get -y update && apt-get -y install qtbase5-dev \
	libxcb-xinerama0-dev 

# Other useful tools
RUN apt-get -y update && apt-get -y install tmux \
	zip \
	vim
	
RUN apt install qtcreator -y

#If you want Qt 5 to be the default Qt version to be used when using development binaries like qmake, install the following package:
RUN apt install qt5-default -y

#install opencv
ARG DEBIAN_FRONTEND=noninteractive
ENV OPENCV_VERSION="4.5.2"
ENV OPENCV_CONTRIB_VERSION="4.5.2" 

WORKDIR /opt
RUN set -ex \
    && apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
        libhdf5-dev \
        libopenblas-dev \
        libprotobuf-dev \
        libjpeg8 libjpeg8-dev \
        libpng16-16 libpng-dev \
        libtiff5 libtiff-dev \
        libwebp6 libwebp-dev \
        libopenjp2-7 libopenjp2-7-dev \
        tesseract-ocr tesseract-ocr-por libtesseract-dev \
        python3 python3-pip python3-numpy python3-dev \
		libjpeg-dev libpng-dev libtiff-dev \
		libavcodec-dev libavformat-dev libswscale-dev \
		libgtk2.0-dev libcanberra-gtk* \
		python3-dev python3-numpy python3-pip \
		libxvidcore-dev libx264-dev libgtk-3-dev \
		libtbb2 libtbb-dev libdc1394-22-dev \
		libv4l-dev v4l-utils \
		libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
		libavresample-dev libvorbis-dev libxine2-dev \
		libfaac-dev libmp3lame-dev libtheora-dev \
		libopencore-amrnb-dev libopencore-amrwb-dev \
		libopenblas-dev libatlas-base-dev libblas-dev \
		liblapack-dev libeigen3-dev gfortran \
		libhdf5-dev protobuf-compiler \
		libprotobuf-dev libgoogle-glog-dev libgflags-dev 

RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /opt/opencv_contrib
RUN git checkout $OPENCV_CONTRIB_VERSION
WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git
WORKDIR /opt/opencv
RUN git checkout $OPENCV_VERSION
WORKDIR /opt/opencv/build

RUN cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
		-D ENABLE_FAST_MATH=ON \
        -D WITH_JPEG=ON \
        -D WITH_PNG=ON \
        -D WITH_TIFF=ON \
        -D WITH_WEBP=ON \
        -D WITH_JASPER=ON \
        -D WITH_TBB=ON \
        -D WITH_LAPACK=ON \
        -D WITH_V4L=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_GTK=ON \
        -D WITH_QT=OFF \
        -D WITH_VTK=OFF \
        -D WITH_OPENEXR=OFF \
        -D WITH_FFMPEG=ON \
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
RUN make -j$(nproc)
RUN make install 