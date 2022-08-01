CONFIG += console c++17
QT += core
QT +=  gui

CONFIG   -= app_bundle
TEMPLATE = app


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# Copy the assets path
copydata.commands = $(COPY_DIR) $$PWD/assets $$OUT_PWD
first.depends = $(first) copydata
export(first.depends)
export(copydata.commands)

QMAKE_EXTRA_TARGETS += first copydata

# Add modbus libary
CONFIG += link_pkgconfig
CONFIG +=  TENSORFLOW  EDGETPU OPENCV HCNETSDK

OPENCV {PKGCONFIG +=  opencv}
EDGETPU {LIBS += -l:libedgetpu.so.1.0}

HOME = $$system(echo $HOME)
message($$HOME)


TENSORFLOW {
    LIBS +=  -L/tensorflow_src/tensorflow/lite/tools/make/gen/linux_x86_64/lib/ -ltensorflow-lite
    INCLUDEPATH += /tensorflow_src
    DEPENDPATH += /tensorflow_src
    PRE_TARGETDEPS += /tensorflow_src/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a
    LIBS += -ldl
}

HCNETSDK {
    LIBS +=  -lpthread
    LIBS += -lboost_thread -lboost_system
    LIBS += -L//home/watad/Downloads/EN-HCNetSDKV6.1.6.3_build20200925_Linux64/lib -lPlayCtrl -lAudioRender -lSuperRender -lhcnetsdk
    INCLUDEPATH += //home/watad/Downloads/EN-HCNetSDKV6.1.6.3_build20200925_Linux64/incEn
    DEPENDPATH += //home/watad/Downloads/EN-HCNetSDKV6.1.6.3_build20200925_Linux64/incEn
    INCLUDEPATH += /home/watad/hik-camera/

}

#/home/watad/Downloads/EN-HCNetSDKV6.1.6.3_build20200925_Linux64

SOURCES += \
    src/main.cpp \
    src/base64.cpp \
    src/tensorflowlite.cpp \
    src/model_utils.cpp \
    src/nms.cpp \
    /home/watad/hik-camera/HKIPCamCapture.cpp \
    /home/watad/hik-camera/hkipcamera.cpp \

HEADERS += \
    src/base64.h \
    src/tensorflowlite.h \
    src/my_precompiled_header.h \
    src/model_utils.h \
    src/nms.h \
    /home/watad/hik-camera/HKIPCamCapture.h \
    /home/watad/hik-camera/hkipcamera.h \


PRECOMPILED_HEADER = src/my_precompiled_header.h
CONFIG += precompile_header


DISTFILES += \
    assets/config.json \
    Dockerfile \
    assets/config_dell.json

