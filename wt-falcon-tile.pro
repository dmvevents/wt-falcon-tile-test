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
CONFIG +=  TENSORFLOW  EDGETPU OPENCV

OPENCV {PKGCONFIG +=  opencv4}
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


SOURCES += \
    src/main.cpp \
    src/base64.cpp \
    src/tensorflowlite.cpp \
    src/model_utils.cpp \
    src/nms.cpp

HEADERS += \
    src/base64.h \
    src/tensorflowlite.h \
    src/my_precompiled_header.h \
    src/model_utils.h \
    src/nms.h


PRECOMPILED_HEADER = src/my_precompiled_header.h
CONFIG += precompile_header


DISTFILES += \
    assets/config.json \
    Dockerfile

