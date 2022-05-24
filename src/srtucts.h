#ifndef SRTUCTS_H
#define SRTUCTS_H

/* for Opencv */
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
/* for Encoder */
#include "src/base64.h"
#include <QString>
#include <QJsonObject>

#include <QFile>
struct detStruct{
    cv::Mat org;
    cv::Mat dst;
    cv::Mat crop;
    QJsonObject dataObject;
    QString savePathBase;
    std::string sendDefAddress;
};Q_DECLARE_METATYPE(detStruct)

struct frameStruct{
//    cv::Mat frame;
//    QString camera;
    std::string csAddress;
    int csPort;
    std::string jsonStringObj;
};Q_DECLARE_METATYPE(frameStruct)


#endif // SRTUCTS_H
