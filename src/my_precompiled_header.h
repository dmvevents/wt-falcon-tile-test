#ifndef MY_PRECOMPILED_HEADER_H
#define MY_PRECOMPILED_HEADER_H
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#include <memory>



/* for Opencv */
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/tracking.hpp>

/* for Encoder */
#include "src/base64.h"

/* for Edge TPU */
#include "src/edgetpu.h"
#include "src/model_utils.h"
#include "src/tensorflowlite.h"

#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor.h"

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"

/* for CPR */
#include <cpr/cpr.h>


#include <QString>
#include <QXmlStreamWriter>
#include <QFile>
#include <QDir>
#include <QJsonObject>
#include <QByteArray>
#include <QJsonDocument>
#include <QDebug>

//Tracker
#include "model_utils.h"


#include<utility>

#endif // MY_PRECOMPILED_HEADER_H
