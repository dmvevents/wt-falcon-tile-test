/*** Include ***/
/* for general */
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <math.h>
#include <thread>
#include <mutex>
#include <unistd.h>

/* for Opencv */
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
/* for Encoder */
#include "src/base64.h"

/* for Edge TPU */
#include "edgetpu.h"
#include "model_utils.h"
#include "tensorflowlite.h"

#include <QtGlobal>

/* Model parameters */
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL 3

/*** Function ***/
static cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}

/* Declare Ml Models */
Tensorflowlite* model_obj;
Tensorflowlite* model_class;
Tensorflowlite* model_class_crop;
Tensorflowlite* model_class_child;

void initTF(){

    /* Create object detection model */
    const char* model_f_obj = "/ssd/build-peak-Desktop-Debug/assets/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
    const char* model_l_obj = "/ssd/build-peak-Desktop-Debug/assets/coco_labels.txt";
    model_obj = new Tensorflowlite(0,model_f_obj, model_l_obj,.5);


    std::cout << "Tensorflow initialization: OK" << std::endl;
}


void runObj(cv::Mat frame){
    model_obj->setFrame(frame);
    model_obj->runDet();
}

static void readLabel(const std::string filename, std::vector<std::string> &labels)
{
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        printf("failed to read %s\n", filename);
        return;
    }
    std::string str;
    while(getline(ifs, str)) {
        labels.push_back(str);
    }

    std::cout << "There are" << labels.size() << "labels" << std::endl;
}

cv::Rect cropMat(int h, int w, int xmin, int ymin, int xmax, int ymax){

    // Get widths of box
    int bw = xmax - xmin;
    int bh = ymax - ymin;

    // Get center of box
    int bcx = xmin + bw/2;
    int bcy = ymin + bh/2;

    // Pad boundry condition
    int p = 50;
    int c_delta = std::max(bw,bh)/2;

    int pxmin = p + c_delta;
    int pxmax = w - pxmin;
    int pymin = pxmin;
    int pymax = h - pymin;


    // assign center based on boundry conditions of x
    int cx = 0;
    if (bcx < pxmin) {
      cx = pxmin;
    }
    else if (bcx > pxmax){
      cx = pxmax;
    }
    else{
      cx = bcx;
    }

    // assign center based on boundry conditions of y
    int cy = 0;
    if (bcy < pymin){
      cy = pymin;
    }
    else if (bcy > pymax){
      cy  = pymax;
    }
    else{
      cy = bcy;
    }

    // New BBox cordinates
    int delta = p + c_delta;
    int x0 = cx - delta;
    w = delta*2;
    int y0 = cy - delta;
    h = w;

    // Setup a rectangle to define your region of interest
    cv::Rect myROI(x0, y0, w, h);


    return myROI;
}

std::string encodeBase64(cv::Mat frame){
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf);

    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string encoded =  base64_encode(enc_msg, buf.size());

    return encoded;
}


QJsonDocument loadJson(QString fileName) {
    QFile jsonFile(fileName);
    jsonFile.open(QFile::ReadOnly);
    return QJsonDocument().fromJson(jsonFile.readAll());
}


QJsonObject getJson(int w, int h, int xmin, int ymin, int xmax, int ymax, QString detectClass, QString time,
                                    QString cropName, QString cropClassChild, QString camera){


    QString path = QDir::currentPath();

    QJsonObject json_obj;

    json_obj["detectedAt"] = time;
    json_obj["cropLabel"] =  cropClassChild;
    json_obj["picLabel"]=  detectClass;
    json_obj["width"]=  QString::number(w);
    json_obj["height"]=  QString::number(h);
    json_obj["xMin"]= QString::number(xmin);
    json_obj["yMin"]= QString::number(ymin);
    json_obj["xMax"]= QString::number(xmax);
    json_obj["yMax"]= QString::number(ymax);
    json_obj["camera"] = camera;
    json_obj["depth"] =  3;

    return json_obj;
}

QJsonObject data;

int main(int argc, char *argv[])
{

    initTF();

    const std::string filename = "../Test.mp4";
    cv::VideoCapture cap(filename);

    /*
     * Read in the time code
     */

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" <<  std::endl;
        return -1;
    }

    double fps;
    int frameNum = 0;
    // Start and end times
    time_t start, end;

    // Start time
    time(&start);

    while(1){
        cv::Mat frame;

        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty()){
            break;
        }

        // Set frames
        std::thread tObj(runObj, frame);

        // Inc PLC
        tObj.join();

        //TODO POINTER
        OUTPUT_OBJ result = model_obj->getObjResults();

        //TODO POINTER
        std::vector<std::string> labels_obj = model_obj->getLabels();


        for (const auto& object : result.objectList) {
            cv::rectangle(frame, cv::Rect(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y), static_cast<int32_t>(object.width), static_cast<int32_t>(object.height)), cv::Scalar(255, 255, 0), 3);
            cv::putText(frame, labels_obj[object.classId].c_str(), cv::Point(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y) + 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 0, 0), 3);

            if (object.hasCrop == true){


                //std::string Tensorflowlite::getJson(int w, int h, int xmin, int ymin, int xmax, int ymax, QString detectClass, QString dirLabel, QString time){

                int xmin = static_cast<int32_t>(object.x);
                int ymin = static_cast<int32_t>(object.y);
                int xmax = xmin + static_cast<int32_t>(object.width);
                int ymax = ymin + static_cast<int32_t>(object.height);

                std::string detectClass = labels_obj[object.classId].c_str();
            }
        }
        /*
        */

        // to half size or even smaller
        // resize(frame, frame, cv::Size(frame.cols/4, frame.rows/4));

        cv::namedWindow( "Display frame",cv::WINDOW_AUTOSIZE);
        imshow("Display frame", frame);


        //Increase frame count
        frameNum++;

        char c=(char)cv::waitKey(10);
        if(c==27)
            break;
    }

    // End Time
    time(&end);

    // Time elapsed
    double seconds = difftime (end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // Calculate frames per second
    fps  = frameNum / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;
    std::cout << "Frames : " << frameNum << std::endl;

    // When everything done, release
    cap.release();
    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
