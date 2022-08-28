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

#include <hkhelper.h>


#include "HKIPCamCapture.h"


long channel = 1;
long streamtype = 1;
HKIPCamCapture _hkipc1;
HKIPCamCapture _hkipc2;
HKIPCamCapture _hkipc3;

std::string ip1 = "192.168.1.64";
std::string ip2 = "192.168.1.65";
std::string ip3 = "192.168.1.66";

std::string username = "admin";
std::string password = "Watad@2022";
long port = 8000;


struct PredictionResult final
{
    /** OpenCV rectangle which describes where the object is located in the original image.
     *
     * Given this example annotated 230x134 image:
     * @image html xkcd_bike.png
     * The red rectangle returned would be:
     * @li @p rect.x = 96 (top left)
     * @li @p rect.y = 38 (top left)
     * @li @p rect.width = 108
     * @li @p rect.height = 87
     *
     * @see @ref DarkHelp::PredictionResult::original_point @see @ref DarkHelp::PredictionResult::original_size
     */
    cv::Rect rect;

    /** The original normalized X and Y coordinate returned by darknet.  This is the normalized mid-point, not the corner.
     * If in doubt, you probably want to use @p rect.x and @p rect.y instead of this value.
     *
     * Given this example annotated 230x134 image:
     * @image html xkcd_bike.png
     * The @p original_point returned would be:
     * @li @p original_point.x = 0.652174 (mid x / image width, or 150 / 230)
     * @li @p original_point.y = 0.608209 (mid y / image height, or 81.5 / 134)
     *
     * @see @ref DarkHelp::PredictionResult::rect @see @ref DarkHelp::PredictionResult::original_size
     */
    cv::Point2f original_point;

    /** The original normalized width and height returned by darknet.  If in doubt, you probably want to use
     * @p rect.width and @p rect.height instead of this value.
     *
     * Given this example annotated 230x134 image:
     * @image html xkcd_bike.png
     * The @p original_size returned would be:
     * @li @p original_size.width  = 0.469565 (rect width / image width, or 108 / 230)
     * @li @p original_size.height = 0.649254 (rect height / image height, or 87 / 134)
     *
     * @see @ref DarkHelp::PredictionResult::rect @see @ref DarkHelp::PredictionResult::original_point
     */
    cv::Size2f original_size;

    /** This is only useful if you have multiple classes, and an object may be one of several possible classes.
     *
     * @note This will contain all @em non-zero class/probability pairs.
     *
     * For example, if your classes in your @p names file are defined like this:
     * ~~~~{.txt}
     * car
     * person
     * truck
     * bus
     * ~~~~
     *
     * Then an image of a truck may be 10.5% car, 0% person, 95.8% truck, and 60.3% bus.  Only the non-zero
     * values are ever stored in this map, which for this example would be the following:
     *
     * @li 0 -> 0.105 // car
     * @li 2 -> 0.958 // truck
     * @li 3 -> 0.603 // bus
     *
     * The C++ map would contains the following values:
     *
     * ~~~~
     * all_probabilities = { {0, 0.105}, {2, 0.958}, {3, 0.603} };
     * ~~~~
     *
     * (Note how @p person is not stored in the map, since the probability for that class is 0%.)
     *
     * In addition to @p %all_probabilities, the best results will @em also be duplicated in @ref DarkHelp::PredictionResult::best_class
     * and @ref DarkHelp::PredictionResult::best_probability, which in this example would contain the values representing the truck:
     *
     * @li @ref DarkHelp::PredictionResult::best_class == 2
     * @li @ref DarkHelp::PredictionResult::best_probability == 0.958
     */
    std::map<int, float> all_probabilities;

    /** The class that obtained the highest probability.  For example, if an object is predicted to be 80% car
     * or 60% truck, then the class id of the car would be stored in this variable.
     * @see @ref DarkHelp::PredictionResult::best_probability
     * @see @ref DarkHelp::PredictionResult::all_probabilities
     */
    int best_class;

    /** The probability of the class that obtained the highest value.  For example, if an object is predicted to
     * be 80% car or 60% truck, then the value of 0.80 would be stored in this variable.
     * @see @ref DarkHelp::PredictionResult::best_class
     * @see @ref DarkHelp::PredictionResult::all_probabilities
     */
    float best_probability;

    /** A name to use for the object.  If an object has multiple probabilities, then the one with the highest
     * probability will be listed first.  For example, a name could be @p "car 80%, truck 60%".  The @p name
     * is used as a label when calling @ref DarkHelp::NN::annotate().
     * @see @ref DarkHelp::Config::names_include_percentage
     */
    std::string name;

    /** The tile number on which this object was found.  This is mostly for debug purposes and only if tiling
     * has been enabled (see @ref DarkHelp::Config::enable_tiles), otherwise the value will always be zero.
     */
    int tile;
};

/* Model parameters */
#define MODEL_WIDTH 224
#define MODEL_HEIGHT 224
#define MODEL_CHANNEL 3


//std::vector<std::pair<int, int>>      dims{{1920,1920}};
//    std::vector<std::pair<int, int>>        dims{{400,300},{500,500}};
//std::vector<std::pair<int, int>>        dims{{1280,720}};

float tile_rect_factor					= 10.20f;
float tile_edge_factor					= 40.25f;
bool only_combine_similar_predictions   = true;
bool combine_tile_predictions           = true;

bool enable_debug                       = true;
// Init multi frame capture vars for tracking
double fps                              = 0;
int frameNum                            = 0;
std::vector<PredictionResult>           results;

cv::Mat                                 frame;



std::vector<size_t>                     indexes_of_predictions_near_edges;
/* Create Locks for Threads*/
std::mutex                              mutex;
std::vector<std::string>                labels_obj;

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
Tensorflowlite* model_obj1;
Tensorflowlite* model_obj2;
Tensorflowlite* model_obj3;
Tensorflowlite* model_obj4;
Tensorflowlite* model_obj5;
Tensorflowlite* model_obj6;
Tensorflowlite* model_obj_org;

void initTF(){

    /* Create object detection model */
    const char* model_f_obj = "/home/watad/build-wt-falcon-tile-Desktop-Debug/assets/ssdlite_mobilenet_tile_v1_edgetpu.tflite";
    const char* model_l_obj = "/home/watad/build-wt-falcon-tile-Desktop-Debug/assets/labels_tile.txt";
    const double con = .5;
    model_obj = new Tensorflowlite(0,model_f_obj, model_l_obj,con,true);
    labels_obj     = model_obj->getLabels();

    model_obj1 = new Tensorflowlite(1,model_f_obj, model_l_obj,con,true);
    model_obj2 = new Tensorflowlite(2,model_f_obj, model_l_obj,con,true);
    model_obj3 = new Tensorflowlite(3,model_f_obj, model_l_obj,con,true);
    model_obj4 = new Tensorflowlite(4,model_f_obj, model_l_obj,con,true);
    model_obj5 = new Tensorflowlite(5,model_f_obj, model_l_obj,con,true);
    model_obj6 = new Tensorflowlite(6,model_f_obj, model_l_obj,con,true);

//    model_obj7 = new Tensorflowlite(7,model_f_obj, model_l_obj,.5);
//    model_obj8 = new Tensorflowlite(8,model_f_obj, model_l_obj,.5);
//    model_obj9 = new Tensorflowlite(9,model_f_obj, model_l_obj,.5);
//    model_obj10 = new Tensorflowlite(10,model_f_obj, model_l_obj,.5);
//    model_obj11 = new Tensorflowlite(11,model_f_obj, model_l_obj,.5);

    model_obj_org = new Tensorflowlite(7,
                                       "/home/watad/wt-sadara-vision-main/assets/ssdlite_mobiledet_watad_edgetpu.tflite",
                                       "/home/watad/build-wt-falcon-tile-Desktop-Debug/assets/labels.txt",
                                       con,
                                       true);

    std::cout << "Tensorflow initialization: OK" << std::endl;
}


void runObj(cv::Mat frame){
    model_obj_org->setFrame(frame);
    model_obj_org->runDet();
}

int runtile(cv::Mat roi, int tile_count, int tile_edge_factor, int x_offset, int y_offset){

    OUTPUT_OBJ result;
    if(tile_count == 0){
        model_obj->setFrame(roi);
        model_obj->runDet();
        result = model_obj->getObjResults();

    }
    else if(tile_count == 1){
        model_obj1->setFrame(roi);
        model_obj1->runDet();
        result = model_obj1->getObjResults();
//        std::cout << result.objectNum << std::endl;

    }
    else if(tile_count == 2){
        model_obj2->setFrame(roi);
        model_obj2->runDet();
        result = model_obj2->getObjResults();

    }
    else if(tile_count == 3){
        model_obj3->setFrame(roi);
        model_obj3->runDet();
        result = model_obj3->getObjResults();

    }
    else if(tile_count == 4){
        model_obj4->setFrame(roi);
        model_obj4->runDet();
        result = model_obj4->getObjResults();

    }
    else if(tile_count == 5){
        model_obj5->setFrame(roi);
        model_obj5->runDet();
        result = model_obj5->getObjResults();

    }
    else{
        return 0;
    }


    //TODO POINTER

    // fix up the predictions -- need to compensate for the tile not being the top-left corner of the image, and the size of the tile being smaller than the image

    int counter =0;
    for (const auto& object : result.objectList)
    {
//        if(tile_count == 1){
//            counter++;
//            std::cout << "Counter: " << counter<< std::endl;
//        }
        // Create object for prediction to store
        PredictionResult prediction;

        float score = static_cast<float>(object.score);
        int x0 = static_cast<int32_t>(object.x);
        int y0 = static_cast<int32_t>(object.y);

        int width = static_cast<int32_t>(object.width);
        int height = static_cast<int32_t>(object.height);
        prediction.original_point.x = static_cast<int32_t>((x0+width)/roi.cols);
        prediction.original_point.y = static_cast<int32_t>((y0+height)/roi.rows);

        prediction.original_size.width =static_cast<int32_t>(object.width/roi.cols);;
        prediction.original_size.height = static_cast<int32_t>(object.height/roi.rows);;

        prediction.best_class = static_cast<int32_t>(object.classId);
        prediction.best_probability = score;

        prediction.name = labels_obj[prediction.best_class];
        prediction.tile = tile_count;

        prediction.rect.x = x0;
        prediction.rect.y =  y0;

        prediction.rect.width = width;
        prediction.rect.height = height;

        //track which predictions are near the edges, because we may need to re-examine them and join them after we finish with all the tiles

        mutex.lock();
        if (combine_tile_predictions)
        {
            const int minimum_horizontal_distance	= tile_edge_factor * prediction.rect.width;
            const int minimum_vertical_distance		= tile_edge_factor * prediction.rect.height;
            if (prediction.rect.x <= minimum_horizontal_distance					or
                prediction.rect.y <= minimum_vertical_distance						or
                roi.cols - prediction.rect.br().x <= minimum_horizontal_distance	or
                roi.rows - prediction.rect.br().y <= minimum_vertical_distance		)
            {
                // this prediction is near one of the tile borders so we need to remember it
                indexes_of_predictions_near_edges.push_back(results.size());
            }
        }

        // every prediction needs to have x_offset and y_offset added to it
        prediction.rect.x += x_offset;
        prediction.rect.y += y_offset;

        if (enable_debug)
        {
            // draw a black-on-white debug label on the top side of the annotation

            const std::string label		= labels_obj[prediction.best_class];
            const auto font				= cv::HersheyFonts::FONT_HERSHEY_PLAIN;
            const auto scale			= 0.75;
            const auto thickness		= 1;
            int baseline				= 0;
            const cv::Size text_size	= cv::getTextSize(label, font, scale, thickness, &baseline);
            const int text_half_width	= text_size.width			/ 2;
            const int text_half_height	= text_size.height			/ 2;
            const int pred_half_width	= prediction.rect.width		/ 2;
            const int pred_half_height	= prediction.rect.height	/ 2;

            // put the text exactly in the middle of the prediction
            const cv::Rect label_rect(
                    prediction.rect.x + pred_half_width - text_half_width,
                    prediction.rect.y + pred_half_height - text_half_height,
                    text_size.width, text_size.height);
            cv::rectangle(frame, label_rect, {255, 255, 255}, cv::FILLED, cv::LINE_AA);
            cv::putText(frame, label, cv::Point(label_rect.x, label_rect.y + label_rect.height), font, scale, cv::Scalar(0,0,0), thickness, cv::LINE_AA);
        }

        // the original point and size are based on only 1 tile, so they also need to be fixed
        prediction.original_point.x = (static_cast<float>(prediction.rect.x) + static_cast<float>(prediction.rect.width	) / 2.0f) / static_cast<float>(frame.cols);
        prediction.original_point.y = (static_cast<float>(prediction.rect.y) + static_cast<float>(prediction.rect.height) / 2.0f) / static_cast<float>(frame.rows);
        prediction.original_size.width	= static_cast<float>(prediction.rect.width	) / static_cast<float>(frame.cols);
        prediction.original_size.height	= static_cast<float>(prediction.rect.height	) / static_cast<float>(frame.rows);

        results.push_back(prediction);

        mutex.unlock();
    }
    return 1;

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

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500 )
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        //roi.y = ( target_width - roi.height ) / 2;
        roi.y = target_width - roi.height;

    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

int main(int argc, char *argv[])
{

//    auto conn_param1 = _hkipc1.getConnectParam();
//    conn_param1.ip = ip1;
//    conn_param1.username = username;
//    conn_param1.password = password;
//    conn_param1.port = 8004;
//    conn_param1.channel = channel;
//    conn_param1.streamtype = streamtype;
//    conn_param1.link_mode = 1;
//    conn_param1.device_id = -1;
//    conn_param1.buffer_size = 20;
//    _hkipc1.setConnectParam(conn_param1);

//    auto conn_param2 = _hkipc2.getConnectParam();
//    conn_param2.ip = ip2;
//    conn_param2.username = username;
//    conn_param2.password = password;
//    conn_param2.port = 8005;
//    conn_param2.channel = channel;
//    conn_param2.streamtype = streamtype;
//    conn_param2.link_mode = 1;
//    conn_param2.device_id = -1;
//    conn_param2.buffer_size = 20;
//    _hkipc2.setConnectParam(conn_param2);

//    auto conn_param3 = _hkipc3.getConnectParam();
//    conn_param3.ip = ip3;
//    conn_param3.username = username;
//    conn_param3.password = password;
//    conn_param3.port = 8006;
//    conn_param3.channel = channel;
//    conn_param3.streamtype = streamtype;
//    conn_param3.link_mode = 1;
//    conn_param3.device_id = -1;
//    conn_param3.buffer_size = 20;
//    _hkipc3.setConnectParam(conn_param3);

    initTF();

    // Set labels
    std::vector<std::string> labels_obj     = model_obj->getLabels();

    // Start and end times
    time_t start, end;

    // Start time
    time(&start);

    // Start the timer
    std::chrono::high_resolution_clock::duration total_duration
                                            = std::chrono::milliseconds(0);

    //const std::string filename              = "/home/watad/wt-sadara-vision-main/assets/2ppl.mp4";
    //const std::string filename              = "/home/anton/Git/pycoral/test_data/kite_and_cold.jpg";
//    const std::string filename1              = "rtsp://admin:Watad@2022@192.168.1.64:554/Streaming/Channels/101/";
//    const std::string filename2              = "rtsp://admin:Watad@2022@192.168.1.65:554/Streaming/Channels/101/";
//    const std::string filename3              = "rtsp://admin:Watad@2022@192.168.1.66:554/Streaming/Channels/101/";

//    cv::VideoCapture cap1(filename1);
//    cv::VideoCapture cap2(filename2);
//    cv::VideoCapture cap3(filename3);


//    // Check if camera opened successfully
//    if(!cap1.isOpened()){
//        std::cout << "Error opening video stream or file" <<  std::endl;
//        return -1;
//    }

//    if(!cap2.isOpened()){
//        std::cout << "Error opening video stream or file" <<  std::endl;
//        return -1;
//    }

//    if(!cap3.isOpened()){
//        std::cout << "Error opening video stream or file" <<  std::endl;
//        return -1;
//    }

//    _hkipc1.open();
//    _hkipc2.open();
//    _hkipc3.open();
    while(1){
        //cv::Mat frame;

        //frame = cv::imread(filename);
        // Capture frame-by-frame

//        cap1.grab();
//        cap2.grab();
//        cap3.grab();

//        cap1.retrieve(frame1);
//        cap2.retrieve(frame2);
//        cap3.retrieve(frame3);

        cv::Mat                                 frame1;
        cv::Mat                                 frame2;
        cv::Mat                                 frame3;
        _hkipc1.read(frame1);
        _hkipc2.read(frame2);
        _hkipc3.read(frame3);

        if (frame1.empty())
        {
            /// @throw std::invalid_argument if the image is empty.
            break;
        }

        if (frame2.empty())
        {
            /// @throw std::invalid_argument if the image is empty.
            break;
        }
        cv::rotate(frame3, frame3, cv::ROTATE_90_CLOCKWISE);

        cv::Mat tmpFrame;
        cv::Mat frame1Tmp = GetSquareImage(frame1, 1280 );
        cv::Mat frame2Tmp = GetSquareImage(frame2, 1280 );

        hconcat(frame1Tmp, frame2Tmp, tmpFrame);
        hconcat(tmpFrame, frame3, frame);
//        std::vector<std::pair<int, int>>        dims{{frame.cols/1,frame.rows/1},{frame.cols/3,frame.rows/2}};
        std::vector<std::pair<int, int>>        dims{{frame.cols/3,frame.rows/2}};

        // divide the original image into the right number of tiles and call predict() on each tile

        indexes_of_predictions_near_edges.clear();
        std::vector<cv::Mat> all_tile_mats;
        results.clear();

        for( int d=0; d < dims.size(); d++) {
            const float horizontal_factor		= static_cast<float>(frame.cols) / static_cast<float>(dims.at(d).first);
            const float vertical_factor			= static_cast<float>(frame.rows) / static_cast<float>(dims.at(d).second);

            const float horizontal_tiles_count	= std::round(std::max(1.0f, horizontal_factor	));
            const float vertical_tiles_count	= std::round(std::max(1.0f, vertical_factor		));

            // Set tile width
            const float tile_width				= static_cast<float>(frame.cols) / horizontal_tiles_count;
            const float tile_height				= static_cast<float>(frame.rows) / vertical_tiles_count;

            // Create tile size
            const cv::Size new_tile_size		= cv::Size(std::round(tile_width), std::round(tile_height));

            if (horizontal_tiles_count == 1 and vertical_tiles_count == 1)
            {
                // image is smaller than (or equal to) the network, so use the original predict() call
                //return predict_internal(frame, new_threshold);

                // Set frames
               std::thread tObj(runObj, frame);

               // Inc PLC
               tObj.join();

               //TODO POINTER
               OUTPUT_OBJ result = model_obj_org->getObjResults();
               auto labels_obj_org     = model_obj_org->getLabels();
               for (const auto& object : result.objectList) {
                   cv::rectangle(frame, cv::Rect(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y), static_cast<int32_t>(object.width), static_cast<int32_t>(object.height)), cv::Scalar(255, 255, 0), 3);
                   cv::putText(frame, labels_obj_org[object.classId].c_str(), cv::Point(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y) + 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 0, 0), 3);

               }
            }
            else{

                std::vector<std::thread> t;

                // otherwise, if we get here then we have more than 1 tile
                for (float y = 0.0f; y < vertical_tiles_count; y ++)
                {
                    for (float x = 0.0f; x < horizontal_tiles_count; x ++)
                    {

                        // Get tile count
                        const int tile_count = y * horizontal_tiles_count + x;

                        const int x_offset = std::round(x * tile_width);
                        const int y_offset = std::round(y * tile_height);

                        cv::Rect r(cv::Point(x_offset, y_offset), new_tile_size);

                        // make sure the rectangle does not extend beyond the edges of the image
                        if (r.x + r.width >= frame.cols)
                        {
                            r.width = frame.cols - r.x - 1;
                        }

                        if (r.y + r.height >= frame.rows)
                        {
                            r.height = frame.rows - r.y - 1;
                        }

                        cv::Mat roi = frame(r);

                        // Just object detection
                        // Set frames
                        std::thread th(runtile,roi,  tile_count, tile_edge_factor, x_offset, y_offset);

                        t.push_back(std::move(th));  //<=== move (after, th doesn't hold it anymore

                        //std::thread tObj(runObj, roi);
                    }
                }

                for(auto& th : t){              //<=== range-based for uses & reference
                    th.join();
                }

                if (indexes_of_predictions_near_edges.empty() == false)
                {
                    // we need to go through all the results from the various tiles and merged together the ones that are side-by-side

                    for (const auto & lhs_idx : indexes_of_predictions_near_edges)
                    {
                        if (results[lhs_idx].rect.area() == 0 and results[lhs_idx].tile == -1)
                        {
                            // this items has already been consumed and is marked for deletion
                            continue;
                        }

                        const cv::Rect & lhs_rect = results[lhs_idx].rect;

                        // now compare this rect against all other rects that come *after* this
                        for (const auto & rhs_idx : indexes_of_predictions_near_edges)
                        {
                            if (rhs_idx <= lhs_idx)
                            {
                                // if the RHS object is on an earlier tile, then we've already compared it
                                continue;
                            }

                            if (results[lhs_idx].tile == results[rhs_idx].tile)
                            {
                                // if two objects are on the exact same tile, don't bother trying to combine them
                                continue;
                            }

                            if (results[rhs_idx].rect.area() == 0 and results[rhs_idx].tile == -1)
                            {
                                // this item has already been consumed and is marked for deletion
                                continue;
                            }


                            if (only_combine_similar_predictions)
                            {
                                // check the probabilities to see if there is any similarity:
                                //
                                // 1) does the LHS contain the best class from the RHS?
                                // 2) does the RHS contain the best class from the LHS?
                                //
                                if (results[lhs_idx].all_probabilities.count(results[rhs_idx].best_class) == 0 and
                                    results[rhs_idx].all_probabilities.count(results[lhs_idx].best_class) == 0)
                                {
                                    // the two objects have completely different classes, so we cannot combine them together
                                    continue;
                                }
                            }

                            const cv::Rect & rhs_rect		= results[rhs_idx].rect;
                            const cv::Rect combined_rect	= lhs_rect | rhs_rect;

                            // if this is a good match, then the area of the combined rect will be similar to the area of lhs+rhs
                            const int lhs_area		= lhs_rect		.area();
                            const int rhs_area		= rhs_rect		.area();
                            const int lhs_plus_rhs	= (lhs_area + rhs_area) * tile_rect_factor;
                            const int combined_area	= combined_rect	.area();

                            if (combined_area <= lhs_plus_rhs)
                            {
                                auto & lhs = results[lhs_idx];
                                auto & rhs = results[rhs_idx];

                                lhs.rect = combined_rect;

                                lhs.original_point.x = (static_cast<float>(lhs.rect.x) + static_cast<float>(lhs.rect.width	) / 2.0f) / static_cast<float>(frame.cols);
                                lhs.original_point.y = (static_cast<float>(lhs.rect.y) + static_cast<float>(lhs.rect.height	) / 2.0f) / static_cast<float>(frame.rows);

                                lhs.original_size.width		= static_cast<float>(lhs.rect.width	) / static_cast<float>(frame.cols);
                                lhs.original_size.height	= static_cast<float>(lhs.rect.height) / static_cast<float>(frame.rows);

                                // rebuild "all_probabilities" by combining both objects and keeping the max percentage
                                for (auto iter : rhs.all_probabilities)
                                {
                                    const auto & key		= iter.first;
                                    const auto & rhs_val	= iter.second;
                                    const auto & lhs_val	= lhs.all_probabilities[key];

                                    lhs.all_probabilities[key] = std::max(lhs_val, rhs_val);
                                }

                                // come up with a decent + consistent name to use for this object

                                //name_prediction(lhs);

                                // mark the RHS to be deleted once we're done looping through all the results
                                rhs.rect = {0, 0, 0, 0};
                                rhs.tile = -1;
                            }
                        }
                    }

                    // now go through the results and delete any with an empty rect and tile of -1
                    auto iter = results.begin();
                    while (iter != results.end())
                    {
                        if (iter->rect.area() == 0 and iter->tile == -1)
                        {
                            // delete this prediction from the results since it has been combined with something else
                            iter = results.erase(iter);
                        }
                        else
                        {
                            iter ++;
                        }
                    }
                }

                if (enable_debug)
                {
                    // draw vertical lines to show the tiles
                    for (float x=1.0; x < horizontal_tiles_count; x++)
                    {
                        const int x_pos = std::round(frame.cols / horizontal_tiles_count * x);
                        cv::line(frame, cv::Point(x_pos, 0), cv::Point(x_pos, frame.rows), {255, 0, 0});
                    }

                    // draw horizontal lines to show the tiles
                    for (float y=1.0; y < vertical_tiles_count; y++)
                    {
                        const int y_pos = std::round(frame.rows / vertical_tiles_count * y);
                        cv::line(frame, cv::Point(0, y_pos), cv::Point(frame.cols, y_pos), {255, 0, 0});
                    }
                }

                int annotation_line_thickness           = 2;
                float threshold							= 0.1f;
                float annotation_shade_predictions		= 0.25;
                auto colour                             = cv::Scalar(0x5E, 0x35, 0xFF);
                bool annotation_suppress_all_labels		= false;
                cv::HersheyFonts annotation_font_face	= cv::HersheyFonts::FONT_HERSHEY_SIMPLEX;
                float annotation_font_scale				= 0.5;
                int annotation_font_thickness			= 1;
                bool annotation_auto_hide_labels		= true;

                for (const auto & pred : results)
                {
    //                if (config.annotation_suppress_classes.count(pred.best_class) != 0)
    //                {
    //                    continue;
    //                }

                    if (annotation_line_thickness > 0 and pred.best_probability >= threshold)
                    {
                        //const auto colour = config.annotation_colours[pred.best_class % config.annotation_colours.size()];

                        int line_thickness_or_fill = annotation_line_thickness;
                        if (annotation_shade_predictions >= 1.0)
                        {
                            line_thickness_or_fill = cv::FILLED;
                        }
                        else if (annotation_shade_predictions > 0.0)
                        {
                            cv::Mat roi = frame(pred.rect);
                            cv::Mat coloured_rect(roi.size(), roi.type(), colour);

                            const double alpha = annotation_shade_predictions;
                            const double beta = 1.0 - alpha;
                            cv::addWeighted(coloured_rect, alpha, roi, beta, 0.0, roi);
                        }

            //			std::cout << "class id=" << pred.best_class << ", probability=" << pred.best_probability << ", point=(" << pred.rect.x << "," << pred.rect.y << "), name=\"" << pred.name << "\", duration=" << duration_string() << std::endl;
                        cv::rectangle(frame, pred.rect, {0x5E, 0x35, 0xFF}, line_thickness_or_fill);

                        if (annotation_suppress_all_labels)
                        {
                            continue;
                        }

                        int baseline = 0;
                        const cv::Size text_size = cv::getTextSize(pred.name, annotation_font_face, annotation_font_scale, annotation_font_thickness, &baseline);

                        if (annotation_auto_hide_labels)
                        {
                            if (text_size.width >= pred.rect.width or
                                text_size.height >= pred.rect.height)
                            {
                                // label is too large to display
                                continue;
                            }
                        }
    //                    int baseline = 0;
    //                    int text_size = 2;
                        cv::Rect r(cv::Point(pred.rect.x - annotation_line_thickness/2, pred.rect.y -2 - baseline + annotation_line_thickness), cv::Size(2 + annotation_line_thickness, 2 + baseline));
                        if (r.x < 0) r.x = 0;																			// shift the label to the very left edge of the screen, otherwise it would be off-screen
                        if (r.x + r.width >= frame.cols) r.x = pred.rect.x + pred.rect.width - r.width + 1;	// first attempt at pushing the label to the left
                        if (r.x + r.width >= frame.cols) r.x = frame.cols - r.width;				// more drastic attempt at pushing the label to the left

                        if (r.y < 0) r.y = pred.rect.y + pred.rect.height;	// shift the label to the bottom of the prediction, otherwise it would be off-screen
                        if (r.y + r.height >= frame.rows) r.y = pred.rect.y + 1; // shift the label to the inside-top of the prediction (CV seems to have trouble drawing text where the upper bound is y=0, so move it down 1 pixel)
                        if (r.y < 0) r.y = 0; // shift the label to the top of the image if it is off-screen


                        cv::rectangle(frame, r, {0x5E, 0x35, 0xFF}, cv::FILLED);
                        cv::putText(frame, pred.name, cv::Point(r.x + annotation_line_thickness/2, r.y +2), annotation_font_face, annotation_font_scale, cv::Scalar(0,0,0), annotation_font_thickness, cv::LINE_AA);
                    }
                }

            }
        }


        int down_width = frame.cols/2;
        int down_height = frame.rows/2;
        cv::resize(frame, frame, cv::Size(down_width, down_height), cv::INTER_LINEAR);
        cv::namedWindow( "Display frame",cv::WINDOW_AUTOSIZE);
        imshow("Display frame", frame);

        //Increase frame count
        frameNum++;

        char c=(char)cv::waitKey(1);
        if(c==27)
            break;

    }

    //    original_image			= frame;
    //    binary_inverted_image	= cv::Mat();
    //    prediction_results		= results;
    //    duration				= total_duration;
    //    horizontal_tiles		= horizontal_tiles_count;
    //    vertical_tiles			= vertical_tiles_count;
    //    tile_size				= new_tile_size;

    time(&end);

    // Time elapsed
    double seconds = difftime (end, start);
    std::cout << "Time taken : " << seconds << " seconds" << std::endl;

    // Calculate frames per second
    fps  = frameNum / seconds;
    std::cout << "Estimated frames per second : " << fps << std::endl;
    std::cout << "Frames : " << frameNum << std::endl;

    // When everything done, release
    //cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
