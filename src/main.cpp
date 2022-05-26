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

//typedef std::pair<int, int> point;
std::chrono::high_resolution_clock::duration duration;

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

void initTF(){

    /* Create object detection model */
    const char* model_f_obj = "/home/anton/Git/pycoral/test_data/ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite";
    const char* model_l_obj = "/home/anton/Git/pycoral/test_data/coco_labels.txt";
    model_obj = new Tensorflowlite(0,model_f_obj, model_l_obj,.8);


    std::cout << "Tensorflow initialization: OK" << std::endl;
}


void runObj(cv::Mat frame){
    model_obj->setFrame(frame);
    model_obj->runDet();
}


//void predict_tile(cv::Mat frame, const float new_threshold)
//{
//    std::vector<std::pair<int, int>> dims{{250,250},{400,400}};

//    float tile_rect_factor					= 1.20f;
//    float tile_edge_factor					= 0.25f;
//    bool only_combine_similar_predictions   = false;
//    bool combine_tile_predictions           = false;
//    const std::string filename              = "/home/anton/Git/pycoral/test_data/kite_and_cold.jpg";
//    bool enable_debug                       = false;

//    initTF();


//    //dims.push_back();


//    if (frame.empty())
//    {
//        /// @throw std::invalid_argument if the image is empty.
//        throw std::invalid_argument("cannot predict with an empty OpenCV image");
//    }

//    const float horizontal_factor		= static_cast<float>(frame.cols) / static_cast<float>(dims.at(0).first);
//    const float vertical_factor			= static_cast<float>(frame.rows) / static_cast<float>(dims.at(0).second);

//    const float horizontal_tiles_count	= std::round(std::max(1.0f, horizontal_factor	));
//    const float vertical_tiles_count	= std::round(std::max(1.0f, vertical_factor		));

//    // Set tile width
//    const float tile_width				= static_cast<float>(frame.cols) / horizontal_tiles_count;
//    const float tile_height				= static_cast<float>(frame.rows) / vertical_tiles_count;

//    // Create tile size
//    const cv::Size new_tile_size		= cv::Size(std::round(tile_width), std::round(tile_height));

//    // Init multi frame capture vars for tracking
//    double fps;
//    int frameNum = 0;

//    // Start and end times
//    time_t start, end;

//    // Start time
//    time(&start);

//    // Start the timer
//    std::chrono::high_resolution_clock::duration total_duration = std::chrono::milliseconds(0);

//    // Set labels
//    std::vector<std::string> labels_obj = model_obj->getLabels();

//    if (horizontal_tiles_count == 1 and vertical_tiles_count == 1)
//    {
//        // image is smaller than (or equal to) the network, so use the original predict() call
//        //return predict_internal(frame, new_threshold);

//        // Set frames
//        std::thread tObj(runObj, frame);

//        // Inc PLC
//        tObj.join();

//        //TODO POINTER
//        OUTPUT_OBJ result = model_obj->getObjResults();

//    }

//    // otherwise, if we get here then we have more than 1 tile

//    // divide the original image into the right number of tiles and call predict() on each tile
//    std::vector<PredictionResult> results;
//    std::vector<size_t> indexes_of_predictions_near_edges;
//    std::vector<cv::Mat> all_tile_mats;



//    for (float y = 0.0f; y < vertical_tiles_count; y ++)
//    {
//        for (float x = 0.0f; x < horizontal_tiles_count; x ++)
//        {

//            // Get tile count
//            const int tile_count = y * horizontal_tiles_count + x;

//            const int x_offset = std::round(x * tile_width);
//            const int y_offset = std::round(y * tile_height);

//            cv::Rect r(cv::Point(x_offset, y_offset), new_tile_size);

//            // make sure the rectangle does not extend beyond the edges of the image
//            if (r.x + r.width >= frame.cols)
//            {
//                r.width = frame.cols - r.x - 1;
//            }

//            if (r.y + r.height >= frame.rows)
//            {
//                r.height = frame.rows - r.y - 1;
//            }

//            cv::Mat roi = frame(r);

//            // Just object detection
//            // Set frames
//            std::thread tObj(runObj, frame);

//            // Inc PLC
//            tObj.join();

//            //TODO POINTER
//            OUTPUT_OBJ result = model_obj->getObjResults();


//            total_duration += duration;

//            // fix up the predictions -- need to compensate for the tile not being the top-left corner of the image, and the size of the tile being smaller than the image
////            for (auto & prediction : prediction_results)
//            for (const auto& object : result.objectList)
//            {
//                PredictionResult prediction;


//                float score = static_cast<float>(object.score);
//                int x0 = static_cast<int32_t>(object.x);
//                int y0 = static_cast<int32_t>(object.y);

//                int width = static_cast<int32_t>(object.width);
//                int height = static_cast<int32_t>(object.height);

//                prediction.original_point.x = static_cast<int32_t>((x0+width)/frame.cols);
//                prediction.original_point.y = static_cast<int32_t>((y0+height)/frame.rows);

//                prediction.original_size.width =static_cast<int32_t>(object.width/frame.cols);;
//                prediction.original_size.height = static_cast<int32_t>(object.height/frame.rows);;

//                prediction.best_class = object.classId;
//                prediction.best_probability = score;
//                prediction.name = labels_obj[object.classId].c_str();
//                prediction.tile = tile_count;

//                prediction.rect.x = x0;
//                prediction.rect.y =  y0;

//                prediction.rect.width = width;
//                prediction.rect.height = height;


//                //track which predictions are near the edges, because we may need to re-examine them and join them after we finish with all the tiles

//                if (combine_tile_predictions)
//                {
//                    const int minimum_horizontal_distance	= tile_edge_factor * prediction.rect.width;
//                    const int minimum_vertical_distance		= tile_edge_factor * prediction.rect.height;
//                    if (prediction.rect.x <= minimum_horizontal_distance					or
//                        prediction.rect.y <= minimum_vertical_distance						or
//                        roi.cols - prediction.rect.br().x <= minimum_horizontal_distance	or
//                        roi.rows - prediction.rect.br().y <= minimum_vertical_distance		)
//                    {
//                        // this prediction is near one of the tile borders so we need to remember it
//                        indexes_of_predictions_near_edges.push_back(results.size());
//                    }
//                }

//                // every prediction needs to have x_offset and y_offset added to it
//                prediction.rect.x += x_offset;
//                prediction.rect.y += y_offset;
//                prediction.tile = tile_count;

//                if (enable_debug)
//                {
//                    // draw a black-on-white debug label on the top side of the annotation

//                    const std::string label		= "TEST"; //std::to_string(results.size());
//                    const auto font				= cv::HersheyFonts::FONT_HERSHEY_PLAIN;
//                    const auto scale			= 0.75;
//                    const auto thickness		= 1;
//                    int baseline				= 0;
//                    const cv::Size text_size	= cv::getTextSize(label, font, scale, thickness, &baseline);
//                    const int text_half_width	= text_size.width			/ 2;
//                    const int text_half_height	= text_size.height			/ 2;
//                    const int pred_half_width	= prediction.rect.width		/ 2;
//                    const int pred_half_height	= prediction.rect.height	/ 2;

//                    // put the text exactly in the middle of the prediction
//                    const cv::Rect label_rect(
//                            prediction.rect.x + pred_half_width - text_half_width,
//                            prediction.rect.y + pred_half_height - text_half_height,
//                            text_size.width, text_size.height);
//                    cv::rectangle(frame, label_rect, {255, 255, 255}, cv::FILLED, cv::LINE_AA);
//                    cv::putText(frame, label, cv::Point(label_rect.x, label_rect.y + label_rect.height), font, scale, cv::Scalar(0,0,0), thickness, cv::LINE_AA);
//                }

//                // the original point and size are based on only 1 tile, so they also need to be fixed

//                prediction.original_point.x = (static_cast<float>(prediction.rect.x) + static_cast<float>(prediction.rect.width	) / 2.0f) / static_cast<float>(frame.cols);
//                prediction.original_point.y = (static_cast<float>(prediction.rect.y) + static_cast<float>(prediction.rect.height) / 2.0f) / static_cast<float>(frame.rows);

//                prediction.original_size.width	= static_cast<float>(prediction.rect.width	) / static_cast<float>(frame.cols);
//                prediction.original_size.height	= static_cast<float>(prediction.rect.height	) / static_cast<float>(frame.rows);

//                results.push_back(prediction);
//            }
//        }
//    }

//    if (indexes_of_predictions_near_edges.empty() == false)
//    {
//        // we need to go through all the results from the various tiles and merged together the ones that are side-by-side

//        for (const auto & lhs_idx : indexes_of_predictions_near_edges)
//        {
//            if (results[lhs_idx].rect.area() == 0 and results[lhs_idx].tile == -1)
//            {
//                // this items has already been consumed and is marked for deletion
//                continue;
//            }

//            const cv::Rect & lhs_rect = results[lhs_idx].rect;

//            // now compare this rect against all other rects that come *after* this
//            for (const auto & rhs_idx : indexes_of_predictions_near_edges)
//            {
//                if (rhs_idx <= lhs_idx)
//                {
//                    // if the RHS object is on an earlier tile, then we've already compared it
//                    continue;
//                }

//                if (results[lhs_idx].tile == results[rhs_idx].tile)
//                {
//                    // if two objects are on the exact same tile, don't bother trying to combine them
//                    continue;
//                }

//                if (results[rhs_idx].rect.area() == 0 and results[rhs_idx].tile == -1)
//                {
//                    // this item has already been consumed and is marked for deletion
//                    continue;
//                }


////                if (config.only_combine_similar_predictions)
////                {
////                    // check the probabilities to see if there is any similarity:
////                    //
////                    // 1) does the LHS contain the best class from the RHS?
////                    // 2) does the RHS contain the best class from the LHS?
////                    //
////                    if (results[lhs_idx].all_probabilities.count(results[rhs_idx].best_class) == 0 and
////                        results[rhs_idx].all_probabilities.count(results[lhs_idx].best_class) == 0)
////                    {
////                        // the two objects have completely different classes, so we cannot combine them together
////                        continue;
////                    }
////                }

//                const cv::Rect & rhs_rect		= results[rhs_idx].rect;
//                const cv::Rect combined_rect	= lhs_rect | rhs_rect;

//                // if this is a good match, then the area of the combined rect will be similar to the area of lhs+rhs
//                const int lhs_area		= lhs_rect		.area();
//                const int rhs_area		= rhs_rect		.area();
//                const int lhs_plus_rhs	= (lhs_area + rhs_area) * tile_rect_factor;
//                const int combined_area	= combined_rect	.area();

//                if (combined_area <= lhs_plus_rhs)
//                {
//                    auto & lhs = results[lhs_idx];
//                    auto & rhs = results[rhs_idx];

//                    lhs.rect = combined_rect;

//                    lhs.original_point.x = (static_cast<float>(lhs.rect.x) + static_cast<float>(lhs.rect.width	) / 2.0f) / static_cast<float>(frame.cols);
//                    lhs.original_point.y = (static_cast<float>(lhs.rect.y) + static_cast<float>(lhs.rect.height	) / 2.0f) / static_cast<float>(frame.rows);

//                    lhs.original_size.width		= static_cast<float>(lhs.rect.width	) / static_cast<float>(frame.cols);
//                    lhs.original_size.height	= static_cast<float>(lhs.rect.height) / static_cast<float>(frame.rows);

//                    // rebuild "all_probabilities" by combining both objects and keeping the max percentage
//                    for (auto iter : rhs.all_probabilities)
//                    {
//                        const auto & key		= iter.first;
//                        const auto & rhs_val	= iter.second;
//                        const auto & lhs_val	= lhs.all_probabilities[key];

//                        lhs.all_probabilities[key] = std::max(lhs_val, rhs_val);
//                    }

//                    // come up with a decent + consistent name to use for this object

//                    //name_prediction(lhs);

//                    // mark the RHS to be deleted once we're done looping through all the results
//                    rhs.rect = {0, 0, 0, 0};
//                    rhs.tile = -1;
//                }
//            }
//        }

//        // now go through the results and delete any with an empty rect and tile of -1
//        auto iter = results.begin();
//        while (iter != results.end())
//        {
//            if (iter->rect.area() == 0 and iter->tile == -1)
//            {
//                // delete this prediction from the results since it has been combined with something else
//                iter = results.erase(iter);
//            }
//            else
//            {
//                iter ++;
//            }
//        }
//    }

//    if (enable_debug)
//    {
//        // draw vertical lines to show the tiles
//        for (float x=1.0; x < horizontal_tiles_count; x++)
//        {
//            const int x_pos = std::round(frame.cols / horizontal_tiles_count * x);
//            cv::line(frame, cv::Point(x_pos, 0), cv::Point(x_pos, frame.rows), {255, 0, 0});
//        }

//        // draw horizontal lines to show the tiles
//        for (float y=1.0; y < vertical_tiles_count; y++)
//        {
//            const int y_pos = std::round(frame.rows / vertical_tiles_count * y);
//            cv::line(frame, cv::Point(0, y_pos), cv::Point(frame.cols, y_pos), {255, 0, 0});
//        }
//    }

////    original_image			= frame;
////    binary_inverted_image	= cv::Mat();
////    prediction_results		= results;
////    duration				= total_duration;
////    horizontal_tiles		= horizontal_tiles_count;
////    vertical_tiles			= vertical_tiles_count;
////    tile_size				= new_tile_size;

////    return results;

//}

int main(int argc, char *argv[])
{


//    initTF();

//    std::vector<std::pair<int, int>> dims{{250,250},{400,400}};

//    //dims.push_back();

//    const std::string filename = "/home/anton/Git/pycoral/test_data/kite_and_cold.jpg";
//    //cv::VideoCapture cap(filename);



//    // Check if camera opened successfully

//    /*
//    if(!cap.isOpened()){
//        std::cout << "Error opening video stream or file" <<  std::endl;
//        return -1;
//    }
//    */

//    double fps;
//    int frameNum = 0;
//    // Start and end times
//    time_t start, end;

//    // Start time
//    time(&start);

//    // Set labels
//    std::vector<std::string> labels_obj = model_obj->getLabels();

//    while(1){

//        cv::Mat frame;

//        frame = cv::imread(filename);

//        // Capture frame-by-frame
//        //cap >> frame;

//        // If the frame is empty, break immediately
//        if (frame.empty()){
//            break;
//        }

//        for (auto dim: dims) {


//        }
//        // Set frames
//        std::thread tObj(runObj, frame);

//        // Inc PLC
//        tObj.join();

//        //TODO POINTER
//        OUTPUT_OBJ result = model_obj->getObjResults();




//        for (const auto& object : result.objectList) {
//            cv::rectangle(frame, cv::Rect(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y), static_cast<int32_t>(object.width), static_cast<int32_t>(object.height)), cv::Scalar(255, 255, 0), 3);
//            cv::putText(frame, labels_obj[object.classId].c_str(), cv::Point(static_cast<int32_t>(object.x), static_cast<int32_t>(object.y) + 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 0, 0), 3);

//            if (object.hasCrop == true){

//                int xmin = static_cast<int32_t>(object.x);
//                int ymin = static_cast<int32_t>(object.y);
//                int xmax = xmin + static_cast<int32_t>(object.width);
//                int ymax = ymin + static_cast<int32_t>(object.height);

//                std::string detectClass = labels_obj[object.classId].c_str();
//            }
//        }

//        /*
//        */

//        // to half size or even smaller
//        // resize(frame, frame, cv::Size(frame.cols/4, frame.rows/4));

//        cv::namedWindow( "Display frame",cv::WINDOW_AUTOSIZE);
//        imshow("Display frame", frame);


//        //Increase frame count
//        frameNum++;

//        char c=(char)cv::waitKey(10);
//        if(c==27)
//            break;
//    }



    std::vector<std::pair<int, int>> dims{{250,250},{400,400}};

    float tile_rect_factor					= 1.20f;
    float tile_edge_factor					= 0.25f;
    bool only_combine_similar_predictions   = true;
    bool combine_tile_predictions           = true;
    const std::string filename              = "/home/anton/Git/pycoral/test_data/kite_and_cold.jpg";
    bool enable_debug                       = true;

    cv::Mat frame;

    frame = cv::imread(filename);
    initTF();


    //dims.push_back();


    if (frame.empty())
    {
        /// @throw std::invalid_argument if the image is empty.
        throw std::invalid_argument("cannot predict with an empty OpenCV image");
    }

    const float horizontal_factor		= static_cast<float>(frame.cols) / static_cast<float>(dims.at(0).first);
    const float vertical_factor			= static_cast<float>(frame.rows) / static_cast<float>(dims.at(0).second);

    const float horizontal_tiles_count	= std::round(std::max(1.0f, horizontal_factor	));
    const float vertical_tiles_count	= std::round(std::max(1.0f, vertical_factor		));

    // Set tile width
    const float tile_width				= static_cast<float>(frame.cols) / horizontal_tiles_count;
    const float tile_height				= static_cast<float>(frame.rows) / vertical_tiles_count;

    // Create tile size
    const cv::Size new_tile_size		= cv::Size(std::round(tile_width), std::round(tile_height));

    // Init multi frame capture vars for tracking
    double fps;
    int frameNum = 0;

    // Start and end times
    time_t start, end;

    // Start time
    time(&start);

    // Start the timer
    std::chrono::high_resolution_clock::duration total_duration = std::chrono::milliseconds(0);

    // Set labels
    std::vector<std::string> labels_obj = model_obj->getLabels();



//    original_image			= frame;
//    binary_inverted_image	= cv::Mat();
//    prediction_results		= results;
//    duration				= total_duration;
//    horizontal_tiles		= horizontal_tiles_count;
//    vertical_tiles			= vertical_tiles_count;
//    tile_size				= new_tile_size;

//    return results;

    while(1){

        if (horizontal_tiles_count == 1 and vertical_tiles_count == 1)
        {
            // image is smaller than (or equal to) the network, so use the original predict() call
            //return predict_internal(frame, new_threshold);

            // Set frames
            std::thread tObj(runObj, frame);

            // Inc PLC
            tObj.join();

            //TODO POINTER
            OUTPUT_OBJ result = model_obj->getObjResults();

        }

        // otherwise, if we get here then we have more than 1 tile

        // divide the original image into the right number of tiles and call predict() on each tile
        std::vector<PredictionResult> results;
        std::vector<size_t> indexes_of_predictions_near_edges;
        std::vector<cv::Mat> all_tile_mats;



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
                std::thread tObj(runObj, roi);

                // Inc PLC
                tObj.join();

                //TODO POINTER
                OUTPUT_OBJ result = model_obj->getObjResults();


                total_duration += duration;
                // fix up the predictions -- need to compensate for the tile not being the top-left corner of the image, and the size of the tile being smaller than the image
    //            for (auto & prediction : prediction_results)
                for (const auto& object : result.objectList)
                {
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
                    //qDebug() << "HERE1";
                    prediction.best_probability = score;

                    //qDebug() << object.classId;

                    prediction.name = "TEST"; //labels_obj[prediction.best_class];
                    prediction.tile = tile_count;

                    prediction.rect.x = x0;
                    prediction.rect.y =  y0;

                    prediction.rect.width = width;
                    prediction.rect.height = height;

                    //track which predictions are near the edges, because we may need to re-examine them and join them after we finish with all the tiles

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
                    prediction.tile = tile_count;

                    if (enable_debug)
                    {
                        // draw a black-on-white debug label on the top side of the annotation

                        const std::string label		= "T777EST"; //std::to_string(results.size());
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
                }



            }
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
        cv::namedWindow( "Display frame",cv::WINDOW_AUTOSIZE);
        imshow("Display frame", frame);


        //Increase frame count
        frameNum++;

        char c=(char)cv::waitKey(10);
        if(c==27)
            break;
    }

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


