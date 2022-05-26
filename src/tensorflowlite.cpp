#include "tensorflowlite.h"


template <typename T>
absl::Span<const T> TensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<const T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

Tensorflowlite::Tensorflowlite(int tpu_num,const char* model_f, const char* model_l, float threshold)
{
    // Start model



    /* read obj det label */
    this->model_f = model_f;
    this->model_l = model_l;
    this->threshold = threshold;

    readLabel();
    model = tflite::FlatBufferModel::BuildFromFile(model_f);
    if (model == nullptr)
    {
        qDebug() << "Model " << model_f << " cannot be found or opened!";

    }
    else{
        qDebug() << "Model " << model_f << " found!";

    }

    const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    qDebug() << "available_tpus_path: " << QString::fromStdString(available_tpus[tpu_num].path);
    //qDebug() << "available_tpus_type: " << QString::fromStdString(available_tpus[tpu_num].type);
    qDebug() << "tpu_num: " <<QString::number( tpu_num);

    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(available_tpus[tpu_num].type, available_tpus[tpu_num].path);
    //edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    if (edgetpu_context == nullptr)
    {
        qDebug() << "TPU cannot be found or opened!";
        //resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

        tflite::InterpreterBuilder builder(*model.get(), resolver);
        if(builder(&interpreter) != kTfLiteOk) {
          qDebug() << "Failed to build interpreter.";
        }

        // Bind given context with interpreter.
        //interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
        interpreter->SetNumThreads(1);
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            qDebug() << "Failed to allocate tensors.";
        }
    }
    else{
        qDebug() << "TPU found or opened!";

        resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

        // Link model & resolver
        tflite::InterpreterBuilder builder(*model.get(), resolver);


        if(builder(&interpreter) != kTfLiteOk) {
          qDebug() << "Failed to build interpreter.";
        }

        // Bind given context with interpreter.
        interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
        //interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
        interpreter->SetNumThreads(1);
        if (interpreter->AllocateTensors() != kTfLiteOk) {
          qDebug() << "Failed to allocate tensors.";
        }
    }
}

Tensorflowlite::~Tensorflowlite(){

    // this custom op.
    interpreter.reset();
    //
    // Closes the edge TPU.
    edgetpu_context.reset();
    model.reset();
}
void Tensorflowlite::readLabel()
{
    std::ifstream ifs(model_l);
    if (ifs.fail()) {
        printf("failed to read %s\n", model_l);
        return;
    }
    std::string str;
    while(getline(ifs, str)) {
        labels.push_back(str);
    }

    printf("There are %i labels.\n",  labels.size());
}
void Tensorflowlite::setFrame(const cv::Mat& image){

    frame = image.clone();

}

void Tensorflowlite::runAnom(){

    // Pre-process
    const auto& tPreProcess0 = std::chrono::steady_clock::now();
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, frame, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    std::vector<uint8_t> inputData(frame.data, frame.data + (frame.cols * frame.rows * frame.elemSize()));
    const auto& tPreProcess1 = std::chrono::steady_clock::now();

    // Pre-process
    const auto& tInference0 = std::chrono::steady_clock::now();
    const auto& scores = coral::RunInference(inputData, interpreter.get());
    const auto& tInference1 = std::chrono::steady_clock::now();

    // Retrieve the result
    const auto& tPostProcess0 = std::chrono::steady_clock::now();
    int maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
//    float maxScore = *std::max_element(scores.begin(), scores.end());
//    outClass.classId = maxIndex;
//    outClass.score = maxScore;
    const auto& tPostProcess1 = std::chrono::steady_clock::now();
    //printf("%s (%.3f)\n", labels[maxIndex].c_str(), maxScore);

    outResults.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;

    outResults.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
    outResults.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;


}

void Tensorflowlite::runClass(){

    // Pre-process
    const auto& tPreProcess0 = std::chrono::steady_clock::now();

    cv::Mat input = frame.clone();

    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    cv::resize(input, input, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    std::vector<uint8_t> inputData(input.data, input.data + (input.cols * input.rows * input.elemSize()));
    const auto& tPreProcess1 = std::chrono::steady_clock::now();

    // Pre-process
    const auto& tInference0 = std::chrono::steady_clock::now();
    const auto& scores = coral::RunInference(inputData, interpreter.get());
    const auto& tInference1 = std::chrono::steady_clock::now();

    // Retrieve the result
    const auto& tPostProcess0 = std::chrono::steady_clock::now();
    int maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
    float maxScore = *std::max_element(scores.begin(), scores.end());
    outClass.classId = maxIndex;
    outClass.score = maxScore;
    const auto& tPostProcess1 = std::chrono::steady_clock::now();
    //printf("%s (%.3f)\n", labels[maxIndex].c_str(), maxScore);

    outResults.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;

    outResults.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
    outResults.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;


}

void Tensorflowlite::runDet()
{
    // Pre-process
    const auto& tPreProcess0 = std::chrono::steady_clock::now();
    int width = frame.cols;
    int height = frame.rows;
    cv::Mat input = frame.clone();

    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    cv::resize(input, input, cv::Size(MODEL_WIDTH_DET, MODEL_HEIGHT_DET));
    std::vector<uint8_t> inputData(input.data, input.data + (input.cols * input.rows * input.elemSize()));
    const auto& tPreProcess1 = std::chrono::steady_clock::now();

    // Run inference
    const auto& tInference0 = std::chrono::steady_clock::now();
    const auto& out = coral::RunInference(inputData, interpreter.get());
    const auto& tInference1 = std::chrono::steady_clock::now();

    // Retrieve the result
    const auto& tPostProcess0 = std::chrono::steady_clock::now();
    auto bboxes = TensorData<float>(*interpreter->output_tensor(0));
    auto ids = TensorData<float>(*interpreter->output_tensor(1));
    auto scores = TensorData<float>(*interpreter->output_tensor(2));
    auto count = TensorData<float>(*interpreter->output_tensor(3));

    int objectNum = 0;

    for (int i = 0; i < count[0]; ++i) {
      const int id = std::round(ids[i]);
      const float score = scores[i];
      outResults.objectList[objectNum].hasCrop = false;
      if (score >= threshold){
          //continue;
          const float ymin = std::max(0.0f, bboxes[4 * i])*height;
          const float xmin = std::max(0.0f, bboxes[4 * i + 1])*width;
          const float ymax = std::min(1.0f, bboxes[4 * i + 2])*height;
          const float xmax = std::min(1.0f, bboxes[4 * i + 3])*width;
          //q.push(Object{id, score, BBox<float>{ymin, xmin, ymax, xmax}});
          //if (q.size() > top_k) q.pop();
          qDebug() << "ID: ", ids[i];
          outResults.objectList[objectNum].classId = id;
          //snprintf(outputParam->objectList[objectNum].label, sizeof(outputParam->objectList[objectNum].label), "%s", object.label.c_str());
          outResults.objectList[objectNum].score = score;
          outResults.objectList[objectNum].x = xmin;
          outResults.objectList[objectNum].y = ymin;
          outResults.objectList[objectNum].width = xmax- xmin ;
          outResults.objectList[objectNum].height = ymax - ymin;

          if ((xmax - xmin) < (frame.cols-0)){

              cv::Rect region_of_interest = cropMat(frame.rows, frame.cols, xmin,ymin,xmax,ymax);

//              std::cout << "Box: " << region_of_interest.x << "  " <<
//                           region_of_interest.y << "  " <<
//                           region_of_interest.width << "  " <<
//                           region_of_interest.height << std::endl;

              if ((region_of_interest.x > 0) & (region_of_interest.y >0)){

                  try{
                    cv::Mat crop = frame(region_of_interest);
                    outResults.objectList[objectNum].crop = crop.clone();
                    outResults.objectList[objectNum].hasCrop = true;
                  }
                  catch (int e)
                  {
                    std::cout << "An exception occurred. Exception Nr. " << e << endl;
                  }
              }


          }
          objectNum++;
          if (objectNum >= NUM_MAX_RESULT) break;
      }
    }

    outResults.objectNum = objectNum;
    const auto& tPostProcess1 = std::chrono::steady_clock::now();

    outResults.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
    outResults.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
    outResults.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;

//    std::cout << "Time PreProcess: " << outResults.timePreProcess << std::endl;
//    std::cout << "Time Inference: " << outResults.timeInference << std::endl;
//    std::cout << "Time PostProcess: " << outResults.timePostProcess << std::endl;

}

cv::Rect Tensorflowlite::cropMat(int h, int w, int xmin, int ymin, int xmax, int ymax){

    // Get widths of box
    int bw = xmax - xmin;
    int bh = ymax - ymin;

    // Get center of box
    int bcx = xmin + bw/2;
    int bcy = ymin + bh/2;

    // Pad boundry condition
    int p = 0;
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
    h = bh;

    if (x0 <= 0) {
      x0 = 1;
      w = w-1;
    }
    if (y0 <= 0) {
      y0 = 1;
      h = h-1;

    }
    cv::Rect myROI(x0, y0, w, h);
    return myROI;
}

OUTPUT_CLASS Tensorflowlite::getOutClass() const
{
    return outClass;
}


OUTPUT_OBJ Tensorflowlite::getObjResults() const
{
    return outResults;
}

std::vector<std::string> Tensorflowlite::getLabels() const
{
    return labels;
}
