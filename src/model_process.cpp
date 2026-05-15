/**
* @file model_process.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "model_process.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <random>
#include "utils.h"
#include "opencv2/imgproc.hpp"

using namespace std;

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

ModelProcess::ModelProcess()
{
}

ModelProcess::~ModelProcess()
{
    Unload();
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
}

Result ModelProcess::LoadModelFromFileWithMem(const std::string& modelPath)
{
    uint32_t fileSize = 0;
    modelMemPtr_ = Utils::ReadBinFile(modelPath, fileSize);
    modelMemSize_ = fileSize;
    svp_acl_error ret = svp_acl_mdl_load_from_mem(static_cast<uint8_t* >(modelMemPtr_), modelMemSize_, &modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        svp_acl_rt_free(modelMemPtr_);
        ERROR_LOG("load model from file failed, model file is %s", modelPath.c_str());
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath.c_str());
    return SUCCESS;
}

Result ModelProcess::CreateDesc()
{
    modelDesc_ = svp_acl_mdl_create_desc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    svp_acl_error ret = svp_acl_mdl_get_desc(modelDesc_, modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)svp_acl_mdl_destroy_desc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::InitInput()
{
    input_ = svp_acl_mdl_create_dataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize, int stride)
{
    svp_acl_data_buffer* inputData = svp_acl_create_data_buffer(inputDataBuffer, bufferSize, stride);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    svp_acl_error ret = svp_acl_mdl_add_dataset_buffer(input_, inputData);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed");
        svp_acl_destroy_data_buffer(inputData);
        inputData = nullptr;
        return FAILED;
    }

    return SUCCESS;
}

size_t ModelProcess::GetInputDataSize(int index) const
{
    svp_acl_data_type dataType = svp_acl_mdl_get_input_data_type(modelDesc_, index);
    return svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
}

size_t ModelProcess::GetOutputDataSize(int index) const
{
    svp_acl_data_type dataType = svp_acl_mdl_get_output_data_type(modelDesc_, index);
    return svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
}

Result ModelProcess::GetInputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const
{
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, index, &dims);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
        return FAILED;
    }
    stride = svp_acl_mdl_get_input_default_stride(modelDesc_, index);
    if (stride == 0) {
        ERROR_LOG("svp_acl_mdl_get_input_default_stride error!");
        return FAILED;
    }
    bufSize = svp_acl_mdl_get_input_size_by_index(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("svp_acl_mdl_get_input_size_by_index error!");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const
{
    svp_acl_error ret = svp_acl_mdl_get_output_dims(modelDesc_, index, &dims);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_mdl_get_output_dims error!");
        return FAILED;
    }
    stride = svp_acl_mdl_get_output_default_stride(modelDesc_, index);
    if (stride == 0) {
        ERROR_LOG("svp_acl_mdl_get_output_default_stride error!");
        return FAILED;
    }
    bufSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("svp_acl_mdl_get_output_size_by_index error!");
        return FAILED;
    }
    return SUCCESS;
}

void ModelProcess::DestroyInput()
{
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(input_); ++i) {
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(input_, i);
        void* tmp = svp_acl_get_data_buffer_addr(dataBuffer);
        svp_acl_rt_free(tmp);
        svp_acl_destroy_data_buffer(dataBuffer);
    }
    svp_acl_mdl_destroy_dataset(input_);
    input_ = nullptr;
}

Result ModelProcess::CreateOutput()
{
    output_ = svp_acl_mdl_create_dataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t outputSize = svp_acl_mdl_get_num_outputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t stride = svp_acl_mdl_get_output_default_stride(modelDesc_, i);
        if (stride == 0) {
            ERROR_LOG("Error, output default stride is %lu.", stride);
            return FAILED;
        }
        size_t bufferSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, i);
        if (bufferSize == 0) {
            ERROR_LOG("Error, output size is %lu.", bufferSize);
            return FAILED;
        }

        void *outputBuffer = nullptr;
        svp_acl_error ret = svp_acl_rt_malloc(&outputBuffer, bufferSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", bufferSize);
            return FAILED;
        }
        Utils::InitData(static_cast<int8_t*>(outputBuffer), bufferSize);

        svp_acl_data_buffer* outputData = svp_acl_create_data_buffer(outputBuffer, bufferSize, stride);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            svp_acl_rt_free(outputBuffer);
            return FAILED;
        }
        ret = svp_acl_mdl_add_dataset_buffer(output_, outputData);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed");
            svp_acl_rt_free(outputBuffer);
            svp_acl_destroy_data_buffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

Result ModelProcess::ClearOutputStrideInvalidBuf(std::vector<int8_t>& buffer, size_t index) const
{
    size_t bufSize = 0;
    size_t bufStride = 0;
    svp_acl_mdl_io_dims dims;
    svp_acl_error ret = GetOutputStrideParam(index, bufSize, bufStride, dims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetOutputStrideParam failed");
        return FAILED;
    }
    if ((bufSize == 0) || (bufStride == 0)) {
        ERROR_LOG("Error, bufSize(%zu) bufStride(%zu) invalid", bufSize, bufStride);
        return FAILED;
    }
    if ((dims.dim_count == 0) || (dims.dims[dims.dim_count - 1] <= 0)) {
        ERROR_LOG("Error, dims para invalid");
        return FAILED;
    }
    int64_t lastDim = dims.dims[dims.dim_count - 1];

    size_t dataSize = GetOutputDataSize(index);
    if (dataSize == 0) {
        ERROR_LOG("Error, dataSize == 0 invalid");
        return FAILED;
    }
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = bufSize / bufStride;
    size_t invalidSize = bufStride - lastDimSize;
    if (invalidSize == 0) {
        // not stride invalid space, return directly.
        return SUCCESS;
    }

    for (size_t i = 0; i < loopNum; ++i) {
        size_t offset = bufStride * i + lastDimSize;
        int8_t* ptr = &buffer[offset];
        for (size_t index = 0; index < invalidSize; index++) {
            ptr[index] = 0;
        }
    }
    return SUCCESS;
}

void ModelProcess::WriteOutput(const string& outputFileName, size_t index) const
{
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, index);
    if (dataBuffer == nullptr) {
        ERROR_LOG("output[%zu] dataBuffer nullptr invalid", index);
        return;
    }
    int8_t* outData = (int8_t*)svp_acl_get_data_buffer_addr(dataBuffer);
    size_t outSize = svp_acl_get_data_buffer_size(dataBuffer);
    if (outData == nullptr || outSize == 0) {
        ERROR_LOG("output[%zu] data or size(%zu) invalid", index, outSize);
        return;
    }

    // malloc temp buffer to clear stride useless temp buffer to help output.bin compare
    std::vector<int8_t> tempBuf(outData, outData + outSize);
    if (tempBuf.empty()) {
        ERROR_LOG("tempBuf malloc fail");
        return;
    }

    Result ret = ClearOutputStrideInvalidBuf(tempBuf, index);
    if (ret != SUCCESS) {
        ERROR_LOG("ClearStrideInvalidBuf fail");
        return;
    }

    ofstream fout(outputFileName, ios::out|ios::binary);
    if (fout.good() == false) {
        ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
        return;
    }
    fout.write((char*)&tempBuf[0], tempBuf.size() * sizeof(int8_t));
    fout.close();
    return;
}

void ModelProcess::DumpModelOutputResult() const
{
    stringstream ss;
    size_t outputNum = svp_acl_mdl_get_dataset_num_buffers(output_);
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << executeNum_ << "_" << i << ".bin";
        string outputFileName = ss.str();
        WriteOutput(outputFileName, i);
        ss.str("");
    }
    INFO_LOG("dump data success");
}

enum BoxValue {
    TOP_LEFT_X = 0,
    TOP_LEFT_Y = 1,
    BOTTOM_RIGHT_X = 2,
    BOTTOM_RIGHT_Y = 3,
    SCORE = 4,
    CLASS_ID = 5,
    BBOX_SIZE = 6
};

bool Cmp(const std::vector<float>& veci, const vector<float>& vecj)
{
    if (veci[CLASS_ID] < vecj[CLASS_ID]) {
        return true;
    } else if (veci[CLASS_ID] == vecj[CLASS_ID]) {
        return veci[SCORE] > vecj[SCORE];
    }
    return false;
}

static void PrintResult(const std::vector<std::vector<float>>& boxValue)
{
    if (boxValue.empty()) {
        WARN_LOG("input box empty");
        return;
    }
    std::vector<int> clsNum;
    float cId = boxValue[0][CLASS_ID];
    int validNum = 0;
    for (size_t loop = 0; loop < boxValue.size(); loop++) {
        if (boxValue[loop][CLASS_ID] == cId) {
            validNum++;
        } else {
            clsNum.push_back(validNum);
            cId = boxValue[loop][CLASS_ID];
            validNum = 1;
        }
    }
    clsNum.push_back(validNum);
    int idx = 0;
    int sumNum = 0;
    INFO_LOG("current class valid box number is: %d", clsNum[idx]);
    sumNum += clsNum[idx];
    size_t totalBoxNum = boxValue.size();
    for (size_t loop = 0; loop < totalBoxNum; loop++) {
        if (loop == static_cast<size_t>(sumNum)) {
            idx++;
            INFO_LOG("current class valid box number is: %d", clsNum[idx]);
            sumNum += clsNum[idx];
        }
        INFO_LOG("lx: %lf, ly: %lf, rx: %lf, ry: %lf, score: %lf; class id: %d",
            boxValue[loop][TOP_LEFT_X], boxValue[loop][TOP_LEFT_Y], boxValue[loop][BOTTOM_RIGHT_X],
            boxValue[loop][BOTTOM_RIGHT_Y], boxValue[loop][SCORE], (int)boxValue[loop][CLASS_ID]);
    }
}

static std::string GetDetResultStr(const std::vector<float>& detResult)
{
    enum DET_RESULT_WIDTH {
        INDEX_INTEGER_WIDTH = 4,
        SCORE_INTERGER_WIDTH = 9,
        SCORE_DECIMAL_WIDTH = 8,
        COORD_INTERGER_WIDTH = 4,
        COORD_DECIMAL_WIDTH = 2,
    };
    stringstream ss;
    ss << int(detResult[CLASS_ID]) << "  ";

    ss << std::fixed;

    auto floatStream = [](float val, int width, int precision, stringstream& ss) {
        ss << std::setw(width) << std::setprecision(precision) << val << "  ";
    };

    floatStream(detResult[SCORE], SCORE_INTERGER_WIDTH + SCORE_DECIMAL_WIDTH, SCORE_DECIMAL_WIDTH, ss);
    floatStream(detResult[TOP_LEFT_X], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detResult[TOP_LEFT_Y], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detResult[BOTTOM_RIGHT_X], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detResult[BOTTOM_RIGHT_Y], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);

    ss << std::endl;
    return ss.str();
}

static void WriteResult(const std::vector<std::vector<float>>& boxValue,
    uint32_t oriImgWidth, uint32_t oriImgHeight, int32_t modelId)
{
    size_t boxSize = boxValue.size();
    if (boxSize == 0) {
        WARN_LOG("input box empty");
        return;
    }
    string tmp = "";
    tmp = (modelId == static_cast<int>(Yolo::YOLOV8_FACE)) ? "yolov8_face" : tmp;
    const string fileName = tmp + "_detResult.txt";
    ofstream fout(fileName.c_str());

    std::stringstream ss;
    ss << oriImgWidth << "  " << oriImgHeight << std::endl;
    fout << ss.str();
    ss.str("");
    if (fout.good() == false) {
        ERROR_LOG("fout open fail");
        return;
    }
    for (size_t loop = 0; loop < boxSize; loop++) {
        auto value = boxValue[loop];
        if (value[SCORE] > 1.0f || value[SCORE] < 0.0f) {
            WARN_LOG("invalid score %f", value[SCORE]);
            continue;
        }
        ss << GetDetResultStr(value);
        fout << ss.str();
        ss.str("");
    }
    fout.close();
    return;
}

// yolo/ssd output 0 is num, output 1 is bbox
enum InputOutputId {
    INPUT_IMG_ID = 0,
    OUTPUT_NUM_ID = 0,
    OUTPUT_BBOX_ID = 1
};

const std::vector<std::string> CLASSES = { "face" };

static void SaveResult(const std::vector<std::vector<float>>& boxValue, std::vector<Detection>& detections)
{
    for (auto bbox : boxValue) {
        Detection result;
        result.classId = bbox[CLASS_ID];
        result.confidence = bbox[SCORE];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255); // 100: minx value; 255: max value
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        result.className = CLASSES[result.classId];
        cv::Rect boxs(bbox[TOP_LEFT_X], bbox[TOP_LEFT_Y],
            static_cast<int>(bbox[BOTTOM_RIGHT_X] - bbox[TOP_LEFT_X]),
            static_cast<int>(bbox[BOTTOM_RIGHT_Y] - bbox[TOP_LEFT_Y]));
        result.box = boxs;

        detections.push_back(result);
    }
}

static void DrawResult(const string& imgName, const std::vector<Detection>& detections,
    int32_t imgWidth, int32_t imgHeight, int32_t modelId)
{
    cv::Mat frame = cv::imread(imgName);
    cv::resize(frame, frame, cv::Size(imgWidth, imgHeight));
    for (auto detection : detections) {
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2); // 2: thickness

        // Detection box text,yolov1 doesnt draw className & score
        if (modelId != 1) {
            // 4: len
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0); // 2: thickness
            auto textBoxX = (box.x >= 0) ? box.x : 0;
            auto textBoxY = ((box.y - 40) >= 0) ? (box.y - 40) : box.y;  // 40: offset;
            // 10: offset; 20: offset
            cv::Rect textBox(textBoxX, textBoxY, textSize.width + 10, textSize.height + 20);
            cv::rectangle(frame, textBox, color, cv::FILLED);
            // 40: offset; 10: offset; 30: offset;
            auto textPointY = ((box.y - 40) >= 0) ? (box.y - 10) : (textBoxY + 30);
            cv::putText(frame, classString, cv::Point(textBoxX + 5, textPointY), // 5: offset;
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0); // 2: thickness
        }
    }
    string outImgPath = "";
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV8_FACE)) ? "out_img_yolov8_face.jpg" : outImgPath;
    cv::imwrite(outImgPath, frame);
}

void ModelProcess::OutputModelResult(int32_t modelId, const string& imgName) const
{
    if (modelId == static_cast<int>(Yolo::YOLOV8_FACE)) {
        OutputModelResultYoloV8Face(modelId, imgName);
    } else {
        ERROR_LOG("Only YOLOV8_FACE is supported");
    }
}

static float CalcIou(const vector<float> &box1, const vector<float> &box2)
{
    float area1 = box1[6];
    float area2 = box2[6];
    float xx1 = max(box1[0], box2[0]);
    float yy1 = max(box1[1], box2[1]);
    float xx2 = min(box1[2], box2[2]);
    float yy2 = min(box1[3], box2[3]);
    float w = max(0.0f, xx2 - xx1 + 1);
    float h = max(0.0f, yy2 - yy1 + 1);
    float inter = w * h;
    float ovr = inter / (area1 + area2 - inter);
    return ovr;
}

static void MulticlassNms(vector<vector<float>>& bboxes, const vector<vector<float>>& vaildBox, float nmsThr)
{
    const uint8_t scoreIdx = 0;
    const uint8_t xcenterIdx = 1;
    const uint8_t ycenterIdx  = 2; // 2 index
    const uint8_t wIdx = 3; // 3: index
    const uint8_t hIdx = 4; // 4: index
    const uint8_t classIdIdx = 5; // 5: index
    for (auto &item : vaildBox) { /* score, xcenter, ycenter, w, h, classId */
        float boxXCenter = item[xcenterIdx];
        float boxYCenter = item[ycenterIdx];
        float boxWidth = item[wIdx];;
        float boxHeight = item[hIdx];;

        float x1 = (boxXCenter - boxWidth / 2);
        float y1 = (boxYCenter - boxHeight / 2);
        float x2 = (boxXCenter + boxWidth / 2);
        float y2 = (boxYCenter + boxHeight / 2);
        float area = (x2 - x1 + 1) * (y2 - y1 + 1);
        bool keep = true;
        /* lx, ly, rx, ry, score, class id, area */
        vector<float> bbox {x1, y1, x2, y2, item[scoreIdx], item[classIdIdx], area};
        for (size_t j = 0; j < bboxes.size(); j++) {
            if (CalcIou(bbox, bboxes[j]) > nmsThr) {
                keep = false;
                break;
            }
        }
        if (keep) {
            bboxes.push_back(bbox);
        }
    }
}


void ModelProcess::FilterYolov8FaceBox(int32_t modelId, vector<vector<float>>& vaildBox) const
{
    svp_acl_mdl_io_dims outDims;
    svp_acl_error ret = svp_acl_mdl_get_output_dims(modelDesc_, OUTPUT_NUM_ID, &outDims);
    if (ret != SVP_ACL_SUCCESS || outDims.dim_count <= 2) {
        ERROR_LOG("svp_acl_mdl_get_output_dims error!");
    }

    int outWidth = outDims.dims[outDims.dim_count - 1]; // 8400
    auto stride = svp_acl_mdl_get_output_default_stride(modelDesc_, 0) / sizeof(float);
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, 0);
    auto xCenter = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(dataBuffer));
    auto yCenter = xCenter + stride;
    auto boxWidth = yCenter + stride;
    auto boxHeight = boxWidth + stride;
    auto classScore = boxHeight + stride; // This is the single class score

    for (int j = 0; j < outWidth; j++) {
        float score = *classScore;
        if (score > scoreThr_) {
            vaildBox.push_back({score, xCenter[j], yCenter[j], boxWidth[j],
                boxHeight[j], 0.0f}); // classId is 0 (face)
        }
        classScore++;
    }
}

void ModelProcess::OutputModelResultYoloV8Face(int32_t modelId, const string& imgName) const
{
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, INPUT_IMG_ID, &inDims);
    if (ret != SVP_ACL_SUCCESS || inDims.dim_count <= 2) {
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
    }
    int imgHeight = inDims.dims[inDims.dim_count - 2];
    int imgWidth = inDims.dims[inDims.dim_count - 1];
    INFO_LOG("input image width[%d]; height[%d]", imgWidth, imgHeight);

    vector<vector<float>> vaildBox;
    FilterYolov8FaceBox(modelId, vaildBox);
    std::sort(vaildBox.begin(), vaildBox.end(), [](const vector<float>& veci, const vector<float>& vecj) {
        if (veci[0] > vecj[0]) {
            return true;
        }
        return false;
    });

    vector<vector<float>> bboxes;
    const float nmsThr = 0.45;
    MulticlassNms(bboxes, vaildBox, nmsThr);
    if (bboxes.size() == 0) {
        INFO_LOG("total valid num is zero");
        return;
    }
    vector<Detection> detections {};
    std::sort(bboxes.begin(), bboxes.end(), Cmp);
    PrintResult(bboxes);
    WriteResult(bboxes, imgWidth, imgHeight, modelId);
    SaveResult(bboxes, detections);

    DrawResult(imgName, detections, imgWidth, imgHeight, modelId);
    INFO_LOG("output data success");
    return;
}

void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(output_); ++i) {
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, i);
        void* data = svp_acl_get_data_buffer_addr(dataBuffer);
        (void)svp_acl_rt_free(data);
        (void)svp_acl_destroy_data_buffer(dataBuffer);
    }

    (void)svp_acl_mdl_destroy_dataset(output_);
    output_ = nullptr;
}

Result ModelProcess::Execute()
{
    svp_acl_error ret = svp_acl_mdl_execute(modelId_, input_, output_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }
    executeNum_++;
    INFO_LOG("model execute success");
    return SUCCESS;
}

Result ModelProcess::CreateBuf(int index)
{
    void *bufPtr = nullptr;
    size_t bufSize = 0;
    size_t bufStride = 0;
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = GetInputStrideParam(index, bufSize, bufStride, inDims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetInputStrideParam failed");
        return FAILED;
    }

    ret = svp_acl_rt_malloc(&bufPtr, bufSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu", bufSize);
        return FAILED;
    }
    Utils::InitData(static_cast<int8_t*>(bufPtr), bufSize);

    ret = CreateInput(bufPtr, bufSize, bufStride);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        svp_acl_rt_free(bufPtr);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInputBuf(const string& filePath)
{
    size_t devSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims inputDims;
    // only support single input model
    Result ret = GetInputStrideParam(0, devSize, stride, inputDims);
    if (ret != SUCCESS) {
        ERROR_LOG("GetStrideParam error");
        return FAILED;
    }
    size_t dataSize = GetInputDataSize(0);
    if (dataSize == 0) {
        ERROR_LOG("GetInputDataSize == 0 error");
        return FAILED;
    }
    void *picDevBuffer = Utils::GetDeviceBufferOfFile(filePath, inputDims, stride, dataSize);
    if (picDevBuffer == nullptr) {
        ERROR_LOG("get pic device buffer failed");
        return FAILED;
    }

    ret = InitInput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute InitInput failed");
        svp_acl_rt_free(picDevBuffer);
        return FAILED;
    }

    ret = CreateInput(picDevBuffer, devSize, stride);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        svp_acl_rt_free(picDevBuffer);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateTaskBufAndWorkBuf()
{
    // 2 is stand taskbuf and workbuf
    if (svp_acl_mdl_get_num_inputs(modelDesc_) <= 2) {
        ERROR_LOG("input dataset Num is error.");
        return FAILED;
    }
    size_t datasetSize = svp_acl_mdl_get_dataset_num_buffers(input_);
    if (datasetSize == 0) {
        ERROR_LOG("input dataset Num is 0.");
        return FAILED;
    }
    for (size_t loop = datasetSize; loop < svp_acl_mdl_get_num_inputs(modelDesc_); loop++) {
        Result ret = CreateBuf(loop);
        if (ret != SUCCESS) {
            ERROR_LOG("execute Create taskBuffer and workBuffer failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

void ModelProcess::Unload()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    svp_acl_error ret = svp_acl_mdl_unload(modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)svp_acl_mdl_destroy_desc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        svp_acl_rt_free(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        svp_acl_rt_free(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}
