/**
* @file model_process.h
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef MODEL_PROCESS_H
#define MODEL_PROCESS_H

#include <iostream>
#include <vector>
#include "utils.h"
#include "acl/svp_acl.h"
#include "opencv2/opencv.hpp"

struct Detection {
    int classId { 0 };
    std::string className {};
    float confidence { 0.0f };
    cv::Scalar color {};
    cv::Rect box {};
};

class ModelProcess {
public:
    /**
    * @brief Constructor
    */
    ModelProcess();

    /**
    * @brief Destructor
    */
    ~ModelProcess();

    /**
    * @brief load model from file with mem
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModelFromFileWithMem(const std::string& modelPath);

    /**
    * @brief unload model
    */
    void Unload();

    /**
    * @brief create dataset
    * @return result
    */
    Result InitInput();

    /**
    * @brief create model desc
    * @return result
    */
    Result CreateDesc();

    /**
    * @brief destroy desc
    */
    void DestroyDesc();

    /**
    * @brief create model input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @return result
    */
    Result CreateInput(void *inputDataBuffer, size_t bufferSize, int stride);

    Result CreateInputBuf(const std::string& filePath);

    Result CreateTaskBufAndWorkBuf();

    Result SetDetParas(int32_t modelId);

    /**
    * @brief destroy input resource
    */
    void DestroyInput();

    /**
    * @brief create output buffer
    * @return result
    */
    Result CreateOutput();

    /**
    * @brief destroy output resource
    */
    void DestroyOutput();

    /**
    * @brief model execute
    * @return result
    */
    Result Execute();

    /**
    * @brief dump model output result to file
    */
    void DumpModelOutputResult() const;

    /**
    * @brief get model output result
    */
    void OutputModelResult(int32_t modelId, const std::string& imgName) const;

    Result CreateBuf(int index);

    Result GetInputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const;

    Result GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const;

    size_t GetInputDataSize(int index) const;

    size_t GetOutputDataSize(int index) const;

private:
    void WriteOutput(const std::string& outputFileName, size_t index) const;

    Result ClearOutputStrideInvalidBuf(std::vector<int8_t>& buffer, size_t index) const;

    Result SetDetParas(const std::vector<float>& detPara);
    void OutputModelResultYoloV(int32_t modelId, const std::string& imgName) const;
    void OutputModelResultYoloX(int32_t modelId, const std::string& imgName) const;
    void OutputModelResultYoloVCpu(int32_t modelId, const std::string& imgName) const;
    void OutputModelResultYoloV8Cpu(int32_t modelId, const std::string& imgName) const;
    void OutputModelResultYoloV8Face(int32_t modelId, const std::string& imgName) const;
    void FilterYolov5Box(std::vector<std::vector<float>>& vaildBox) const;
    void ProcessPerDectection(size_t detectIdx, std::vector<std::vector<float>>& vaildBox) const;
    void FilterYolov8Box(int32_t modelId, std::vector<std::vector<float>>& vaildBox) const;
    void FilterYolov8FaceBox(int32_t modelId, std::vector<std::vector<float>>& vaildBox) const;

    uint32_t executeNum_ { 0 };
    uint32_t modelId_ { 0 };
    size_t modelMemSize_ { 0 };
    size_t modelWeightSize_ { 0 };
    void *modelMemPtr_ { nullptr };
    void *modelWeightPtr_ { nullptr };
    bool loadFlag_ { false };
    svp_acl_mdl_desc *modelDesc_ { nullptr };
    svp_acl_mdl_dataset *input_ { nullptr };
    svp_acl_mdl_dataset *output_ { nullptr };
    float scoreThr_ { 0.25 };
    bool isMultiThr {false};
    uint32_t rpnDataH {0};
    uint32_t rpnDataW {0};
};
constexpr uint8_t SCALE_SIZE = 3;
constexpr uint8_t CLASS_NUM = 80;
constexpr uint8_t OUT_PARM_NUM = 85; /* x, y, w,h, obj , class(80) */

struct DetectionInnerParam {
    float *outData { nullptr };
    size_t detectIdx { 0 };
    size_t wStrideOffset { 0 };
    float scoreThr { 0.0f};
    uint32_t outWidth { 0 };
    uint32_t chnStep { 0 };
    uint32_t outHeightIdx { 0 };
    uint32_t objScoreOffset { 0 };
};

#endif // MODEL_PROCESS_H
