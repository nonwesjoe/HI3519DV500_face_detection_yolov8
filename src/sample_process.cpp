/**
* @file sample_process.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "sample_process.h"
#include <map>
#include "model_process.h"
#include "acl/svp_acl.h"
#include "utils.h"

using namespace std;

static const map<Yolo, string> g_modelPath = {
    { Yolo::YOLOV8_FACE, "yolov8_face_original.om" },
};

static const map<Yolo, string> g_imgPath = {
    { Yolo::YOLOV8_FACE, "img.bin" },
};

static const map<Yolo, string> g_imageFile = {
    { Yolo::YOLOV8_FACE, "img.jpg" },
};

SampleProcess::SampleProcess()
{
}

SampleProcess::SampleProcess(int32_t modelId)
{
    this->modelId = modelId;
    isCpuProcess_ = (modelId == static_cast<int>(Yolo::YOLOV8_FACE));
}

SampleProcess::~SampleProcess()
{
    DestroyResource();
}

Result SampleProcess::InitResource()
{
    // ACL init
    const char* aclConfigPath = "../src/acl.json";
    svp_acl_error ret = svp_acl_init(aclConfigPath);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // set device
    ret = svp_acl_rt_set_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // set no timeout
    ret = svp_acl_rt_set_op_wait_timeout(0);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl set op wait time failed");
        return FAILED;
    }
    INFO_LOG("set op wait time success");

    // create context (set current)
    ret = svp_acl_rt_create_context(&context_, deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = svp_acl_rt_create_stream(&stream_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    svp_acl_rt_run_mode runMode;
    ret = svp_acl_rt_get_run_mode(&runMode);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    if (runMode != SVP_ACL_DEVICE) {
        ERROR_LOG("acl run mode failed");
        return FAILED;
    }
    INFO_LOG("get run mode success");
    return SUCCESS;
}

Result SampleProcess::Process()
{
    ModelProcess modelProcess;
    const string omModelPath = "../model/" + g_modelPath.at(static_cast<Yolo>(modelId));
    Result ret = modelProcess.LoadModelFromFileWithMem(omModelPath.c_str());
    CHECK_EXPS_RETURN(ret != SUCCESS, FAILED, "execute LoadModelFromFileWithMem failed");

    ret = modelProcess.CreateDesc();
    CHECK_EXPS_RETURN(ret != SUCCESS, FAILED, "execute CreateDesc failed");

    ret = modelProcess.CreateOutput();
    CHECK_EXPS_RETURN(ret != SUCCESS, FAILED, "execute CreateOutput failed");

    std::string imgPath = "../data/" + g_imgPath.at(static_cast<Yolo>(modelId));

    string testFile[] = { imgPath };

    for (size_t index = 0; index < sizeof(testFile) / sizeof(testFile[0]); ++index) {
        INFO_LOG("start to process file:%s", testFile[index].c_str());
        ret = modelProcess.CreateInputBuf(testFile[index]);
        CHECK_EXPS_RETURN(ret != SUCCESS, FAILED, "CreateInputBuf failed");
        if (!isCpuProcess_) {
            ret = modelProcess.SetDetParas(modelId);
            CHECK_EXPS_RETURN(ret != SUCCESS, FAILED, "SetDetParas failed");
        }

        ret = modelProcess.CreateTaskBufAndWorkBuf();
        CHECK_EXPS_RETURN(ret != SUCCESS, FAILED, "CreateTaskBufAndWorkBuf failed");

        ret = modelProcess.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            modelProcess.DestroyInput();
            return FAILED;
        }

        std::string imgName = "../data/" + g_imageFile.at(static_cast<Yolo>(modelId));
        modelProcess.DumpModelOutputResult();
        modelProcess.OutputModelResult(modelId, imgName);

        // release model input buffer
        modelProcess.DestroyInput();
    }

    return SUCCESS;
}

void SampleProcess::DestroyResource()
{
    svp_acl_error ret;
    if (stream_ != nullptr) {
        ret = svp_acl_rt_destroy_stream(stream_);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = svp_acl_rt_destroy_context(context_);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = svp_acl_rt_reset_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = svp_acl_finalize();
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}
