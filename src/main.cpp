/**
* @file main.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <iostream>
#include <map>
#include <string>
#include "sample_process.h"
#include "utils.h"

using namespace std;

static void LogInfoContext(void)
{
    INFO_LOG("./main param\nparam is model and input image bin pair(default 8_face)");
    INFO_LOG("param 8_face: yolov8_face.om and dog_bike_car_yolov8.bin");
}

int main(int argc, char *argv[])
{
    LogInfoContext();
    int modelOpt = static_cast<int>(Yolo::YOLOV8_FACE);
    if (argc > 1) {
        string tmp(argv[1]);
        if (tmp == "8_face") {
            modelOpt = static_cast<int>(Yolo::YOLOV8_FACE);
        } else {
            ERROR_LOG("option invalid %s, only support 8_face", argv[1]);
            return FAILED;
        }
        INFO_LOG("yolo %s", argv[1]);
    }
    SampleProcess sampleProcess(modelOpt);
    Result ret = sampleProcess.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }

    ret = sampleProcess.Process();
    if (ret != SUCCESS) {
        ERROR_LOG("sample process failed");
        return FAILED;
    }

    INFO_LOG("execute sample success");
    return SUCCESS;
}