/**
* @file sample_process.h
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef SAMPLE_PROCESS_H
#define SAMPLE_PROCESS_H

#include "utils.h"
#include "acl/svp_acl.h"

/**
* SampleProcess
*/
class SampleProcess {
public:
    /**
    * @brief Constructor
    */
    SampleProcess();

    /**
    * @brief Constructor
    */
    SampleProcess(int32_t modelId);

    /**
    * @brief Destructor
    */
    ~SampleProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource();

    /**
    * @brief sample process
    * @return result
    */
    Result Process();

private:
    void DestroyResource();
    int32_t modelId { 0 };
    int32_t deviceId_ { 0 };
    svp_acl_rt_context context_ { nullptr };
    svp_acl_rt_stream stream_ { nullptr };
    bool isCpuProcess_ { false };
};
#endif // SAMPLE_PROCESS_H
