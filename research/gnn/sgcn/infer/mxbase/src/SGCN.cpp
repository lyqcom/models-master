/*
* Copyright 2022 Huawei Technologies Co., Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "SGCN.h"
#include <unistd.h>
#include <sys/stat.h>
#include <map>
#include <fstream>
#include <typeinfo>
#include <iomanip>
#include <iostream>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR sgcn::Init(const InitParam& initParam) {
    deviceId_ = initParam.deviceId;
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        LogError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR sgcn::DeInit() {
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR sgcn::ReadTensorFromFile(const std::string& file, int* data, uint32_t size) {
    if (data == NULL) {
        LogError << "input data is invalid.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    std::ifstream fp(file);
    std::string line;
    while (std::getline(fp, line)) {
        std::string number;
        std::istringstream readstr(line);
        for (uint32_t j = 0; j < size; j++) {
            std::getline(readstr, number, ' ');
            data[j] = atoi(number.c_str());
        }
    }
    return APP_ERR_OK;
}

APP_ERROR sgcn::ReadInputTensor(const std::string& fileName, uint32_t index,
    std::vector<MxBase::TensorBase>* inputs, uint32_t size,
    MxBase::TensorDataType type) {
    int* data = new int[size];
    APP_ERROR ret = ReadTensorFromFile(fileName, data, size);
    if (ret != APP_ERR_OK) {
        LogError << "Read Tensor From File failed.";
        return ret;
    }
    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    LogInfo << "dataSize:" << dataSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void*>(data), dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);
    ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    std::vector<uint32_t> shape = { 1, size };
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape, type));
    return APP_ERR_OK;
}

APP_ERROR sgcn::Inference(const std::vector<MxBase::TensorBase>& inputs,
    std::vector<MxBase::TensorBase>* outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs->push_back(tensor);
    }
    MxBase::DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, *outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR sgcn::PostProcess(std::vector<MxBase::TensorBase>* outputs, std::vector<float>* result) {
    LogInfo << "Outputs size:" << outputs->size();
    MxBase::TensorBase& tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    uint32_t classNum = outputShape[1];
    LogInfo << "output shape is: " << outputShape[0] << " " << outputShape[1] << std::endl;

    void* data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        for (uint32_t j = 0; j < classNum; j++) {
            // get real data by index, the variable 'data' is address
            float value = *(reinterpret_cast<float*>(data) + i * classNum + j);
            // LogInfo << "value " << value;
            result->push_back(value);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR sgcn::SaveResult(std::vector<float >* result) {
    std::ofstream outfile("res", std::ofstream::app);
    if (outfile.fail()) {
        LogError << "Failed to open result file: ";
        return APP_ERR_COMM_FAILURE;
    }
    for (uint32_t i = 0; i < result->size(); ++i) {
        outfile << std::setiosflags(std::ios::fixed) << std::setprecision(6) << result->at(i) << " ";
    }
    outfile << std::endl;
    outfile.close();
    return APP_ERR_OK;
}

APP_ERROR sgcn::Process(const std::string& inferPath, const std::string& dataType) {
    std::vector<MxBase::TensorBase> inputs = {};
    std::string inputReposFile = inferPath + "repos.txt";
    uint32_t size1 = (dataType == "otc") ? 29248 : 20430;
    uint32_t size2 = (dataType == "otc") ? 5044 : 2098;
    APP_ERROR ret = ReadInputTensor(inputReposFile, 0, &inputs, size1, MxBase::TENSOR_DTYPE_UINT32);
    if (ret != APP_ERR_OK) {
        LogError << "Read repos data failed, ret= " << ret << ".";
    }
    std::string inputRenegFile = inferPath + "reneg.txt";
    ret = ReadInputTensor(inputRenegFile, 1, &inputs, size2, MxBase::TENSOR_DTYPE_UINT32);
    if (ret != APP_ERR_OK) {
        LogError << "Read reneg data failed, ret= " << ret << ".";
    }

    std::vector<MxBase::TensorBase> outputs = {};
    ret = Inference(inputs, &outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<float> result;
    ret = PostProcess(&outputs, &result);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }
    ret = SaveResult(&result);
    if (ret != APP_ERR_OK) {
        LogError << "CalcF1Score read label failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}
