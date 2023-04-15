/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Hypertext.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <map>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR HypertextNerBase::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    maxLength_ = initParam.maxLength;
    resultName_ = initParam.resultName;
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
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
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

APP_ERROR HypertextNerBase::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR HypertextNerBase::ReadTensorFromFile(const std::string &file, int32_t *data,
                                               uint32_t size) {
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

APP_ERROR HypertextNerBase::ReadInputTensor(int32_t *data, const std::string &fileName,
                                            uint32_t index,
                                            std::vector<MxBase::TensorBase> *inputs,
                                            const uint32_t size) {
    const uint32_t dataSize = modelDesc_.inputTensors[index].tensorSize;
    MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE,
                                     deviceId_);
    MxBase::MemoryData memoryDataSrc(reinterpret_cast<void *>(data), dataSize,
                                     MxBase::MemoryData::MEMORY_HOST_MALLOC);
    APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc and copy failed.";
        return ret;
    }
    std::vector<uint32_t> shape = {1, size};
    inputs->push_back(MxBase::TensorBase(memoryDataDst, false, shape,
                                         MxBase::TENSOR_DTYPE_INT32));
    delete[] data;
    return APP_ERR_OK;
}

APP_ERROR HypertextNerBase::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                      std::vector<MxBase::TensorBase> *outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        MxBase::TensorBase tensor(shape, dtypes[i],
                                  MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                  deviceId_);
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
    double costMs =
            std::chrono::duration<double, std::milli>(endTime - startTime).count();
    g_inferCost.push_back(costMs);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR HypertextNerBase::PostProcess(std::vector<MxBase::TensorBase> *outputs,
                                        std::vector<uint32_t> *predict) {
    MxBase::TensorBase &tensor = outputs->at(0);
    APP_ERROR ret = tensor.ToHost();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Tensor deploy to host failed.";
        return ret;
    }
    // check tensor is available
    auto outputShape = tensor.GetShape();
    uint32_t length = outputShape[0];
    void *data = tensor.GetBuffer();
    for (uint32_t i = 0; i < length; i++) {
        int32_t value = *(reinterpret_cast<int32_t *>(data) + i);
        predict->push_back(value);
    }
    return APP_ERR_OK;
}

APP_ERROR HypertextNerBase::WriteResult(const std::string &fileName,
                                        const std::vector<uint32_t> &predict) {
    std::string resultPathName = "result";
    // create result directory when it does not exit
    if (access(resultPathName.c_str(), 0) != 0) {
        int ret = mkdir(resultPathName.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
        if (ret != 0) {
            LogError << "Failed to create result directory: " << resultPathName
                     << ", ret = " << ret;
            return APP_ERR_COMM_OPEN_FAIL;
        }
    }
    // create result file under result directory
    resultPathName = resultPathName + "/" + resultName_;
    std::ofstream tfile(resultPathName, std::ofstream::app);
    if (tfile.fail()) {
        LogError << "Failed to open result file: " << resultPathName;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // write inference result into file
    LogInfo << "==============================================================";
    LogInfo << "Infer finished!";

    tfile << predict[0];
    tfile << std::endl;

    LogInfo << "==============================================================";
    tfile.close();
    return APP_ERR_OK;
}

APP_ERROR HypertextNerBase::Process(const std::string &inferIdsPath, const std::string &inferNgradPath,
                                    const std::string &fileName) {
    std::string inputAdjacencyFile = inferIdsPath;
    std::string inputFeatureFile = inferNgradPath;

    APP_ERROR ret;
    std::ifstream fp1(inputAdjacencyFile);
    std::string line1;
    std::ifstream fp2(inputFeatureFile);
    std::string line2;
    while (std::getline(fp1, line1), std::getline(fp2, line2)) {
        int32_t *data1 = new int32_t[maxLength_];
        int32_t *data2 = new int32_t[maxLength_];
        std::vector<MxBase::TensorBase> inputs = {};
        std::vector<MxBase::TensorBase> outputs = {};
        std::string number1;
        std::istringstream readstr1(line1);
        std::string number2;
        std::istringstream readstr2(line2);
        for (uint32_t j = 0; j < maxLength_; j++) {
            std::getline(readstr1, number1, ' ');
            data1[j] = atoi(number1.c_str());
            std::getline(readstr2, number2, ' ');
            data2[j] = atoi(number2.c_str());
        }
        ret = ReadInputTensor(data1, inputAdjacencyFile, INPUT_ADJACENCY, &inputs, maxLength_);
        if (ret != APP_ERR_OK) {
            LogError << "Read input adjacency failed, ret=" << ret << ".";
            return ret;
        }
        ret = ReadInputTensor(data2, inputFeatureFile, INPUT_FEATURE, &inputs,
                              maxLength_);
        if (ret != APP_ERR_OK) {
            LogError << "Read input feature file failed, ret=" << ret << ".";
            return ret;
        }

        ret = Inference(inputs, &outputs);
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        std::vector<uint32_t> predict;
        ret = PostProcess(&outputs, &predict);
        if (ret != APP_ERR_OK) {
            LogError << "PostProcess failed, ret=" << ret << ".";
            return ret;
        }
        ret = WriteResult(fileName, predict);
        if (ret != APP_ERR_OK) {
            LogError << "save result failed, ret=" << ret << ".";
            return ret;
        }
    }
    return APP_ERR_OK;
}
