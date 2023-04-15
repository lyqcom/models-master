/*
 * Copyright 2021. Huawei Technologies Co., Ltd.
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
#include "TextfusenetMindsporePost.h"
#include <string>
#include <memory>
#include <boost/property_tree/json_parser.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "acl/acl.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
// Output Tensor
const int OUTPUT_TENSOR_SIZE = 4;
const int OUTPUT_BBOX_SIZE = 3;
const int OUTPUT_BBOX_TWO_INDEX_SHAPE = 5;
const int OUTPUT_BBOX_INDEX = 0;
const int OUTPUT_CLASS_INDEX = 1;
const int OUTPUT_MASK_INDEX = 2;
const int OUTPUT_MASK_AREA_INDEX = 3;

const int BBOX_INDEX_LX = 0;
const int BBOX_INDEX_LY = 1;
const int BBOX_INDEX_RX = 2;
const int BBOX_INDEX_RY = 3;
const int BBOX_INDEX_PROB = 4;
const int BBOX_INDEX_SCALE_NUM = 5;
}  // namespace

namespace MxBase {

TextfusenetMindsporePost &TextfusenetMindsporePost::operator=(const TextfusenetMindsporePost &other) {
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator=(other);
    return *this;
}

APP_ERROR TextfusenetMindsporePost::ReadConfigParams() {
    APP_ERROR ret = configData_.GetFileValue<uint32_t>("CLASS_NUM", classNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No CLASS_NUM in config file, default value(" << classNum_ << ").";
    }
    ret = configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No SCORE_THRESH in config file, default value(" << scoreThresh_ << ").";
    }

    ret = configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No IOU_THRESH in config file, default value(" << iouThresh_ << ").";
    }

    ret = configData_.GetFileValue<uint32_t>("RPN_MAX_NUM", rpnMaxNum_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No RPN_MAX_NUM in config file, default value(" << rpnMaxNum_ << ").";
    }

    ret = configData_.GetFileValue<uint32_t>("MAX_PER_IMG", maxPerImg_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MAX_PER_IMG in config file, default value(" << maxPerImg_ << ").";
    }

    ret = configData_.GetFileValue<float>("MASK_THREAD_BINARY", maskThrBinary_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MASK_THREAD_BINARY in config file, default value(" << maskThrBinary_ << ").";
    }

    ret = configData_.GetFileValue<uint32_t>("MASK_SHAPE_SIZE", maskSize_);
    if (ret != APP_ERR_OK) {
        LogWarn << GetError(ret) << "No MASK_SHAPE_SIZE in config file, default value(" << maskSize_ << ").";
    }

    LogInfo << "The config parameters of post process are as follows: \n"
            << "  CLASS_NUM: " << classNum_ << " \n"
            << "  SCORE_THRESH: " << scoreThresh_ << " \n"
            << "  IOU_THRESH: " << iouThresh_ << " \n"
            << "  RPN_MAX_NUM: " << rpnMaxNum_ << " \n"
            << "  MAX_PER_IMG: " << maxPerImg_ << " \n"
            << "  MASK_THREAD_BINARY: " << maskThrBinary_ << " \n"
            << "  MASK_SHAPE_SIZE: " << maskSize_;
}

APP_ERROR
TextfusenetMindsporePost::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogInfo << "Begin to initialize TextfusenetMindsporePost.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superinit  in ObjectPostProcessBase.";
        return ret;
    }

    ReadConfigParams();

    LogInfo << "End to initialize TextfusenetMindsporePost.";
    return APP_ERR_OK;
}

APP_ERROR TextfusenetMindsporePost::DeInit() {
    LogInfo << "Begin to deinitialize TextfusenetMindsporePost.";
    LogInfo << "End to deinitialize TextfusenetMindsporePost.";
    return APP_ERR_OK;
}

bool TextfusenetMindsporePost::IsValidTensors(const std::vector<TensorBase> &tensors) const {
    if (tensors.size() < OUTPUT_TENSOR_SIZE) {
        LogError << "The number of tensor (" << tensors.size() << ") is less than required (" << OUTPUT_TENSOR_SIZE
                 << ")";
        return false;
    }

    auto bboxShape = tensors[OUTPUT_BBOX_INDEX].GetShape();
    if (bboxShape.size() != OUTPUT_BBOX_SIZE) {
        LogError << "The number of tensor[" << OUTPUT_BBOX_INDEX << "] dimensions (" << bboxShape.size()
                 << ") is not equal to (" << OUTPUT_BBOX_SIZE << ")";
        return false;
    }

    uint32_t total_num = rpnMaxNum_;
    if (bboxShape[VECTOR_SECOND_INDEX] != total_num) {
        LogError << "The output tensor is mismatched: " << total_num << "/" << bboxShape[VECTOR_SECOND_INDEX] << ").";
        return false;
    }

    if (bboxShape[VECTOR_THIRD_INDEX] != OUTPUT_BBOX_TWO_INDEX_SHAPE) {
        LogError << "The number of bbox[" << VECTOR_THIRD_INDEX << "] dimensions (" << bboxShape[VECTOR_THIRD_INDEX]
                 << ") is not equal to (" << OUTPUT_BBOX_TWO_INDEX_SHAPE << ")";
        return false;
    }

    auto classShape = tensors[OUTPUT_CLASS_INDEX].GetShape();
    if (classShape[VECTOR_SECOND_INDEX] != total_num) {
        LogError << "The output tensor is mismatched: (" << total_num << "/" << classShape[VECTOR_SECOND_INDEX]
                 << "). ";
        return false;
    }

    auto maskShape = tensors[OUTPUT_MASK_INDEX].GetShape();
    if (maskShape[VECTOR_SECOND_INDEX] != total_num) {
        LogError << "The output tensor is mismatched: (" << total_num << "/" << maskShape[VECTOR_SECOND_INDEX] << ").";
        return false;
    }

    auto maskAreaShape = tensors[OUTPUT_MASK_AREA_INDEX].GetShape();
    if (maskAreaShape[VECTOR_SECOND_INDEX] != total_num) {
        LogError << "The output tensor is mismatched: (" << total_num << "/" << maskAreaShape[VECTOR_SECOND_INDEX]
                 << ").";
        return false;
    }

    if (maskAreaShape[VECTOR_THIRD_INDEX+1] != maskSize_) {
        LogError << "The output tensor of mask is mismatched: (" << maskAreaShape[VECTOR_THIRD_INDEX] << "/"
                 << maskSize_ << ").";
        return false;
    }
    return true;
}

static bool CompareDetectBoxes(const MxBase::DetectBox &box1, const MxBase::DetectBox &box2) {
    return box1.prob > box2.prob;
}

static void GetDetectBoxesTopK(std::vector<MxBase::DetectBox> &detBoxes, size_t kVal) {
    std::sort(detBoxes.begin(), detBoxes.end(), CompareDetectBoxes);
    if (detBoxes.size() <= kVal) {
        return;
    }

    LogDebug << "Total detect boxes: " << detBoxes.size() << ", kVal: " << kVal;
    detBoxes.erase(detBoxes.begin() + kVal, detBoxes.end());
}

void TextfusenetMindsporePost::GetValidDetBoxes(const std::vector<TensorBase> &tensors,
                                                std::vector<DetectBox> &detBoxes,
                                                const uint32_t batchNum,
                                                const ResizedImageInfo &resizedImageInfo) {
    LogInfo << "Begin to GetValidDetBoxes Mask GetValidDetBoxes.";
    auto *bboxPtr = reinterpret_cast<aclFloat16 *>(GetBuffer(tensors[OUTPUT_BBOX_INDEX], batchNum));      // 1 * 90 * 5
    auto *labelPtr = reinterpret_cast<int32_t *>(GetBuffer(tensors[OUTPUT_CLASS_INDEX], batchNum));       // 1 * 90 * 1
    auto *maskPtr = reinterpret_cast<bool *>(GetBuffer(tensors[OUTPUT_MASK_INDEX], batchNum));            // 1 * 90 * 1
    // 1 * 90 * 64 * 28 * 28
    auto *maskAreaPtr = reinterpret_cast<aclFloat16 *>(GetBuffer(tensors[OUTPUT_MASK_AREA_INDEX], batchNum));
    float prob = 0;
    size_t total = rpnMaxNum_;
    for (size_t index = 0; index < total; ++index) {
        if (!maskPtr[index]) {
            continue;
        }
        size_t startIndex = index * BBOX_INDEX_SCALE_NUM;
        prob = aclFloat16ToFloat(bboxPtr[startIndex + BBOX_INDEX_PROB]);
        if (prob <= scoreThresh_ || labelPtr[index]) {
            continue;
        }

        MxBase::DetectBox detBox;
        float x1 = aclFloat16ToFloat(bboxPtr[startIndex + BBOX_INDEX_LX]);
        float y1 = aclFloat16ToFloat(bboxPtr[startIndex + BBOX_INDEX_LY]);
        float x2 = aclFloat16ToFloat(bboxPtr[startIndex + BBOX_INDEX_RX]);
        float y2 = aclFloat16ToFloat(bboxPtr[startIndex + BBOX_INDEX_RY]);
        detBox.x = (x1 + x2) / COORDINATE_PARAM;
        detBox.y = (y1 + y2) / COORDINATE_PARAM;
        detBox.width = x2 - x1;
        detBox.height = y2 - y1;
        detBox.prob = prob;
        detBox.classID = labelPtr[index];
        detBox.maskPtr = maskAreaPtr + index * 64 * maskSize_ * maskSize_ + maskSize_ * maskSize_;
        detBoxes.push_back(detBox);
    }
    GetDetectBoxesTopK(detBoxes, maxPerImg_);
}


APP_ERROR TextfusenetMindsporePost::MaskPostProcess(MxBase::ObjectInfo &objInfo,
                                                    void *maskPtr, const ResizedImageInfo &imgInfo) {
    // resize
    cv::Mat maskMat(maskSize_, maskSize_, CV_32FC1);
    // maskPtr aclFloat16 to float
    auto *maskAclPtr = reinterpret_cast<aclFloat16 *>(maskPtr);
    for (int row = 0; row < maskMat.rows; ++row) {
        aclFloat16 *maskTempPtr = maskAclPtr + row * maskMat.cols;;
        for (int col = 0; col < maskMat.cols; ++col) {
            maskMat.at<float>(row, col) = aclFloat16ToFloat(*(maskTempPtr + col));
        }
    }
    size_t bboxWidth = maskSize_;
    size_t bboxHeight = maskSize_;
    std::vector<std::vector<uint8_t>> maskResult(bboxHeight, std::vector<uint8_t>(bboxWidth));

    for (size_t row = 0; row < bboxHeight; ++row) {
        for (size_t col = 0; col < bboxWidth; ++col) {
            if (maskMat.at<float>(row, col) <= maskThrBinary_) {
                continue;
            }
            maskResult[row][col] = 1;
        }
    }
    objInfo.mask = maskResult;
    return APP_ERR_OK;
}

void TextfusenetMindsporePost::ConvertObjInfoFromDetectBox(std::vector<DetectBox> &detBoxes,
                                                        std::vector<ObjectInfo> &objectInfos,
                                                        const ResizedImageInfo &resizedImageInfo) {
    APP_ERROR ret = APP_ERR_OK;
    for (auto &detBoxe : detBoxes) {
        if (detBoxe.classID < 0) {
            continue;
        }
        ObjectInfo objInfo = {};
        objInfo.classId = static_cast<float>(detBoxe.classID);
        objInfo.className = configData_.GetClassName(detBoxe.classID);
        objInfo.confidence = detBoxe.prob;

        objInfo.x0 = std::max<float>(detBoxe.x - detBoxe.width / COORDINATE_PARAM, 0);
        objInfo.y0 = std::max<float>(detBoxe.y - detBoxe.height / COORDINATE_PARAM, 0);
        objInfo.x1 = std::max<float>(detBoxe.x + detBoxe.width / COORDINATE_PARAM, 0);
        objInfo.y1 = std::max<float>(detBoxe.y + detBoxe.height / COORDINATE_PARAM, 0);

        ret = MaskPostProcess(objInfo, detBoxe.maskPtr, resizedImageInfo);
        if (ret == APP_ERR_COMM_FAILURE) {
            continue;
        } else if (ret != APP_ERR_OK) {
            break;
        }

        objectInfos.push_back(objInfo);
    }

    if (ret != APP_ERR_OK && ret != APP_ERR_COMM_FAILURE) {
        LogError << "Convert obj info failed, ret:(" << ret << ").";
    }
}

void TextfusenetMindsporePost::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                  std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                  const std::vector<ResizedImageInfo> &resizedImageInfos) {
    LogDebug << "TextfusenetMindsporePost start to write results.";
    auto shape = tensors[OUTPUT_BBOX_INDEX].GetShape();
    uint32_t batchSize = shape[0];
    for (uint32_t i = 0; i < batchSize; ++i) {
        std::vector<MxBase::DetectBox> detBoxes;
        std::vector<ObjectInfo> objectInfo;
        GetValidDetBoxes(tensors, detBoxes, i, resizedImageInfos[i]);
        NmsSort(detBoxes, iouThresh_, MxBase::MAX);
        ConvertObjInfoFromDetectBox(detBoxes, objectInfo, resizedImageInfos[i]);
        objectInfos.push_back(objectInfo);
    }

    LogDebug << "TextfusenetMindsporePost write results succeeded.";
}

APP_ERROR TextfusenetMindsporePost::Process(const std::vector<TensorBase> &tensors,
                                         std::vector<std::vector<ObjectInfo>> &objectInfos,
                                         const std::vector<ResizedImageInfo> &resizedImageInfos,
                                         const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    LogDebug << "Begin to process TextfusenetMindsporePost.";
    auto inputs = tensors;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed, ret=" << ret;
        return ret;
    }
    ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
    LogInfo << "End to process TextfusenetMindsporePost.";
    return APP_ERR_OK;
}

extern "C" {
std::shared_ptr<MxBase::TextfusenetMindsporePost> GetObjectInstance() {
    LogInfo << "Begin to get TextfusenetMindSporePost instance.";
    auto instance = std::make_shared<TextfusenetMindsporePost>();
    LogInfo << "End to get TextfusenetMindsporePost Instance";
    return instance;
}
}

}  // namespace MxBase
