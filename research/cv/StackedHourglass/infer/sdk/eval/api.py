# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""infer class"""
import json
import logging
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector


class SdkApi:
    '''sdk api'''
    INFER_TIMEOUT = 3000
    STREAM_NAME = "im_hourglass"

    def __init__(self, pipeline_cfg="./config/StackedHourglass.pipeline"):
        self.pipeline_cfg = pipeline_cfg
        self._stream_api = None
        self._data_input = None
        self._device_id = None

    def init(self):
        '''sdk init '''
        with open(self.pipeline_cfg, 'r') as fp:
            self._device_id = int(
                json.loads(fp.read())[self.STREAM_NAME]["stream_config"]
                ["deviceId"])
            print(f"The device id: {self._device_id}.")
        # create api
        self._stream_api = StreamManagerApi()
        # init stream mgr
        ret = self._stream_api.InitManager()
        if ret != 0:
            print(f"Failed to init stream manager, ret={ret}.")
            return False
        # create streams
        with open(self.pipeline_cfg, 'rb') as fp:
            pipe_line = fp.read()
        ret = self._stream_api.CreateMultipleStreams(pipe_line)
        if ret != 0:
            print(f"Failed to create stream, ret={ret}.")
            return False

        self._data_input = MxDataInput()
        return True

    def __del__(self):
        if not self._stream_api:
            return

        self._stream_api.DestroyAllStreams()

    def send_data_input(self, stream_name, plugin_id, input_data):
        """" send data to om """
        data_input = MxDataInput()
        data_input.data = input_data
        unique_id = self._stream_api.SendData(stream_name, plugin_id,
                                              data_input)
        if unique_id < 0:
            logging.error("Fail to send data to stream.")
            return False
        return True

    def send_package_buf(self, stream_name, data, appsrc_pos):
        """" send data buffer to om """
        # create MxpiTensorPackageList
        tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package = tensor_package_list.tensorPackageVec.add()
        array_bytes = data.tobytes()
        data_input = MxDataInput()
        data_input.data = array_bytes
        tensor_vec = tensor_package.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0
        for i in data.shape:
            tensor_vec.tensorShape.append(i)
        tensor_vec.dataStr = data_input.data
        tensor_vec.tensorDataSize = len(array_bytes)
        key = "appsrc{}".format(appsrc_pos).encode('utf-8')
        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b"MxTools.MxpiTensorPackageList"
        protobuf.protobuf = tensor_package_list.SerializeToString()
        protobuf_vec.push_back(protobuf)
        unique_id = self._stream_api.SendProtobuf(
            stream_name, appsrc_pos, protobuf_vec)
        return unique_id

    def _send_protobuf(self, stream_name, plugin_id, element_name, buf_type,
                       pkg_list):
        """" send data buffer to om """
        protobuf = MxProtobufIn()
        protobuf.key = element_name.encode("utf-8")
        protobuf.type = buf_type
        protobuf.protobuf = pkg_list.SerializeToString()
        protobuf_vec = InProtobufVector()
        protobuf_vec.push_back(protobuf)
        err_code = self._stream_api.SendProtobuf(stream_name, plugin_id,
                                                 protobuf_vec)
        if err_code != 0:
            logging.error(
                "Failed to send data to stream, stream_name(%s), plugin_id(%s), element_name(%s), "
                "buf_type(%s), err_code(%s).", stream_name, plugin_id,
                element_name, buf_type, err_code)
            return False
        return True

    def send_img_input(self, stream_name, plugin_id, element_name, input_data,
                       img_size):
        """" send image data to om """
        vision_list = MxpiDataType.MxpiVisionList()
        vision_vec = vision_list.visionVec.add()
        vision_vec.visionInfo.format = 1
        vision_vec.visionInfo.width = img_size[1]
        vision_vec.visionInfo.height = img_size[0]
        vision_vec.visionInfo.widthAligned = img_size[1]
        vision_vec.visionInfo.heightAligned = img_size[0]
        vision_vec.visionData.memType = 0
        vision_vec.visionData.dataStr = input_data
        vision_vec.visionData.dataSize = len(input_data)
        buf_type = b"MxTools.MxpiVisionList"
        return self._send_protobuf(stream_name, plugin_id, element_name,
                                   buf_type, vision_list)

    def send_tensor_input(self, stream_name, plugin_id, element_name,
                          input_data, input_shape, data_type):
        """" send tensor data to om """
        tensor_list = MxpiDataType.MxpiTensorPackageList()
        tensor_pkg = tensor_list.tensorPackageVec.add()
        # init tensor vector
        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.deviceId = self._device_id
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(input_shape)
        tensor_vec.tensorDataType = data_type
        tensor_vec.dataStr = input_data
        tensor_vec.tensorDataSize = len(input_data)

        buf_type = b"MxTools.MxpiTensorPackageList"
        return self._send_protobuf(stream_name, plugin_id, element_name,
                                   buf_type, tensor_list)

    def get_result(self, stream_name, out_plugin_id=0):
        """get_result"""
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = self._stream_api.GetProtobuf(
            stream_name, out_plugin_id, key_vec)
        if infer_result.size() == 0:
            print("inferResult is null")
            return None
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        return result.tensorPackageVec[0].tensorVec[0].dataStr, result.tensorPackageVec[0].tensorVec[1].dataStr
