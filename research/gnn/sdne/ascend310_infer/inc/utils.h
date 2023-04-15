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
#ifndef MINDSPORE_INFERENCE_UTILS_H_
#define MINDSPORE_INFERENCE_UTILS_H_

#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>

#include "include/api/types.h"

std::string RealPath(std::string_view path);
std::unordered_map< int, std::unordered_set<int> > GetGraph(std::string gfp, bool is_bidir);
std::vector<float> GetDataFromGraph(const std::unordered_set<int>& subgraph, int data_size);
std::vector<float> Tensor2Vector(const float* pdata, int data_size);
int WriteResult(const std::vector< std::vector<float> >& embeddings,
                const std::vector< std::vector<float> >& reconstructions);

#endif
