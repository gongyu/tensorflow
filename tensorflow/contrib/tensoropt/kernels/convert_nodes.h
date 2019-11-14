/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
   Copyright (C) Codeplay Software Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_CONVERT_NODES_H_
#define TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_CONVERT_NODES_H_

#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
#include "tensorflow/contrib/tensoropt/api/topt_lib_api.h"

namespace tensorflow {
namespace tensoropt {
namespace convert {

tensorflow::Status ConvertSubGraphToTensorOptModel(
    const std::list<tensorflow::Node*>& order,
    const std::vector<string>& input_names,
    const std::vector<TensorShape>& input_shapes,
    const std::vector<DataType>& input_types,
    const std::vector<string>& output_names,
    const std::vector<DataType>& output_types,
    std::vector<std::vector<uint8_t>>& weight_store,
    ANeuralNetworksModel** topt_model);

}  // namespace convert
}  // namespace tensoropt
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

#endif  // TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_CONVERT_NODES_H_
