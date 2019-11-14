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
#ifndef TENSORFLOW_CONTRIB_TENSOROPT_CONVERT_CONVERT_GRAPH_H_
#define TENSORFLOW_CONTRIB_TENSOROPT_CONVERT_CONVERT_GRAPH_H_

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

namespace tensorflow {
namespace tensoropt {
namespace convert {

tensorflow::Status ConvertGraphDefToTensorOpt(
    const tensorflow::GraphDef& graph, const std::vector<string>& output_names,
    tensorflow::GraphDef* new_graph_def, int minimum_segment_size,
    const tensorflow::grappler::Cluster* cluster = nullptr);

}  // namespace convert
}  // namespace tensoropt
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

#endif  // TENSORFLOW_CONTRIB_TENSOROPT_CONVERT_CONVERT_GRAPH_H_
