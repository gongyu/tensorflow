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

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

REGISTER_OP("TOPTEngineOp")
    .Attr("proto_graph: string")
    .Attr("input_names: list(string)")
    .Attr("output_names: list(string)")
    .Attr("InT: list(type) >= 0")
    .Attr("OutT: list(type)")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
