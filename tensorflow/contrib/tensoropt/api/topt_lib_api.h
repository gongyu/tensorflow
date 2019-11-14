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

#ifndef TENSORFLOW_CONTRIB_TENSOROPT_API_TOPT_LIB_API_H_
#define TENSORFLOW_CONTRIB_TENSOROPT_API_TOPT_LIB_API_H_

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

#include "tensorflow/core/platform/default/logging.h"
#include "tensoropt/tensoropt.hpp"

#define TOPT_CHECK_OK(CMD)                                                     \
  {                                                                            \
    auto ret = ANeuralNetworks##CMD;                                           \
    if (ret != ResultCode::ANEURALNETWORKS_NO_ERROR)                           \
      LOG(FATAL) << "TensorOpt failed with " << int(ret) << " at " << __FILE__ \
                 << ":" << __LINE__;                                           \
  }

inline tensorflow::Status ConvertDTypeToTOPTTensorType(
    tensorflow::DataType tf_dtype, ANeuralNetworksOperandCode* topt_dtype) {
  switch (tf_dtype) {
    case tensorflow::DataType::DT_FLOAT:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_FLOAT32;
      break;
    case tensorflow::DataType::DT_INT32:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_INT32;
      break;
    case tensorflow::DataType::DT_BOOL:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_BOOL8;
      break;
    case tensorflow::DataType::DT_INVALID:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_INVALID;
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          "Unsupported data type " + tensorflow::DataTypeString(tf_dtype));
  }
  return tensorflow::Status::OK();
}

inline tensorflow::Status ConvertDTypeToTOPTHostType(
    tensorflow::DataType tf_dtype, ANeuralNetworksOperandCode* topt_dtype) {
  switch (tf_dtype) {
    case tensorflow::DataType::DT_FLOAT:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_FLOAT32;
      break;
    case tensorflow::DataType::DT_INT32:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_INT32;
      break;
    case tensorflow::DataType::DT_UINT32:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_UINT32;
      break;
    case tensorflow::DataType::DT_BOOL:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_BOOL;
      break;
    case tensorflow::DataType::DT_INVALID:
      *topt_dtype = ANeuralNetworksOperandCode::ANEURALNETWORKS_INVALID;
      break;
    default:
      return tensorflow::errors::InvalidArgument(
          "Unsupported data type " + tensorflow::DataTypeString(tf_dtype));
  }
  return tensorflow::Status::OK();
}

inline tensorflow::DataType ConvertTOPTTypeToDType(
    ANeuralNetworksOperandCode topt_dtype) {
  switch (topt_dtype) {
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_FLOAT32:
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_FLOAT32:
      return tensorflow::DataType::DT_FLOAT;
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_INT32:
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_INT32:
      return tensorflow::DataType::DT_INT32;
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_UINT32:
      return tensorflow::DataType::DT_UINT32;
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_BOOL:
    case ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_BOOL8:
      return tensorflow::DataType::DT_BOOL;
    default:
      return tensorflow::DataType::DT_INVALID;
  }
}

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

#endif  // TENSORFLOW_CONTRIB_TENSOROPT_API_TOPT_LIB_API_H_
