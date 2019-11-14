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

#ifndef TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_HASH_SHAPES_H_
#define TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_HASH_SHAPES_H_

#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"

namespace {

/**
 * Container hash function from
 * https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
 */
template <class T, class Container>
size_t hash_container(size_t size, const Container& container) {
  for (const auto& x : container) {
    size ^= std::hash<T>()(x) + 0x9e3779b9 + (size << 6) + (size >> 2);
  }
  return size;
}

}  // namespace

namespace std {

template <>
struct hash<tensorflow::TensorShapeDim> {
  size_t operator()(const tensorflow::TensorShapeDim& dim) const {
    return dim.size;
  }
};

template <>
struct hash<tensorflow::TensorShape> {
  size_t operator()(const tensorflow::TensorShape& shape) const {
    return hash_container<tensorflow::TensorShapeDim>(shape.dims(), shape);
  }
};

template <>
struct hash<std::vector<tensorflow::TensorShape>> {
  size_t operator()(const std::vector<tensorflow::TensorShape>& shapes) const {
    return hash_container<tensorflow::TensorShape>(shapes.size(), shapes);
  }
};

}  // namespace std

#endif  // TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_HASH_SHAPES_H_
