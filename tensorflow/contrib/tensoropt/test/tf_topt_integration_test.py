# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Copyright (C) Codeplay Software Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to test TF-TensorOpt integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from tensorflow.contrib import tensoropt as topt
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.python.framework import constant_op as cop
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops as aops
from tensorflow.python.ops import nn as nn
from tensorflow.python.ops import nn_ops as nn_ops
from tensorflow.python.platform import googletest


class IntegrationTest(test_util.TensorFlowTestCase):
  """Class to test Tensorflow-TensorOpt integration."""

  def setUp(self):
    """Setup method."""
    super(IntegrationTest, self).setUp()
    warnings.simplefilter("always")
    inp_dims = (100, 24, 24, 2)
    self._input = np.random.random_sample(inp_dims)
    self._original_graph = self.get_simple_graph_def()
    self._gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    self._config = cpb2.ConfigProto(gpu_options=self._gpu_options)
    self._reference = self.run_graph(self._original_graph, self._input)

  def get_simple_graph_def(self):
    """Create a simple graph and return its graph_def."""
    g = ops.Graph()
    with g.as_default():
      a = aops.placeholder(
          dtype=dtypes.float32, shape=(None, 24, 24, 2), name="input")
      e = cop.constant(
          [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
          name="weights",
          dtype=dtypes.float32)
      conv = nn.conv2d(
          input=a, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
      b = cop.constant(
          [4., 1.5, 2., 3., 5., 7.], name="bias", dtype=dtypes.float32)
      t = nn.bias_add(conv, b, name="biasAdd")
      relu = nn.relu(t, "relu")
      idty = aops.identity(relu, "ID")
      v = nn_ops.max_pool(
          idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
      aops.squeeze(v, name="output")
    return g.as_graph_def()

  def run_graph(self, gdef, dumm_inp):
    """Run given graphdef once."""
    ops.reset_default_graph()
    g = ops.Graph()
    with g.as_default():
      inp, out = importer.import_graph_def(
          graph_def=gdef, return_elements=["input", "output"])
      inp = inp.outputs[0]
      out = out.outputs[0]
    with self.test_session(
        graph=g, config=self._config, use_gpu=True, force_gpu=True) as sess:
      val = sess.run(out, {inp: dumm_inp})
    return val

  def get_topt_graph(self, mode):
    """Return topt converted graph."""
    if mode in ["FP32", "FP16", "INT8"]:
      return topt.create_inference_graph(
          input_graph_def=self._original_graph,
          outputs=["output"],
          max_batch_size=self._input.shape[0],
          max_workspace_size_bytes=1 << 25,
          precision_mode=mode,  # TensorOpt Engine precision "FP32","FP16" or "INT8"
          minimum_segment_size=2  # minimum number of nodes in an engine
      )
    return None

  def testFP32(self):
    """Test FP32 conversion. Results should be identical to native case."""
    topt_graph = self.get_topt_graph("FP32")
    result = self.run_graph(topt_graph, self._input)
    self.assertNDArrayNear(self._reference, result, 1e-3)
    result1 = self.run_graph(topt_graph, self._input)
    self.assertNDArrayNear(result1, result, 1e-3)


if __name__ == "__main__":
  googletest.main()
