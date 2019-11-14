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

#include "tensorflow/contrib/tensoropt/convert/convert_graph.h"

#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/tensoropt/segment/segment.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"  // NOLINT

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
#include "tensorflow/contrib/tensoropt/api/topt_lib_api.h"

namespace tensorflow {
namespace tensoropt {
namespace convert {
using ::tensorflow::str_util::Split;
using ::tensorflow::str_util::StrContains;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;
namespace {

/**
 * Returns true if the node must be converted using TensorOpt,
 * ignoring the min segment parameter.
 */
bool IsTensorOptMandatoryNode(const tensorflow::Node* node) {
  // clang-format off
  static const std::set<string> ops = {
      "Conv2D",
      "DepthwiseConv2dNative",
      "MatMul",
  };
  // clang-format on
  return ops.count(node->type_string());
}

/**
 * Returns true if the node is a "weak" node.
 * Weak nodes are nodes that do not count in the segment size as
 * even long sequences of these nodes are not worth converting on their own.
 */
bool IsTensorOptWeakNode(const tensorflow::Node* node) {
  // clang-format off
  static const std::set<string> ops = {
      "Const",
      "Identity",
      "Snapshot",
      "Reshape",
      "Squeeze",
      "Cast",
  };
  // clang-format on
  return ops.count(node->type_string());
}

/**
 * Returns ture if the node is supported by TensorOpt.
 */
bool IsTensorOptCandidate(const tensorflow::Node* node) {
  // clang-format off
  static const std::set<string> ops = {
      "Const",
      "Identity",
      "Snapshot",
      "Add",
      "BiasAdd",
      "Mul",
      "Sub",
      "Div",
      "RealDiv",
      "Minimum",
      "Maximum",
      "Relu",
      "Relu1",
      "Relu6",
      "Exp",
      "Sqrt",
      "Rsqrt",
      "Softmax",
      "FusedBatchNorm",
      "FusedBatchNormV2",
      "Conv2D",
      "DepthwiseConv2dNative",
      "MaxPool",
      "AvgPool",
      "MatMul",
      "Transpose",
      "Reshape",
      "Squeeze",
      "Slice",
      "StridedSlice",
      "ConcatV2",
      "Pack",
      "Unpack",
      "Cast",
  };
  // clang-format on
  return ops.count(node->type_string());
}

string GetCommonNameScope(const string& op_name_a, const string& op_name_b) {
  size_t last_scope_separator = 0;
  for (size_t i = 0; i < std::min(op_name_a.size(), op_name_b.size()); ++i) {
    if (op_name_a[i] != op_name_b[i]) {
      break;
    } else if (op_name_a[i] == '/') {
      last_scope_separator = i + 1;
    }
  }
  return op_name_a.substr(0, last_scope_separator);
}

tensorflow::Status ReverseTopologicalSort(
    const tensorflow::Graph& graph, const std::set<int>& subgraph_node_ids,
    std::list<tensorflow::Node*>& order) {
  std::vector<tensorflow::Node*> order_vec;
  tensorflow::GetPostOrder(graph, &order_vec);
  for (tensorflow::Node* node : order_vec) {
    if (subgraph_node_ids.count(node->id())) {
      order.push_front(node);
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status SetInputList(
    const std::vector<std::pair<int, int>>& input_inds,
    const std::vector<string>& input_names,
    const std::vector<tensorflow::DataType>& input_dtypes,
    tensorflow::NodeDefBuilder& op_builder) {
  std::vector<tensorflow::NodeDefBuilder::NodeOut> income_edges;
  VLOG(2) << "input edge size: " << input_names.size();
  for (size_t i = 0; i < input_names.size(); ++i) {
    VLOG(2) << "input edges: " << i << " " << input_names.at(i);
    int output_idx = input_inds.at(i).second;
    auto incoming_edge = tensorflow::NodeDefBuilder::NodeOut(
        input_names.at(i), output_idx, input_dtypes.at(i));
    income_edges.push_back(incoming_edge);
  }
  tensorflow::gtl::ArraySlice<tensorflow::NodeDefBuilder::NodeOut> input_list(
      income_edges);
  op_builder.Input(input_list);
  return tensorflow::Status::OK();
}

string SubgraphNameScopeGenerator(const std::list<tensorflow::Node*>* order) {
  string subgraph_name_scope;
  if (!order->empty()) {
    subgraph_name_scope = order->front()->name();
  }
  for (const tensorflow::Node* node : *order) {
    subgraph_name_scope = GetCommonNameScope(subgraph_name_scope, node->name());
  }
  return subgraph_name_scope;
}

void GetSubGraphIncomingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet& incoming_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->in_edges()) {
      if (!subgraph_node_ids.count(edge->src()->id()) &&
          !edge->src()->IsSource() && !edge->IsControlEdge()) {
        incoming_edges.insert(edge);
        VLOG(2) << "INCOMING " << edge->src()->name() << " -> " << node->name();
      }
    }
  }
}

void GetSubGraphOutgoingEdges(const tensorflow::Graph& graph,
                              const std::set<int>& subgraph_node_ids,
                              tensorflow::EdgeSet* outgoing_edges) {
  for (int node_id : subgraph_node_ids) {
    const tensorflow::Node* node = graph.FindNodeId(node_id);
    for (const tensorflow::Edge* edge : node->out_edges()) {
      if (!subgraph_node_ids.count(edge->dst()->id()) &&
          !edge->dst()->IsSink() && !edge->IsControlEdge()) {
        VLOG(2) << "OUTGOING " << node->name() << " -> " << edge->dst()->name();
        outgoing_edges->insert(edge);
      }
    }
  }
}

struct ConvertGraphParams {
  ConvertGraphParams(tensorflow::Graph& inp_graph,
                     const std::set<int>& subgraph_node_id_numbers,
                     const string& device_name, int device_id)
      : graph(inp_graph),
        subgraph_node_ids(subgraph_node_id_numbers),
        device_name_(device_name),
        device_id_(device_id) {}
  tensorflow::Graph& graph;
  const std::set<int>& subgraph_node_ids;
  string device_name_;
  int device_id_;
  std::vector<std::pair<int, int>> subgraph_inputs;
  std::vector<std::pair<int, int>> subgraph_outputs;
  std::map<std::pair<int, int>, int> subgraph_edge_to_input_map;
  std::map<std::pair<int, int>, int> subgraph_edge_to_output_map;
  tensorflow::EdgeSet subgraph_incoming_edges;
  tensorflow::EdgeSet subgraph_outgoing_edges;
  // <node_name>:<port_id>, ":0" is dropped to match protobuf
  std::vector<string> input_names;
  std::vector<tensorflow::DataType> input_dtypes;
  // <node_name>:<port_id>, ":0" is dropped to match protobuf
  std::vector<string> output_names;
  std::vector<tensorflow::DataType> output_dtypes;
};

tensorflow::Status FillSubGraphEdgeSets(ConvertGraphParams* p) {
  GetSubGraphIncomingEdges(p->graph, p->subgraph_node_ids,
                           p->subgraph_incoming_edges);

  std::set<std::pair<int, int>> unique_tensors;
  std::set<string> added_tensors;
  // Add only unique input source nodes. If output of an outside node is shared
  // between multiple nodes inside the engine, only one edge should be created
  for (const tensorflow::Edge* edge : p->subgraph_incoming_edges) {
    unique_tensors.insert({edge->src()->id(), edge->src_output()});
  }
  p->subgraph_inputs.insert(p->subgraph_inputs.begin(), unique_tensors.begin(),
                            unique_tensors.end());
  // input_names and input_dtypes must be filled in the same order that is used
  // to fill subgraph_edge_to_input_map
  for (size_t i = 0; i < p->subgraph_inputs.size(); ++i) {
    const auto& node_port_pair = p->subgraph_inputs.at(i);
    auto node_id = node_port_pair.first;
    auto port_id = node_port_pair.second;
    p->subgraph_edge_to_input_map.insert({node_port_pair, i});
    const tensorflow::Node* node = p->graph.FindNodeId(node_id);
    auto node_name = node->name();
    string input_tensor_name = node_name;
    if (port_id != 0) { // protobuf is droping ":0"
      StrAppend(&input_tensor_name, ":", port_id);
    }
    if (!added_tensors.count(input_tensor_name)) {
      added_tensors.insert(input_tensor_name);
      p->input_names.push_back(input_tensor_name);
      p->input_dtypes.push_back(node->output_type(port_id));
      VLOG(2) << "Add input edgeset: name=\"" << p->input_names.back() << "\""
              << " type=" << p->input_dtypes.back();
    }
  }

  GetSubGraphOutgoingEdges(p->graph, p->subgraph_node_ids,
                           &p->subgraph_outgoing_edges);
  unique_tensors.clear();
  added_tensors.clear();
  // Similar to above, if multiple ouside nodes are sharing the output of an
  // internal node only one output port should be created and shared between
  // outputs
  for (const tensorflow::Edge* edge : p->subgraph_outgoing_edges) {
    unique_tensors.insert({edge->src()->id(), edge->src_output()});
  }
  p->subgraph_outputs.insert(p->subgraph_outputs.begin(),
                             unique_tensors.begin(), unique_tensors.end());
  // output_names and output_dtypes must be filled in the same order that is used
  // to fill subgraph_edge_to_output_map
  for (size_t i = 0; i < p->subgraph_outputs.size(); ++i) {
    const auto& node_port_pair = p->subgraph_outputs.at(i);
    auto node_id = node_port_pair.first;
    auto port_id = node_port_pair.second;
    p->subgraph_edge_to_output_map.insert({node_port_pair, i});
    const tensorflow::Node* node = p->graph.FindNodeId(node_id);
    auto node_name = node->name();
    string output_tensor_name = node_name;
    if (port_id != 0) { // protobuf is droping ":0"
      StrAppend(&output_tensor_name, ":", port_id);
    }
    if (!added_tensors.count(output_tensor_name)) {
      added_tensors.insert(output_tensor_name);
      p->output_names.push_back(output_tensor_name);
      p->output_dtypes.push_back(node->output_type(port_id));
      VLOG(2) << "Add output edgeset: name=\"" << p->output_names.back() << "\""
              << " type=" << p->output_dtypes.back();
    }
  }

  return tensorflow::Status::OK();
}

string ConvertSubGraphToProto(const tensorflow::Graph& graph,
                              const std::list<tensorflow::Node*>& order) {
  GraphDef graph_def;
  *graph_def.mutable_versions() = graph.versions();
  *graph_def.mutable_library() = graph.flib_def().ToProto();

  graph_def.mutable_node()->Reserve(order.size());

  // Construct this outside the loop for speed.
  std::vector<const Edge*> inputs;
  for (auto node : order) {
    if (node == nullptr || !node->IsOp())
      continue;
    NodeDef* node_def = graph_def.add_node();
    *node_def = node->def();

    // Use the node's assigned device, if any, instead of the device requested
    // in the NodeDef.
    if (!node->assigned_device_name().empty()) {
      node_def->set_device(node->assigned_device_name());
    }

    // Get the inputs for this Node.  We make sure control inputs are
    // after data inputs, as required by GraphDef.
    inputs.clear();
    inputs.resize(node->num_inputs(), nullptr);
    for (const Edge* edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        inputs.push_back(edge);
      } else {
        CHECK(inputs[edge->dst_input()] == nullptr)
            << "Edge " << edge->src()->DebugString() << ":"
            << edge->dst()->DebugString() << " with dst_input "
            << edge->dst_input() << " and had pre-existing input edge "
            << inputs[edge->dst_input()]->src()->DebugString() << ":"
            << inputs[edge->dst_input()]->dst()->DebugString();

        inputs[edge->dst_input()] = edge;
      }
    }
    // Sort the control inputs for more predictable serialization.
    std::sort(inputs.begin() + node->num_inputs(), inputs.end(),
              [](const Edge* a, const Edge* b) -> bool {
                return a->src()->name() < b->src()->name();
              });
    node_def->clear_input();
    node_def->mutable_input()->Reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      const Edge* edge = inputs[i];
      if (edge == nullptr) {
        if (i < node->requested_inputs().size()) {
          node_def->add_input(node->requested_inputs()[i]);
        } else {
          node_def->add_input("");
        }
      } else {
        const Node* src = edge->src();
        if (!src->IsOp())
          continue;
        auto& src_name = src->name();
        auto src_slot = edge->src_output();
        if (src_slot == Graph::kControlSlot) {
          node_def->add_input(StrCat("^", src_name));
        } else if (src_slot == 0) {
          node_def->add_input(src_name.data(), src_name.size());
        } else {
          node_def->add_input(StrCat(src_name, ":", src_slot));
        }
      }
    }
  }
  string graph_def_str;
  graph_def.SerializeToString(&graph_def_str);
  return graph_def_str;
}

tensorflow::Status ConvertSubGraphToTensorOpt(ConvertGraphParams* params) {
  TF_RETURN_IF_ERROR(FillSubGraphEdgeSets(params));
  tensorflow::NodeDef topt_node_def;

  // Visit nodes in reverse topological order and construct the TensorOpt
  // network.
  std::list<tensorflow::Node*> order;
  TF_RETURN_IF_ERROR(
      ReverseTopologicalSort(params->graph, params->subgraph_node_ids, order));

  // Store all nodes that are used as a control edge input to the subgraph.
  // The control edges are removed from the graph now so that the subgraph
  // doesn't contain any reference to the outside.
  // Control edges from the same sources to the TOPT node are created later,
  // once the TOPT node is created.
  std::vector<tensorflow::Node*> control_edges_sources;
  for (auto node : order) {
    for (auto edge : node->in_edges()) {
      if (edge->IsControlEdge() &&
          params->subgraph_node_ids.count(edge->src()->id()) == 0) {
        control_edges_sources.push_back(edge->src());
        params->graph.RemoveControlEdge(edge);
      }
    }
  }

  static int static_id = 0;
  string subgraph_name_scope = SubgraphNameScopeGenerator(&order);
  string engine_name = StrCat(subgraph_name_scope, "my_topt_op", static_id++);

  tensorflow::NodeDefBuilder op_builder(engine_name, "TOPTEngineOp");
  TF_RETURN_IF_ERROR(SetInputList(params->subgraph_inputs, params->input_names,
                                  params->input_dtypes, op_builder));

  string graph_def_str = ConvertSubGraphToProto(params->graph, order);

  if (params->device_name_[0] != '/') {
    params->device_name_ = "/device:" + params->device_name_;
  }

  VLOG(2) << "Finished op preparation";

  tensorflow::Status status;
  status = op_builder.Attr("proto_graph", graph_def_str)
               .Attr("input_names", params->input_names)
               .Attr("output_names", params->output_names)
               .Attr("InT", params->input_dtypes)
               .Attr("OutT", params->output_dtypes)
               .Device(params->device_name_)
               .Finalize(&topt_node_def);

  VLOG(2) << status.ToString() << " finished op building for " << engine_name
          << " on device " << params->device_name_;
  TF_RETURN_IF_ERROR(status);
  tensorflow::Node* topt_node = params->graph.AddNode(topt_node_def, &status);
  TF_RETURN_IF_ERROR(status);

  // Re-map incoming control eddes to the new TOPT node.
  for (auto src_node : control_edges_sources) {
    params->graph.AddControlEdge(src_node, topt_node);
    VLOG(2) << "Wire control edge ^" << src_node->name()
            << " -> " << topt_node->name() << std::endl;
  }

  // AddNode does not wire edges.
  // Re-map incoming edges to use the new TensorOpt node instead of the original
  // subgraph
  std::set<std::pair<int, int>> unique_tensors;
  for (const tensorflow::Edge* edge : params->subgraph_incoming_edges) {
    std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    if (unique_tensors.count(old_src))
      continue;
    unique_tensors.insert(old_src);
    int new_src_output = params->subgraph_edge_to_input_map.at(old_src);
    params->graph.AddEdge(edge->src(), edge->src_output(), topt_node,
                          new_src_output);
    VLOG(1) << "Wire " << edge->src()->name() << ":" << edge->src_output()
            << " -> " << topt_node->name() << ":" << new_src_output;
    params->graph.RemoveEdge(edge);
  }
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "new edge count: " << topt_node->in_edges().size();
    for (const tensorflow::Edge* edge : topt_node->in_edges()) {
      VLOG(2) << edge->src()->name() << " port: " << edge->src_output();
    }
  }

  // Re-map outgoing edges to use the new TensorOpt node instead of the original
  // subgraph
  for (const tensorflow::Edge* edge : params->subgraph_outgoing_edges) {
    std::pair<int, int> old_src = {edge->src()->id(), edge->src_output()};
    int new_src_output = params->subgraph_edge_to_output_map.at(old_src);
    TF_RETURN_IF_ERROR(params->graph.UpdateEdge(
        topt_node, new_src_output, edge->dst(), edge->dst_input()));
    VLOG(1) << "Wire " << topt_node->name() << ":" << new_src_output << " -> "
            << edge->dst()->name() << ":" << edge->dst_input();
  }
  // Remove the original subgraph
  for (int node_id : params->subgraph_node_ids) {
    tensorflow::Node* node = params->graph.FindNodeId(node_id);
    // Don't remove the input placeholders
    if (node->type_string() == "Placeholder") {
      continue;
    }
    params->graph.RemoveNode(node);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status BuildNodeMap(
    const tensorflow::Graph& graph,
    std::unordered_map<string, tensorflow::Node*>* node_map) {
  for (auto* node : graph.op_nodes()) {
    if (!node_map->insert({node->name(), node}).second) {
      return tensorflow::errors::AlreadyExists(
          "Node name is not unique in graph: " + node->name());
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status ConvertGraphDefToTensorOpt(
    const tensorflow::GraphDef& gdef,
    const std::vector<string>& graph_output_names,
    tensorflow::GraphDef* new_graph_def, int minimum_segment_size,
    const tensorflow::grappler::Cluster* cluster) {
  // Segment the graph into subgraphs that can be converted to TensorOpt
  tensorflow::tensoropt::segment::SegmentOptions segment_options;
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));

  // Conversion of the output node is not supported yet.
  // This would require adding Identity nodes for each output to make sure
  // that the model output names are unchanged with TensorOpt
  for (auto node : graph_output_names) {
    auto splits = Split(node, ":");
    segment_options.exclude_node_list.insert(splits[0]);
  }

  segment_options.minimum_segment_size = minimum_segment_size;
  tensorflow::tensoropt::segment::SegmentNodesVector segments;
  TF_RETURN_IF_ERROR(tensoropt::segment::SegmentGraph(
      &graph, IsTensorOptCandidate, segment_options, &segments,
      IsTensorOptMandatoryNode, IsTensorOptWeakNode));
  if (segments.size() > 1) {
    VLOG(0) << "MULTIPLE TensorOpt candidate conversion: " << segments.size();
  }
  std::unordered_map<string, tensorflow::Node*> node_map;
  TF_RETURN_IF_ERROR(BuildNodeMap(graph, &node_map));
  int count = 0;
  // We create the map here since cluster may not be available in all cases.
  std::map<string, tensorflow::Device*> name_to_device_map;
  if (cluster) {
    for (const auto dm : cluster->GetDeviceSet()->devices()) {
      name_to_device_map[dm->name()] = dm;
    }
  }
  for (const auto& segment_nodes_and_device : segments) {
    const std::set<string>& subgraph_node_names =
        segment_nodes_and_device.first;

    auto target_device =
        name_to_device_map.find(segment_nodes_and_device.second);
    int device_id = 0;
    if (target_device != name_to_device_map.end())
      device_id = target_device->second->parsed_name().id;
    string device_name = segment_nodes_and_device.second;
    if (device_name.empty()) {
      device_name = StrCat("SYCL:", device_id);
    } else if (!StrContains(device_name, ":")) {
      StrAppend(&device_name, ":", device_id);
    }

    if (VLOG_IS_ON(1)) {
      std::stringstream oss;
      for (const string& node_name : subgraph_node_names) {
        tensorflow::Node* node = graph.FindNodeId(node_map.at(node_name)->id());
        oss << " " << node->type_string();
      }
      VLOG(1) << "Subgraph nodes at device " << device_name << ":" << oss.str();
    }

    std::set<int> subgraph_node_ids;
    for (const string& node_name : subgraph_node_names) {
      subgraph_node_ids.insert(node_map.at(node_name)->id());
    }
    ConvertGraphParams p(graph, subgraph_node_ids, device_name, device_id);
    tensorflow::Status status = ConvertSubGraphToTensorOpt(&p);

    if (status != tensorflow::Status::OK()) {
      // TODO(codeplay): Revert to a warning later
      LOG(FATAL) << "subgraph conversion error for subgraph_index:" << count
                 << " due to: \"" << status.ToString() << "\" SKIPPING......( "
                 << subgraph_node_names.size() << " nodes)";
    }
    count++;
  }
  graph.ToGraphDef(new_graph_def);
  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace tensoropt
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
