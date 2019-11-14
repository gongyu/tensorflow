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

#include "tensorflow/contrib/tensoropt/segment/segment.h"

#include <set>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/tensoropt/segment/union_find.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensoropt {
namespace segment {
using ::tensorflow::strings::StrAppend;
using ::tensorflow::str_util::StrContains;
using ::tensorflow::str_util::Uppercase;
// A simple graph representation to mirror tensorflow::Graph. This structure
// helps saving memory since segmenter modifies the graph in place, preventing
// the need to create a copy of the graph. It is composed of edges and nodes.
// Nodes keep pointers to original TF nodes.
class SimpleNode;
class SimpleGraph;
class SimpleEdge {
 public:
  SimpleEdge(int id, SimpleNode* src, int src_port, SimpleNode* dst,
             int dst_port, bool is_control = false)
      : id_(id),
        src_(src),
        src_port_(src_port),
        dst_(dst),
        dst_port_(dst_port),
        control_(is_control) {}
  ~SimpleEdge() {}

  SimpleNode* src() const { return src_; }
  SimpleNode* dst() const { return dst_; }
  int src_output() const { return src_port_; }
  int dst_input() const { return dst_port_; }
  int id() const { return id_; }
  bool IsControlEdge() const { return control_; }

 private:
  int id_;
  SimpleNode* src_;
  int src_port_;
  SimpleNode* dst_;
  int dst_port_;
  bool control_;
};

class SimpleNode {
 public:
  SimpleNode(const tensorflow::Node* node, const int id);

  const std::vector<SimpleEdge*>& in_edges() const { return in_edges_; }
  const std::vector<SimpleEdge*>& out_edges() const { return out_edges_; }
  std::vector<SimpleNode*> in_nodes() const {
    std::vector<SimpleNode*> res;
    res.reserve(in_edges_.size());
    for (const auto e : in_edges_) {
      if (e)
        res.push_back(e->src());
    }
    return res;
  }
  const string& name() const { return node_->name(); }
  const tensorflow::Node* tf_node() const { return node_; }
  int id() const { return id_; }

 private:
  const tensorflow::Node* node_;
  std::vector<SimpleEdge*> in_edges_;
  std::vector<SimpleEdge*> out_edges_;
  int id_;

  friend class SimpleGraph;
};

class SimpleGraph {
 public:
  explicit SimpleGraph(const tensorflow::Graph* g);
  ~SimpleGraph();

  void AddControlEdge(SimpleNode* src, SimpleNode* dst);
  void AddEdge(SimpleNode* src, int out_port, SimpleNode* dst, int in_port);
  void RemoveEdge(const SimpleEdge*);
  SimpleNode* FindNodeId(int node_id) {
    if (node_id < 0 || node_id > static_cast<int>(nodes_.size())) {
      return nullptr;
    }
    return nodes_[node_id];
  }
  int num_node_ids() const { return nodes_.size(); }
  const SimpleNode* source_node() const {
    return nodes_[tensorflow::Graph::kSourceId];
  }
  const SimpleNode* sink_node() const {
    return nodes_[tensorflow::Graph::kSinkId];
  }

 private:
  const tensorflow::Graph* g_;
  std::vector<SimpleNode*> nodes_;
  std::vector<SimpleEdge*> edges_;
  // free_edge_ids_ and free_node_ids_ contain freed indices.
  std::set<int> free_edge_ids_;
  std::set<int> free_node_ids_;
};

SimpleNode::SimpleNode(const tensorflow::Node* node, const int id)
    : node_(node), id_(id) {
  if (node_) {
    in_edges_.reserve(node_->in_edges().size());
    out_edges_.reserve(node_->out_edges().size());
  }
}

SimpleGraph::SimpleGraph(const tensorflow::Graph* g) : g_(g) {
  int n_nodes = g_->num_node_ids();
  nodes_.resize(n_nodes, nullptr);
  nodes_[g->kSourceId] = new SimpleNode(g->source_node(), g->kSourceId);
  nodes_[g->kSinkId] = new SimpleNode(g->sink_node(), g->kSinkId);
  int n_edges = g->num_edge_ids();
  edges_.resize(n_edges, nullptr);
  for (int i = 2; i < n_nodes; i++) {
    const auto n = g->FindNodeId(i);
    if (n) {
      nodes_[i] = new SimpleNode(n, i);
    } else {
      free_node_ids_.insert(i);
    }
  }
  for (int i = 0; i < n_edges; i++) {
    const auto e = g->FindEdgeId(i);
    if (e) {
      const auto tfsrc = e->src();
      const auto tfdst = e->dst();
      bool is_control = e->IsControlEdge();
      auto src = nodes_[tfsrc->id()];
      auto dst = nodes_[tfdst->id()];
      auto edge = new SimpleEdge(i, src, e->src_output(), dst, e->dst_input(),
                                 is_control);
      edges_[i] = edge;
      src->out_edges_.push_back(edge);
      dst->in_edges_.push_back(edge);
    } else {
      free_edge_ids_.insert(i);
    }
  }
}

void SimpleGraph::AddEdge(SimpleNode* src, int out_port, SimpleNode* dst,
                          int in_port) {
  int i = edges_.size();
  if (!free_edge_ids_.empty()) {
    auto it = free_edge_ids_.begin();
    i = *it;
    free_edge_ids_.erase(it);
  } else {
    edges_.push_back(nullptr);
  }
  bool is_control = (out_port == tensorflow::Graph::kControlSlot);
  is_control |= (in_port == tensorflow::Graph::kControlSlot);
  auto edge = new SimpleEdge(i, src, out_port, dst, in_port, is_control);
  edges_[i] = edge;
  src->out_edges_.push_back(edge);
  dst->in_edges_.push_back(edge);
}

void SimpleGraph::AddControlEdge(SimpleNode* src, SimpleNode* dst) {
  AddEdge(src, tensorflow::Graph::kControlSlot, dst,
          tensorflow::Graph::kControlSlot);
}

void SimpleGraph::RemoveEdge(const SimpleEdge* edge) {
  auto src = edge->src();
  auto dst = edge->dst();
  for (auto it = src->out_edges_.begin(); it != src->out_edges_.end(); ++it) {
    if (*it == edge) {
      src->out_edges_.erase(it);
      break;
    }
  }
  for (auto it = dst->in_edges_.begin(); it != dst->in_edges_.end(); ++it) {
    if (*it == edge) {
      dst->in_edges_.erase(it);
      break;
    }
  }
}

SimpleGraph::~SimpleGraph() {
  for (auto x : nodes_)
    delete x;
  for (auto x : edges_)
    delete x;
}

namespace {

bool CheckCycles(const std::unique_ptr<SimpleGraph>& g, const SimpleNode* src,
                 const std::vector<SimpleNode*>& start) {
  // copied from TF ReverseDFS.
  struct Work {
    SimpleNode* node;
    bool leave;  // Are we entering or leaving n?
  };

  std::vector<Work> stack(start.size());
  for (std::size_t i = 0; i < start.size(); ++i) {
    stack[i] = Work{start[i], false};
  }

  std::vector<bool> visited(g->num_node_ids(), false);
  while (!stack.empty()) {
    Work w = stack.back();
    stack.pop_back();

    auto n = w.node;
    if (w.leave) {
      if (n == src) {
        return true;
      }
      continue;
    }

    if (visited[n->id()])
      continue;
    visited[n->id()] = true;
    // Arrange to call leave(n) when all done with descendants.
    stack.push_back(Work{n, true});

    auto nodes = n->in_nodes();
    for (const auto node : nodes) {
      if (!visited[node->id()]) {
        stack.push_back(Work{node, false});
      }
    }
  }
  return false;
}

bool CanContractEdge(const SimpleEdge* edge,
                     const std::unique_ptr<SimpleGraph>& graph) {
  const auto src = edge->src();
  const auto dst = edge->dst();

  // Can't contract edge if doing so would cause a cycle in the
  // graph. So, if there is a directed path from 'src' to 'dst', other
  // than 'edge' (or any other direct edge from 'src' to 'dst'), then
  // combining 'src' and 'dst' will cause a cycle along that path.
  //
  // In practice, to avoid modifying the graph and to take advantage
  // of existing graph functions, we perform an equivalent.
  //   1. Get all nodes incoming to 'dst', excluding 'src'
  //   2. Reverse DFS from those nodes
  //   3. If reverse DFS reaches 'src' then we have a cycle
  std::vector<SimpleNode*> dfs_start_nodes;
  for (SimpleNode* node : dst->in_nodes()) {
    if (node != src) {
      dfs_start_nodes.push_back(node);
    }
  }

  bool is_cycle = CheckCycles(graph, src, dfs_start_nodes);
  return !is_cycle;
}

bool IsInputConst(const tensorflow::Node* node, int idx) {
  tensorflow::Node* input_node;
  TF_CHECK_OK(node->input_node(idx, &input_node));
  return input_node->type_string() == "Const";
}
// Return true if all inputs are constants
// start_idx is inclusive, end_idx is exclusive
bool AreInputsConst(const tensorflow::Node* node, int start_idx, int end_idx) {
  tensorflow::Node* input_node;
  for (int i = start_idx; i < end_idx; ++i) {
    if (!IsInputConst(node, i)) {
      return false;
    }
  }
  return true;
}
}  // namespace

void ContractEdge(SimpleEdge* edge, SimpleGraph* graph,
                  std::vector<const SimpleEdge*>* remove_edges) {
  // Transfer all inputs and outputs of 'dst' to 'src' except edges
  // connecting the two.
  auto src = edge->src();
  auto dst = edge->dst();

  // We can use '0' for input/output index because we don't need them
  // to be accurate for the way we are using the graph.
  std::vector<const SimpleEdge*> in_edges(dst->in_edges().begin(),
                                          dst->in_edges().end());
  for (const SimpleEdge* in_edge : in_edges) {
    if (in_edge->IsControlEdge()) {
      if (in_edge->src() != src) {
        SimpleEdge* e = const_cast<SimpleEdge*>(in_edge);
        graph->AddControlEdge(e->src(), src);
      }
    } else {
      if (in_edge->src() != src) {
        SimpleEdge* e = const_cast<SimpleEdge*>(in_edge);
        if (e->src() == graph->source_node()) {
          graph->AddEdge(e->src(), e->src_output(), src,
                         tensorflow::Graph::kControlSlot);
        } else {
          graph->AddEdge(e->src(), e->src_output(), src, 0 /* input index */);
        }
      }
    }
  }

  std::vector<const SimpleEdge*> out_edges(dst->out_edges().begin(),
                                           dst->out_edges().end());
  for (const SimpleEdge* out_edge : out_edges) {
    if (out_edge->IsControlEdge()) {
      SimpleEdge* e = const_cast<SimpleEdge*>(out_edge);
      graph->AddControlEdge(src, e->dst());
    } else {
      SimpleEdge* e = const_cast<SimpleEdge*>(out_edge);
      if (e->dst() == graph->sink_node()) {
        VLOG(1) << " edge to sink node " << src->name() << " -> "
                << e->dst()->name();
        graph->AddEdge(src, tensorflow::Graph::kControlSlot, e->dst(),
                       e->dst_input());
      } else {
        graph->AddEdge(src, 0 /* output index */, e->dst(), e->dst_input());
      }
    }
  }

  // Return the edges that must be removed to disconnect 'dst' from
  // the graph. We don't actually remove 'dst' since the caller holds
  // references to all the nodes.
  for (const auto& in_edge : dst->in_edges()) {
    remove_edges->push_back(in_edge);
  }
  for (const auto& out_edge : dst->out_edges()) {
    remove_edges->push_back(out_edge);
  }
}

tensorflow::Status SegmentGraph(const tensorflow::GraphDef& gdef,
                                const NodeTagFunction& is_candidate_fn,
                                const SegmentOptions& options,
                                SegmentNodesVector* segments,
                                const NodeTagFunction& is_mandatory_fn,
                                const NodeTagFunction& is_weak_fn) {
  // Create a Graph representation of the GraphDef.
  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             gdef.library());
  tensorflow::Graph graph(flib);
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), gdef, &graph));
  return SegmentGraph(&graph, is_candidate_fn, options, segments,
                      is_mandatory_fn, is_weak_fn);
}

tensorflow::Status SegmentGraph(tensorflow::Graph* tf_graph,
                                const NodeTagFunction& is_candidate_fn,
                                const SegmentOptions& options,
                                SegmentNodesVector* segments,
                                const NodeTagFunction& is_mandatory_fn,
                                const NodeTagFunction& is_weak_fn) {
  auto graph = std::unique_ptr<SimpleGraph>(new SimpleGraph(tf_graph));
  // Use a union-find to collect the nodes that belong to the same
  // segment. A node value of nullptr indicates that the node is not a candidate
  // for TensorOpt.
  // int is the number of nodes that are not weak, it is UINT_MAX if the segment
  // contains a mandatory node.
  std::vector<std::pair<UnionFind<SimpleNode*>, unsigned>> node_segments;
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    SimpleNode* node = graph->FindNodeId(i);
    auto tf_node = node->tf_node();
    const auto& node_type = tf_node->type_string();
    VLOG(2) << "TensorOpt candidate i=" << i << " node=" << node
            << " name=" << node->name() << " type=" << node_type;
    bool is_supported = options.exclude_node_list.count(node->name()) == 0 &&
                        is_candidate_fn(tf_node);
    if (!is_supported) {
      VLOG(2) << "Node is not a candidate.";
      node_segments.emplace_back(nullptr, 0);
      continue;
    }

    // Check if the device is supported
    string node_device = tf_node->has_assigned_device_name() ?
      tf_node->assigned_device_name() : tf_node->requested_device();
    is_supported = node_device.empty() || StrContains(Uppercase(node_device), "SYCL");
    if (!is_supported) {
      VLOG(2) << "Node's device is not supported.";
      node_segments.emplace_back(nullptr, 0);
      continue;
    }

    // Check if the type is supported
    if (node_type == "ConcatV2") {
      is_supported = tf_node->input_type(tf_node->num_inputs() - 1) == DT_INT32;
    } else if (node_type == "Transpose") {
      is_supported = tf_node->input_type(1) == DT_INT32;
    } else if (node_type != "Cast" && node_type != "Const" && \
               node_type != "Identity" && node_type != "Snapshot" && \
               node_type != "Reshape" && node_type != "Squeeze" && \
               node_type != "Pack" && node_type == "Unpack" && \
               node_type != "Slice" && node_type != "StridedSlice") {
      // Most compute operations require floating type so far
      for (auto type : tf_node->input_types()) {
        if (type != DT_FLOAT) {
          is_supported = false;
          break;
        }
      }
      for (auto type : tf_node->output_types()) {
        if (type != DT_FLOAT) {
          is_supported = false;
          break;
        }
      }
    }
    if (!is_supported) {
      VLOG(2) << "Node is using unsupported types.";
      node_segments.emplace_back(nullptr, 0);
      continue;
    }

    // Check if node's inputs are the ones supported
    if (node_type == "ConcatV2") {
      is_supported = IsInputConst(tf_node, tf_node->num_inputs() - 1);
    } else if (node_type == "FusedBatchNorm" ||
               node_type == "FusedBatchNormV2") {
      is_supported = AreInputsConst(tf_node, 1, 5);
    } else if (node_type == "Reshape") {
      is_supported = IsInputConst(tf_node, 1);
    } else if (node_type == "Slice") {
      is_supported = AreInputsConst(tf_node, 1, 3);
    } else if (node_type == "StridedSlice") {
      is_supported = AreInputsConst(tf_node, 1, 4);
    } else if (node_type == "Transpose") {
      is_supported = IsInputConst(tf_node, 1);
    }
    if (!is_supported) {
      VLOG(2) << "Node's inputs type are not supported.";
      node_segments.emplace_back(nullptr, 0);
      continue;
    }

    // Node is supported, compute its cost and add it as a segment
    unsigned cost = 1;
    if (is_mandatory_fn(tf_node)) {
      cost = UINT_MAX;
    } else if (is_weak_fn(tf_node)) {
      cost = 0;
    }
    node_segments.emplace_back(node, cost);
    VLOG(2) << "Supported, cost=" << cost;
  }

  // The segmentation algorithm below visits nodes in reverse
  // topological order and attempts to merge nodes along output
  // edges. That means that subgraphs grow from the output-side of the
  // network towards the inputs. In general this is not guaranteed to
  // produce a globally optimal segmentation. In the future if we have
  // a measure of how beneficial it is to include a given node in a
  // TensorOpt subgraph then we can revisit this algorithm to take advantage
  // of that information.
  std::vector<tensorflow::Node*> tforder;
  tensorflow::GetPostOrder(*tf_graph, &tforder);
  // use postorder implementation from tensorflow and construct mirror in
  // internal format
  std::vector<SimpleNode*> order;
  order.reserve(tforder.size());
  for (const auto tfnode : tforder) {
    order.push_back(graph->FindNodeId(tfnode->id()));
  }
  for (const SimpleNode* node : order) {
    // All output nodes of 'node' have been visited...
    VLOG(2) << "Trying node " << node->name() << " id=" << node->id();

    // 'node' must be a TensorOpt candidate...
    if (node_segments[node->id()].first.Value() == nullptr) {
      VLOG(2) << "... not a TensorOpt candidate";
      continue;
    }

    // Contract output edges to combine 'node' with output
    // nodes. Iterate since combining two nodes may unblock other
    // combining.
    while (true) {
      std::set<const SimpleEdge*> contract_edges;
      for (const SimpleEdge* out_edge : node->out_edges()) {
        VLOG(2) << "... out node " << out_edge->dst()->name() << " ( "
                << out_edge->dst()->id() << " <- " << node->id() << " )";
        if (out_edge->IsControlEdge()) {
          VLOG(2) << "... ... Control Edge, Skipping";
          continue;
        }
        // Out node must be TensorOpt candidate...
        auto out_node = node_segments[out_edge->dst()->id()].first.Value();
        if (out_node == nullptr) {
          VLOG(2) << "... ... not a TensorOpt candidate";
          continue;
        }

        if (CanContractEdge(out_edge, graph)) {
          VLOG(2) << "... ... can contract";
          contract_edges.insert(out_edge);
        } else {
          VLOG(2) << "... ... cannot contract, would form cycle";
        }
      }

      if (contract_edges.empty()) {
        break;
      }

      // Contract edges and collect the adjacent nodes into the same
      // segment/subgraph.
      while (!contract_edges.empty()) {
        const SimpleEdge* contract_edge = *contract_edges.begin();
        const SimpleNode* src = contract_edge->src();
        const SimpleNode* dst = contract_edge->dst();

        VLOG(2) << "Merge " << src->name() << " <- " << dst->name() << " ("
                << src->id() << " <- " << dst->id() << ")";
        auto& segment_src = node_segments[src->id()];
        auto& segment_dst = node_segments[dst->id()];
        segment_src.first.Merge(&segment_dst.first);

        // Contracting the edge leaves disconnected graph edges.
        // Remove these from the graph and from 'contract_edges' so we
        // don't visit them again.
        SimpleEdge* e = const_cast<SimpleEdge*>(contract_edge);
        std::vector<const SimpleEdge*> remove_edges;
        ContractEdge(e, graph.get(), &remove_edges);

        for (const SimpleEdge* r : remove_edges) {
          contract_edges.erase(r);
          graph->RemoveEdge(r);
        }
      }
    }
  }

  // Collect the segments/subgraphs. Each subgraph is represented by a
  // set of the names of the nodes in that subgraph.
  std::unordered_map<string, std::set<string>> sg_map;
  std::unordered_map<string, std::set<string>> device_maps;
  std::unordered_map<string, unsigned> cost_map;
  for (auto& pair : node_segments) {
    auto& u = pair.first;
    auto node_cost = pair.second;
    if ((u.Value() != nullptr) && (u.ParentValue() != nullptr)) {
      auto& map_set = sg_map[u.ParentValue()->name()];
      map_set.insert(u.Value()->name());
      auto& old_cost = cost_map[u.ParentValue()->name()];
      auto new_cost = old_cost + node_cost;
      if (new_cost < old_cost || new_cost < node_cost)  // overflow
        new_cost = UINT_MAX;
      old_cost = new_cost;
      auto tf_node = u.Value()->tf_node();
      // has_assigned_device_name() is expected to return true
      // when called from optimization pass. However, since graph
      // is converted back and forth between graph and graphdef,
      // assigned devices demoted to requested devices. If the graph
      // is passed directly to this module, assigned devices will be set.
      if (tf_node->has_assigned_device_name()) {
        device_maps[u.ParentValue()->name()].insert(
            tf_node->assigned_device_name());
      } else if (!tf_node->requested_device().empty()) {
        device_maps[u.ParentValue()->name()].insert(
            tf_node->requested_device());
      } else {
        VLOG(1) << "Node " << tf_node->name()
                << " has no device assigned requested device is: "
                << tf_node->requested_device();
      }
    }
  }

  // Convert the segments into the expected return format
  for (const auto& itr : sg_map) {
    const auto& segment_node_names = itr.second;
    unsigned cost = 0;
    auto cost_itr = cost_map.find(itr.first);
    if (cost_itr == cost_map.end()) {
      VLOG(1) << "No cost assigned to segment " << segments->size();
    } else {
      cost = cost_itr->second;
    }
    if (VLOG_IS_ON(1)) {
      string s;
      for (const auto& name : segment_node_names) {
        s += " " + name;
      }
      VLOG(1) << "Segment " << segments->size() << "(cost=" << cost
              << "):" << s;
    }

    // Don't use small segments.
    if (cost < static_cast<unsigned>(options.minimum_segment_size)) {
      VLOG(1) << "Segment " << segments->size() << " has only a cost of "
              << cost << ", dropping";
      continue;
    }
    const auto& dev_itr = device_maps.find(itr.first);
    if (dev_itr == device_maps.end() || dev_itr->second.empty()) {
      VLOG(1) << "No device assigned to segment " << segments->size();
      segments->emplace_back(std::make_pair(segment_node_names, string()));
    } else if (dev_itr->second.size() > 1) {
      string s("Segment ");
      StrAppend(&s, segments->size(), " has multiple devices attached: ");
      for (const auto& dev : dev_itr->second) {
        StrAppend(&s, dev, ", ");
      }
      LOG(WARNING) << s << " choosing " << *(dev_itr->second.begin());
      segments->emplace_back(
          std::make_pair(segment_node_names, *(dev_itr->second.begin())));
    } else {
      segments->emplace_back(
          std::make_pair(segment_node_names, *(dev_itr->second.begin())));
    }
  }
  if (VLOG_IS_ON(1)) {
    for (const auto& d : device_maps) {
      string s("Segment ");
      StrAppend(&s, ": '", d.first, "' ");
      for (const auto& dd : d.second) {
        StrAppend(&s, dd, ", ");
      }
      VLOG(1) << "Devices " << s;
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace segment
}  // namespace tensoropt
}  // namespace tensorflow
