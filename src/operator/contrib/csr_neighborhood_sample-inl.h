/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file index_copy-inl.h
 * \brief implementation of neighbor_sample tensor operation
 */

#ifndef MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_
#define MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "../operator_common.h"

#include <vector>
#include <cstdlib>
#include <ctime>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <queue>

namespace mxnet {
namespace op {

//------------------------------------------------------------------------------
// For uniform sample op:
// 
// Input:
//
// input[0]: Graph
// input[1]: seed_vertices_0
// input[2]: seed_vertices_1
//  ... (dynamic args)
//
// args[0]: num_args
// args[1]: num_hops
// args[2]: num_neighbor 
// args[3]: max_num_vertices
//
// Output:
//
// output[0]: sampled_vertices_0
// output[1]: sampled_vertices_1
// ...
// output[0+N]: sub_graph_0
// output[1+N]: sub_graph_1
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// For non-uniform sample op:
// 
// Input:
//
// input[0]: Graph
// input[1]: probability
// input[2]: seed_vertices_0
// input[3]: seed_vertices_1
//  ... (dynamic args)
//
// args[0]: num_args
// args[1]: num_hops
// args[2]: num_neighbor 
// args[3]: max_num_vertices
//
// Output:
//
// output[0]: sampled_vertices_0
// output[1]: sampled_vertices_1
// ...
// output[0+N]: sub_graph_0
// output[1+N]: sub_graph_1
//------------------------------------------------------------------------------

// ArrayHeap is used to sample elements from a
// probability vector
class ArrayHeap {
 public:
  // ctor & dctor
  ArrayHeap(const std::vector<float>& prob) {
    this->vec_size = prob.size(); 
    this->bit_len = ceil(log2(vec_size));
    this->limit = 1 << bit_len;
    // allocate twice the size
    this->heap.resize(limit << 1, 0);
    // allocate the leaves
    for (int i = limit; i < vec_size+limit; ++i) {
      heap[i] = prob[i-limit];
    }
    // iterate up the tree (this is O(m))
    for (int i = bit_len-1; i >= 0; --i) {
      for (int j = (1 << i); j < (1 << (i + 1)); ++j) {
        heap[j] = heap[j << 1] + heap[(j << 1) + 1];
      }
    }
  }
  ~ArrayHeap() {}

  // remove term from index (this costs O(log m) steps)
  void Delete(size_t index) {
    size_t i = index + limit;
    float w = heap[i];
    for (int j = bit_len; j >= 0; --j) {
      heap[i] -= w;
      i = i >> 1;
    }
  }

  // add value w to index (this costs O(log m) steps)
  void Add(size_t index, float w) {
    size_t i = index + limit;
    for (int j = bit_len; j >= 0; --j) {
      heap[i] += w;
      i = i >> 1;
    }
  }

  // sample from arrayHeap
  size_t Sample() {
    float xi = heap[1] * (rand()%100/(float)101);
    int i = 1;
    while (i < limit) {
      i = i << 1;
      if (xi >= heap[i]) {
        xi -= heap[i];
        i += 1;
      }
    }
    return i - limit;
  }

  // Sample a vector by given the size n
  void SampleWithoutReplacement(size_t n, 
                    std::vector<size_t>& samples) {
    // sample n elements
    for (size_t i = 0; i < n; ++i) {
      samples[i] = this->Sample();
      this->Delete(samples[i]);
    }
  }

 private:
  int vec_size;  // sample size
  int bit_len;   // bit size
  int limit;
  std::vector<float> heap;
};

typedef int64_t dgl_id_t;

// For BFS traversal
struct ver_node {
  dgl_id_t vertex_id;
  int level;
}; 

struct NeighborSampleParam : public dmlc::Parameter<NeighborSampleParam> {
  int num_args;
  dgl_id_t num_hops;
  dgl_id_t num_neighbor;
  dgl_id_t max_num_vertices;
  DMLC_DECLARE_PARAMETER(NeighborSampleParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(2)
    .describe("Number of input NDArray.");
    DMLC_DECLARE_FIELD(num_hops)
      .set_default(1)
      .describe("Number of hops.");
    DMLC_DECLARE_FIELD(num_neighbor)
      .set_default(2)
      .describe("Number of neighbor.");
    DMLC_DECLARE_FIELD(max_num_vertices)
      .set_default(100)
      .describe("Max number of vertices.");
  }
};

// Uniform Storage Type
static bool CSRNeighborUniformSampleStorageType(const nnvm::NodeAttrs& attrs,
                                                const int dev_mask,
                                                DispatchMode* dispatch_mode,
                                                std::vector<int> *in_attrs,
                                                std::vector<int> *out_attrs) {
  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = params.num_args - 1;
  CHECK_EQ(out_attrs->size(), 2 * num_subgraphs);

  // input[0] is csr_graph
  CHECK_EQ(in_attrs->at(0), mxnet::kCSRStorage);
  // the rest ndarray is seed_vector
  for (size_t i = 0; i < num_subgraphs; i++)
    CHECK_EQ(in_attrs->at(1 + i), mxnet::kDefaultStorage);

  bool success = true;
  // sample_id
  for (size_t i = 0; i < num_subgraphs; i++) {
    if (!type_assign(&(*out_attrs)[i], mxnet::kDefaultStorage)) {
      success = false;
    }
  }
  // sub_graph
  for (size_t i = 0; i < num_subgraphs; i++) {
    if (!type_assign(&(*out_attrs)[i + num_subgraphs], mxnet::kCSRStorage)) {
      success = false;
    }
  }

  *dispatch_mode = DispatchMode::kFComputeEx;

  return success;
}

// Non-uniform Storage Type
static bool CSRNeighborNonUniformSampleStorageType(const nnvm::NodeAttrs& attrs,
                                                   const int dev_mask,
                                                   DispatchMode* dispatch_mode,
                                                   std::vector<int> *in_attrs,
                                                   std::vector<int> *out_attrs) {
  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = params.num_args - 2;
  CHECK_EQ(out_attrs->size(), 2 * num_subgraphs);

  // input[0] is csr_graph
  CHECK_EQ(in_attrs->at(0), mxnet::kCSRStorage);
  // input[1] is probability
  CHECK_EQ(in_attrs->at(1), mxnet::kDefaultStorage);

  // the rest ndarray is seed_vector
  for (size_t i = 0; i < num_subgraphs; i++)
    CHECK_EQ(in_attrs->at(2 + i), mxnet::kDefaultStorage);

  bool success = true;
  // sample_id
  for (size_t i = 0; i < num_subgraphs; i++) {
    if (!type_assign(&(*out_attrs)[i], mxnet::kDefaultStorage)) {
      success = false;
    }
  }
  // sub_graph
  for (size_t i = 0; i < num_subgraphs; i++) {
    if (!type_assign(&(*out_attrs)[i + num_subgraphs], mxnet::kCSRStorage)) {
      success = false;
    }
  }

  *dispatch_mode = DispatchMode::kFComputeEx;

  return success;
}

// Uniform Shape
static bool CSRNeighborUniformSampleShape(const nnvm::NodeAttrs& attrs,
                                          std::vector<TShape> *in_attrs,
                                          std::vector<TShape> *out_attrs) {
  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = params.num_args - 1;
  CHECK_EQ(out_attrs->size(), 2 * num_subgraphs);
  // input[0] is csr graph
  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  CHECK_EQ(in_attrs->at(0)[0], in_attrs->at(0)[1]);

  // the rest ndarray is seed vector
  for (size_t i = 0; i < num_subgraphs; i++) {
    CHECK_EQ(in_attrs->at(1 + i).ndim(), 1U);
  }

  // Output
  bool success = true;
  TShape out_shape(1);
  // We use the last element to store the actual 
  // number of vertices in the subgraph.
  out_shape[0] = params.max_num_vertices + 1;
  for (size_t i = 0; i < num_subgraphs; i++) {
    SHAPE_ASSIGN_CHECK(*out_attrs, i, out_shape);
    success = success && 
              out_attrs->at(i).ndim() != 0U &&
              out_attrs->at(i).Size() != 0U;
  }
  // sub_csr
  TShape out_csr_shape(2);
  out_csr_shape[0] = params.max_num_vertices;
  out_csr_shape[1] = in_attrs->at(0)[1];
  for (size_t i = 0; i < num_subgraphs; i++) {
    SHAPE_ASSIGN_CHECK(*out_attrs, i + num_subgraphs, out_csr_shape);
    success = success && 
              out_attrs->at(i + num_subgraphs).ndim() != 0U &&
              out_attrs->at(i + num_subgraphs).Size() != 0U;
  }

  return success;
}

// Non-uniform Shape
static bool CSRNeighborNonUniformSampleShape(const nnvm::NodeAttrs& attrs,
                                             std::vector<TShape> *in_attrs,
                                             std::vector<TShape> *out_attrs) {
  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = params.num_args - 2;
  CHECK_EQ(out_attrs->size(), 2 * num_subgraphs);
  // input[0] is csr graph
  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  CHECK_EQ(in_attrs->at(0)[0], in_attrs->at(0)[1]);

  // input[1] is probability
  CHECK_EQ(in_attrs->at(1).ndim(), 1U);

  // the rest ndarray is seed vector
  for (size_t i = 0; i < num_subgraphs; i++) {
    CHECK_EQ(in_attrs->at(2 + i).ndim(), 1U);
  }

  // Output
  bool success = true;
  TShape out_shape(1);
  // We use the last element to store the actual 
  // number of vertices in the subgraph.
  out_shape[0] = params.max_num_vertices + 1;
  for (size_t i = 0; i < num_subgraphs; i++) {
    SHAPE_ASSIGN_CHECK(*out_attrs, i, out_shape);
    success = success && 
              out_attrs->at(i).ndim() != 0U &&
              out_attrs->at(i).Size() != 0U;
  }
  // sub_csr
  TShape out_csr_shape(2);
  out_csr_shape[0] = params.max_num_vertices;
  out_csr_shape[1] = in_attrs->at(0)[1];
  for (size_t i = 0; i < num_subgraphs; i++) {
    SHAPE_ASSIGN_CHECK(*out_attrs, i + num_subgraphs, out_csr_shape);
    success = success && 
              out_attrs->at(i + num_subgraphs).ndim() != 0U &&
              out_attrs->at(i + num_subgraphs).Size() != 0U;
  }

  return success;
}

// Uniform Type
static bool CSRNeighborUniformSampleType(const nnvm::NodeAttrs& attrs,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = params.num_args - 1;
  CHECK_EQ(out_attrs->size(), 2 * num_subgraphs);

  bool success = true;
  for (size_t i = 0; i < num_subgraphs; i++) {
    TYPE_ASSIGN_CHECK(*out_attrs, i, in_attrs->at(1));
    TYPE_ASSIGN_CHECK(*out_attrs, i + num_subgraphs, in_attrs->at(0));
    success = success && 
               out_attrs->at(i) != -1 && 
               out_attrs->at(i + num_subgraphs) != -1;
  }

  return success;
}

// Non-uniform Type
static bool CSRNeighborNonUniformSampleType(const nnvm::NodeAttrs& attrs,
                                            std::vector<int> *in_attrs,
                                            std::vector<int> *out_attrs) {
  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = params.num_args - 2;
  CHECK_EQ(out_attrs->size(), 2 * num_subgraphs);

  bool success = true;
  for (size_t i = 0; i < num_subgraphs; i++) {
    TYPE_ASSIGN_CHECK(*out_attrs, i, in_attrs->at(2));
    TYPE_ASSIGN_CHECK(*out_attrs, i + num_subgraphs, in_attrs->at(0));
    success = success && 
               out_attrs->at(i) != -1 && 
               out_attrs->at(i + num_subgraphs) != -1;
  }

  return success;
}

// Get src vertex and edge id for a destination vertex
static void GetSrcList(const dgl_id_t* val_list,
                       const dgl_id_t* col_list,
                       const dgl_id_t* indptr,
                       const dgl_id_t dst_id,
                       std::vector<dgl_id_t>& src_list,
                       std::vector<dgl_id_t>& edge_list) {
  for (dgl_id_t i = *(indptr+dst_id); i < *(indptr+dst_id+1); ++i) {
    src_list.push_back(col_list[i]);
    edge_list.push_back(val_list[i]);
  }
}

// Uniform sample via random shuffle
static void GetUniformSampleShuffle(std::vector<dgl_id_t>& ver_list,
                                    std::vector<dgl_id_t>& edge_list,
                                    const size_t max_num_neighbor,
                                    std::vector<dgl_id_t>& out,
                                    std::vector<dgl_id_t>& out_edge) {
  CHECK_EQ(ver_list.size(), edge_list.size());
  // Copy ver_list to output
  if (ver_list.size() <= max_num_neighbor) {
    for (size_t i = 0; i < ver_list.size(); ++i) {
      out.push_back(ver_list[i]);
      out_edge.push_back(edge_list[i]);
    }
    return;
  }
  // If we just sample a small number of elements from a large neighbor list.
  std::vector<size_t> sorted_idxs(max_num_neighbor);
  if (ver_list.size() > max_num_neighbor * 10) {
    std::unordered_set<size_t> sampled_idxs;
    while (sampled_idxs.size() < max_num_neighbor) {
      // rand_num = [0, ver_list.size()-1]
      size_t rand_num = rand() % ver_list.size();
      sampled_idxs.insert(rand_num);
    }
    size_t i = 0;
    for (auto it = sampled_idxs.begin(); it != sampled_idxs.end(); it++, i++)
      sorted_idxs[i] = *it;
  } else {
    // The vertex list is relatively small. We just shuffle the list and
    // take the first few.
    std::vector<size_t> idxs(ver_list.size());
    for (size_t i = 0; i < idxs.size(); i++) idxs[i] = i;
    std::random_shuffle(idxs.begin(), idxs.end());
    for (size_t i = 0; i < max_num_neighbor; i++)
      sorted_idxs[i] = idxs[i];
  }
  std::sort(sorted_idxs.begin(), sorted_idxs.end());

  for (auto idx : sorted_idxs) {
    out.push_back(ver_list[idx]);
    out_edge.push_back(edge_list[idx]);
  }
}

// Uniform sample via re-sample
static void GetUniformSampleReplace(std::vector<dgl_id_t>& ver_list,
                                    std::vector<dgl_id_t>& edge_list,
                                    const size_t max_num_neighbor,
                                    std::vector<dgl_id_t>& out,
                                    std::vector<dgl_id_t>& out_edge) {
  CHECK_EQ(ver_list.size(), edge_list.size());
  // Copy ver_list to output
  if (ver_list.size() <= max_num_neighbor) {
    for (size_t i = 0; i < ver_list.size(); ++i) {
      out.push_back(ver_list[i]);
      out_edge.push_back(edge_list[i]);
    }
    return;
  }
  // Make sample
  std::unordered_map<size_t, bool> mp;
  size_t sample_count = 0;
  for (;;) {
    // rand_num = [0, ver_list.size()-1]
    size_t rand_num = rand() % ver_list.size(); 
    auto got = mp.find(rand_num);
    if (got != mp.end() && mp[rand_num]) {
      // re-sample
      continue;
    }
    mp[rand_num] = true;
    out.push_back(ver_list[rand_num]);
    out_edge.push_back(edge_list[rand_num]);
    sample_count++;
    if (sample_count == max_num_neighbor) {
      break;
    }
  }
}

// Non-uniform sample via ArrayHeap
static void GetNonUniformSample(const float* probability,
                                std::vector<dgl_id_t>& ver_list,
                                std::vector<dgl_id_t>& edge_list,
                                const size_t max_num_neighbor,
                                std::vector<dgl_id_t>& out,
                                std::vector<dgl_id_t>& out_edge) {
  CHECK_EQ(ver_list.size(), edge_list.size());
  // Copy ver_list to output
  if (ver_list.size() <= max_num_neighbor) {
    for (size_t i = 0; i < ver_list.size(); ++i) {
      out.push_back(ver_list[i]);
      out_edge.push_back(edge_list[i]);
    }
    return;
  }
  // Make sample
  std::vector<size_t> sp_index(max_num_neighbor);
  std::vector<float> sp_prob(ver_list.size());
  for (size_t i = 0; i < ver_list.size(); ++i) {
    sp_prob[i] = probability[ver_list[i]];
  }
  ArrayHeap arrayHeap(sp_prob);
  arrayHeap.SampleWithoutReplacement(max_num_neighbor, sp_index);
  out.resize(max_num_neighbor);
  out_edge.resize(max_num_neighbor);
  for (size_t i = 0; i < max_num_neighbor; ++i) {
    size_t idx = sp_index[i];
    out[i] = ver_list[idx];
    out_edge[i] = edge_list[idx];
  }
}

struct neigh_list {
  std::vector<dgl_id_t> neighs;
  std::vector<dgl_id_t> edges;
  neigh_list(const std::vector<dgl_id_t> &_neighs,
             const std::vector<dgl_id_t> &_edges)
    : neighs(_neighs), edges(_edges) {}
};

static void SampleSubgraph(const NDArray &csr, 
                           const NDArray &seed_arr,
                           const NDArray &sub_csr, 
                           const NDArray &sampled_ids,
                           const float* probability,
                           dgl_id_t num_hops, 
                           dgl_id_t num_neighbor,
                           dgl_id_t max_num_vertices) {
  size_t num_seeds = seed_arr.shape().Size();
  CHECK_GE(max_num_vertices, num_seeds);

  const dgl_id_t* val_list = csr.data().dptr<dgl_id_t>();
  const dgl_id_t* col_list = csr.aux_data(csr::kIdx).dptr<dgl_id_t>();
  const dgl_id_t* indptr = csr.aux_data(csr::kIndPtr).dptr<dgl_id_t>();
  const dgl_id_t* seed = seed_arr.data().dptr<dgl_id_t>();
  dgl_id_t* out = sampled_ids.data().dptr<dgl_id_t>();

  // BFS traverse the graph and sample vertices
  dgl_id_t sub_vertices_count = 0;
  std::unordered_set<dgl_id_t> sub_ver_mp;
  std::queue<ver_node> node_queue;
  // add seed vertices
  for (size_t i = 0; i < num_seeds; ++i) {
    ver_node node;
    node.vertex_id = seed[i];
    node.level = 0;
    node_queue.push(node);
    sub_ver_mp.insert(node.vertex_id);
    sub_vertices_count++;
  }

  std::vector<dgl_id_t> tmp_src_list;
  std::vector<dgl_id_t> tmp_edge_list;
  std::vector<dgl_id_t> tmp_sampled_src_list;
  std::vector<dgl_id_t> tmp_sampled_edge_list;
  std::unordered_map<dgl_id_t, neigh_list> neigh_mp;

  size_t num_edges = 0;
  while (!node_queue.empty()) {
    ver_node& cur_node = node_queue.front();
    if (cur_node.level < num_hops) {

      dgl_id_t dst_id = cur_node.vertex_id;
      tmp_src_list.clear();
      tmp_edge_list.clear();
      tmp_sampled_src_list.clear();
      tmp_sampled_edge_list.clear();

      GetSrcList(val_list, 
                 col_list, 
                 indptr, 
                 dst_id, 
                 tmp_src_list, 
                 tmp_edge_list);

      if (probability == nullptr) {  // uniform-sample
        GetUniformSampleReplace(tmp_src_list, 
                       tmp_edge_list, 
                       num_neighbor, 
                       tmp_sampled_src_list,
                       tmp_sampled_edge_list);
      } else {  // non-uniform-sample
        GetNonUniformSample(probability,
                       tmp_src_list, 
                       tmp_edge_list, 
                       num_neighbor, 
                       tmp_sampled_src_list,
                       tmp_sampled_edge_list);
      }

      neigh_mp.insert(std::pair<dgl_id_t, neigh_list>(dst_id,
        neigh_list(tmp_sampled_src_list, tmp_sampled_edge_list)));
      num_edges += tmp_sampled_src_list.size();
      
      // TODO The code doesn't limit the maximal number of vertices correctly.
      sub_vertices_count++;
      if (sub_vertices_count == max_num_vertices) {
        break;
      }

      for (size_t i = 0; i < tmp_sampled_src_list.size(); ++i) {
        auto ret = sub_ver_mp.insert(tmp_sampled_src_list[i]);
        if (ret.second) {
          sub_vertices_count++;
          ver_node new_node;
          new_node.vertex_id = tmp_sampled_src_list[i];
          new_node.level = cur_node.level + 1;
          if (new_node.level < num_hops)
            node_queue.push(new_node);
        }
      }
    }
    node_queue.pop();
  }
  // Copy sub_ver_mp to output[0]
  size_t idx = 0;
  for (auto& data: sub_ver_mp) {
    *(out+idx) = data;
    idx++;
  }
  size_t num_vertices = sub_ver_mp.size();
  std::sort(out, out + num_vertices);
  // The rest data will be set to -1
  for (dgl_id_t i = idx; i < max_num_vertices; ++i) {
    *(out+i) = -1;
  }
  // The last element stores the actual 
  // number of vertices in the subgraph.
  out[max_num_vertices] = sub_ver_mp.size();

  // Construct sub_csr_graph
  // TODO reduce the memory copy
  std::vector<dgl_id_t> sub_val;
  std::vector<dgl_id_t> sub_col_list;
  std::vector<dgl_id_t> sub_indptr(max_num_vertices+1, 0);
  sub_val.reserve(num_edges);
  sub_col_list.reserve(num_edges);

  for (size_t i = 0, index = 1; i < num_vertices; i++) {
    dgl_id_t dst_id = *(out + i);
    auto it = neigh_mp.find(dst_id);
    if (it != neigh_mp.end()) {
      const auto &edges = it->second.edges;
      const auto &neighs = it->second.neighs;
      CHECK_EQ(edges.size(), neighs.size());
      for (auto& val : edges) {
        sub_val.push_back(val);
      }
      for (auto& val : neighs) {
        sub_col_list.push_back(val);
      }
      sub_indptr[index] = sub_indptr[index-1] + edges.size();
    } else {
      sub_indptr[index] = sub_indptr[index-1];
    }
    index++;
  }

  // Copy sub_csr_graph to output[1]
  TShape shape_1(1);
  TShape shape_2(1);
  shape_1[0] = sub_val.size();
  shape_2[0] = sub_indptr.size();
  sub_csr.CheckAndAllocData(shape_1);
  sub_csr.CheckAndAllocAuxData(csr::kIdx, shape_1);
  sub_csr.CheckAndAllocAuxData(csr::kIndPtr, shape_2);

  dgl_id_t* val_list_out = sub_csr.data().dptr<dgl_id_t>();
  dgl_id_t* col_list_out = sub_csr.aux_data(1).dptr<dgl_id_t>();
  dgl_id_t* indptr_out = sub_csr.aux_data(0).dptr<dgl_id_t>();

  std::copy(sub_val.begin(), sub_val.end(), val_list_out);
  std::copy(sub_col_list.begin(), sub_col_list.end(), col_list_out);
  std::copy(sub_indptr.begin(), sub_indptr.end(), indptr_out);
}

// contrib_csr_neighbor_uniform_sample
static void CSRNeighborUniformSampleComputeExCPU(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  const NeighborSampleParam& params =
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = inputs.size() - 1;
  CHECK_EQ(outputs.size(), 2 * num_subgraphs);

  // set seed for random sampling
  srand(time(nullptr));

//#pragma omp parallel for
  for (size_t i = 0; i < num_subgraphs; i++) {
    SampleSubgraph(inputs[0], 
                   inputs[i + 1], 
                   outputs[i + num_subgraphs], 
                   outputs[i],
                   nullptr,
                   params.num_hops, 
                   params.num_neighbor, 
                   params.max_num_vertices);
  }
}

// contrib_csr_neighbor_non_uniform_sample
static void CSRNeighborNonUniformSampleComputeExCPU(const nnvm::NodeAttrs& attrs,
                                              const OpContext& ctx,
                                              const std::vector<NDArray>& inputs,
                                              const std::vector<OpReqType>& req,
                                              const std::vector<NDArray>& outputs) {
  const NeighborSampleParam& params =
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  size_t num_subgraphs = inputs.size() - 2;
  CHECK_EQ(outputs.size(), 2 * num_subgraphs);

  // set seed for random sampling
  srand(time(nullptr));
  const float* probability = inputs[1].data().dptr<float>();

//#pragma omp parallel for
  for (size_t i = 0; i < num_subgraphs; i++) {
    SampleSubgraph(inputs[0], 
                   inputs[i + 2], 
                   outputs[i + num_subgraphs], 
                   outputs[i],
                   probability,
                   params.num_hops, 
                   params.num_neighbor, 
                   params.max_num_vertices);
  }
}

}  // op
}  // mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_
