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
#include <unordered_map>
#include <algorithm>
#include <queue>

namespace mxnet {
namespace op {

typedef int64_t dgl_id_t;

//------------------------------------------------------------------------------
// input[0]: Graph
// input[1]: Probability
// input[2]: seed_vertices
// args[0]: num_hops
// args[1]: num_neighbor 
// args[2]: max_num_vertices
//------------------------------------------------------------------------------

// For BFS traversal
struct ver_node {
  dgl_id_t vertex_id;
  int level;
}; 

// How to set the default value?
struct NeighborSampleParam : public dmlc::Parameter<NeighborSampleParam> {
  dgl_id_t num_hops, num_neighbor, max_num_vertices;
  DMLC_DECLARE_PARAMETER(NeighborSampleParam) {
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

static bool CSRNeighborSampleStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int> *in_attrs,
                                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 1);

  CHECK_EQ(in_attrs->at(0), mxnet::kCSRStorage);
  CHECK_EQ(in_attrs->at(1), mxnet::kDefaultStorage);
  CHECK_EQ(in_attrs->at(2), mxnet::kDefaultStorage);

  bool success = true;
  if (!type_assign(&(*out_attrs)[0], mxnet::kDefaultStorage)) {
  	success = false;
  }

  *dispatch_mode = DispatchMode::kFComputeEx;

  return success;
}

static bool CSRNeighborSampleShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape> *in_attrs,
                                   std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 1);

  CHECK_EQ(in_attrs->at(0).ndim(), 2U);
  CHECK_EQ(in_attrs->at(1).ndim(), 1U);
  CHECK_EQ(in_attrs->at(2).ndim(), 1U);
  // Check the graph shape
  CHECK_EQ(in_attrs->at(0)[0], in_attrs->at(0)[1]);
  // Probbality shape must be equal to the vertices length
  CHECK_EQ(in_attrs->at(0)[0], in_attrs->at(1)[0]);

  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  TShape out_shape(1);
  out_shape[0] = params.max_num_vertices;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);

  return out_attrs->at(0).ndim() != 0U &&
         out_attrs->at(0).Size() != 0U;
}

static bool CSRNeighborSampleType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 1);
  out_attrs->at(0) = in_attrs->at(0);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(2));
  return out_attrs->at(0) != -1;
}

static void GetSample(std::vector<dgl_id_t>& ver_list,
                      const float* probility,
                      const size_t max_num_neighbor,
                      std::vector<dgl_id_t>& out) {
  // Copy ver_list to output
  if (ver_list.size() <= max_num_neighbor) {
    for (size_t i = 0; i < ver_list.size(); ++i) {
      out.push_back(ver_list[i]);
    }
    return;
  }
  // Make sample
  std::unordered_map<dgl_id_t, bool> mp;
  while (out.size() < max_num_neighbor) {
    random_shuffle(ver_list.begin(), ver_list.end());
    for (size_t i = 0 ; i < ver_list.size(); ++i) {
      int rand_num = (rand() % 100) + 1; 
      float prob = probility[ver_list[i]];
      if (rand_num <= static_cast<int>(prob*100) &&
          mp[ver_list[i]] == false) {
        out.push_back(ver_list[i]);
        mp[ver_list[i]] = true;
        if (out.size() >= max_num_neighbor) {
          return;
        }
      }
    }
  }
}

static void CSRNeighborSampleComputeExCPU(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);

  const NeighborSampleParam& params = 
    nnvm::get<NeighborSampleParam>(attrs.parsed);

  // set seed for random sampling
  srand(time(nullptr));

  dgl_id_t num_hops = params.num_hops;
  dgl_id_t num_neighbor = params.num_neighbor;
  dgl_id_t max_num_vertices = params.max_num_vertices;

  size_t vertices_num = inputs[1].data().Size();
  size_t egde_num = inputs[0].data().Size();
  size_t seed_num = inputs[2].data().Size();

  CHECK_GE(max_num_vertices, seed_num);

  const dgl_id_t* val_list = inputs[0].data().dptr<dgl_id_t>();
  const dgl_id_t* col_list = inputs[0].aux_data(1).dptr<dgl_id_t>();
  const dgl_id_t* indptr = inputs[0].aux_data(0).dptr<dgl_id_t>();
  const dgl_id_t* seed = inputs[2].data().dptr<dgl_id_t>();
  const float* prob_array = inputs[1].data().dptr<float>();

  size_t row_len = inputs[0].aux_data(0).Size() - 1;
  CHECK_EQ(row_len, vertices_num);

  dgl_id_t* out = outputs[0].data().dptr<dgl_id_t>();

  // Get the mapping between edge_id and row_id
  dgl_id_t idx = 0;
  std::unordered_map<dgl_id_t, dgl_id_t> edge_mp;
  for (size_t i = 1; i < vertices_num+1; ++i) {
    size_t edge_in_a_row = indptr[i] - indptr[i-1];
    for (size_t j = 0; j < edge_in_a_row; ++j) {
      edge_mp[val_list[idx++]] = i-1;
    }
  }

  // Get the mapping between col_id (src) and row_id_list (dst)
  std::vector<std::vector<dgl_id_t> > col_row_list(vertices_num);
  for (size_t i = 0; i < egde_num; ++i) {
    dgl_id_t row_id = edge_mp[val_list[i]];
    dgl_id_t col_id = col_list[i];
    col_row_list[col_id].push_back(row_id);
  }

  // BFS traverse the graph and sample vertices
  std::vector<dgl_id_t> sub_ver_mp(vertices_num, 0);
  std::queue<ver_node> node_queue;
  dgl_id_t vertices_count = 0;
  for (size_t i = 0; i < seed_num; ++i) {
    // add seed vertices
    ver_node node;
    node.vertex_id = seed[i];
    node.level = 0;
    node_queue.push(node);
    // use 1 as flag
    sub_ver_mp[node.vertex_id] = 1;
    vertices_count++;
  }

  std::vector<dgl_id_t> sampled_vec;
  while (!node_queue.empty() && 
         vertices_count < max_num_vertices) {
    ver_node& cur_node = node_queue.front();
    node_queue.pop();
    if (cur_node.level < num_hops) {
      dgl_id_t src_id = cur_node.vertex_id;
      sampled_vec.clear();
      GetSample(col_row_list[src_id], // ver_list
                prob_array,      // probability
                num_neighbor,    // max_num_neighbor
                sampled_vec);    // output
      for (size_t i = 0; i < sampled_vec.size(); ++i) {
        if (sub_ver_mp[sampled_vec[i]] == 0) {
          vertices_count++;
          sub_ver_mp[sampled_vec[i]] = 1;
        }
        if (vertices_count >= max_num_vertices) {
          break;
        }
        ver_node new_node;
        new_node.vertex_id = sampled_vec[i];
        new_node.level = cur_node.level+1;
        node_queue.push(new_node);
      }
    }
  }

  // copy sub_ver_list to output
  idx = 0;
  for (size_t i = 0; i < sub_ver_mp.size(); ++i) {
    if (sub_ver_mp[i] != 0) {
      *(out + idx) = i;
      idx++;
    }
  }
  // The rest data is -1
  for (dgl_id_t i = idx; i < max_num_vertices; ++i) {
    *(out + i) = -1;
  }
}

}  // op
}  // mxnet

#endif  // MXNET_OPERATOR_CONTRIB_CSR_NEIGHBORHOOD_SAMPLE_INL_H_