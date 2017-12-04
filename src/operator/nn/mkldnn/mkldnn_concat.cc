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
 * \file mkldnn_concat.cc
 * \brief
 * \author Wenting Jiang
*/
#include <iostream>

#include "../../concat-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void MKLDNNConcat_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data) {
  //printf("---- in MKLDNNConcat_Forward\n");
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_in_data = param.num_args;
  int concat_dim = param.dim;
  std::vector<mkldnn::memory::primitive_desc> data_md;
  std::vector<mkldnn::primitive::at> data_mem;
  for(int i =0; i < num_in_data; i++) {
      std::shared_ptr<const mkldnn::memory> tmp2 = in_data[i].GetMKLDNNData();
      auto tmp3 = tmp2->get_primitive_desc();
      data_md.push_back(tmp3);
      data_mem.push_back(*tmp2);
  }
  mkldnn::concat::primitive_desc fwd_pd(concat_dim, data_md); 

  auto engine = CpuEngine::Instance().get_engine();
  auto out_mem = CreateMKLDNNMem(out_data[concat_enum::kOut],
      fwd_pd.dst_primitive_desc(), req[concat_enum::kOut]);

  MKLDNNStream::Instance().RegisterPrim(mkldnn::concat(fwd_pd, data_mem, *out_mem.second));

  CommitOutput(out_data[concat_enum::kOut], out_mem);
  MKLDNNStream::Instance().Submit();
}

void MKLDNNConcat_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
//inputs: gz, inputs_0, inputs_1,...
//outputs.dim: inputs_0.dim, inputs_1.dim,...
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_in_data = param.num_args;
  int axis_ = param.dim;
  auto engine = CpuEngine::Instance().get_engine();
  std::shared_ptr<const mkldnn::memory>gz_mem = inputs[0].GetMKLDNNData();
  mkldnn::memory::primitive_desc gz_pd = gz_mem->get_primitive_desc(); 
    /* init the offset */
  mkldnn::memory::dims offsets = {0, 0, 0, 0};
    /*output*/
  //std::vector<mkldnn::memory> gradi_mem;
  
  for (int i = 0; i < num_in_data; i++) {
      mkldnn::memory::dims diff_src_tz = {inputs[i+1].shape()[0], inputs[i+1].shape()[1], inputs[i+1].shape()[2], inputs[i+1].shape()[3]};
      auto diff_src_mpd = inputs[i+1].GetMKLDNNData()->get_primitive_desc();
      auto gradi_mem_ = CreateMKLDNNMem(outputs[i], diff_src_mpd, req[i]);
      //gradi_mem.push_back(gradi_mem_);
      // create view from gy to gxs[i]
      std::shared_ptr<mkldnn::view::primitive_desc> view_pd;
      view_pd.reset(new mkldnn::view::primitive_desc(gz_pd, diff_src_tz, offsets));
      // create reorder primitive from gy to gxs[i]
      //std::shared_ptr<mkldnn::reorder::primitive_desc> reorder_pd;
      //reorder_pd.reset(new mkldnn::reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), diff_src_mpd));
      mkldnn::reorder::primitive_desc reorder_pd(view_pd.get()->dst_primitive_desc(), diff_src_mpd);
      //std::shared_ptr<mkldnn::reorder> reorder_prim;
      //reorder_prim.reset(new mkldnn::reorder(reorder_pd, *gz_mem, gradi_mem_));
      //std::unique_ptr<mkldnn::primitive> reorder_prim(new mkldnn::reorder(reorder_pd, *gz_mem, gradi_mem_));
      offsets[axis_] += diff_src_tz[axis_];
      MKLDNNStream::Instance().RegisterPrim(mkldnn::reorder(reorder_pd, *gz_mem, *gradi_mem_.second));//reorder_prim);
      
      CommitOutput(outputs[i], gradi_mem_);
  }
  MKLDNNStream::Instance().Submit();
}
}//op
}//mxnet
#endif
