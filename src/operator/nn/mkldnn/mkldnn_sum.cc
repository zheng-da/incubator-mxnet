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
 * \file mkldnn_sum.cc
 * \brief
 * \author Da Zheng
*/
#include <iostream>

#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

void Sum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
         const mkldnn::memory &out) {
  std::vector<mkldnn::memory::primitive_desc> input_pds(2);
  std::vector<float> scales(2);
  std::vector<mkldnn::primitive::at> inputs;
  input_pds[0] = arr1.get_primitive_desc();
  input_pds[1] = arr2.get_primitive_desc();
  CHECK(input_pds[0] == input_pds[1]);
  scales[0] = 1;
  scales[1] = 1;
  inputs.push_back(arr1);
  inputs.push_back(arr2);
  // TODO(zhengda) I need to reorder memory here.
  mkldnn::sum::primitive_desc sum_pd(scales, input_pds);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::sum(sum_pd, inputs, out));
}

void MKLDNNSumForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs, const OpReqType &req,
                      const NDArray &out_data) {
  fprintf(stderr, "MKLDNNSum1\n");
  TmpMemMgr::Get()->Init(ctx.requested[0]);
  fprintf(stderr, "MKLDNNSum2\n");
  std::vector<mkldnn::primitive::at> in_prims;
  std::vector<mkldnn::memory::primitive_desc> in_pds(inputs.size());
  fprintf(stderr, "MKLDNNSum3\n");
  std::vector<float> scales(inputs.size());
  fprintf(stderr, "MKLDNNSum4\n");
  for (size_t i = 0; i < inputs.size(); i++) {
    fprintf(stderr, "MKLDNNSum5\n");
    auto in_mem = inputs[i].GetMKLDNNData();
    fprintf(stderr, "MKLDNNSum6\n");
    in_prims.push_back(*in_mem);
    fprintf(stderr, "MKLDNNSum7\n");
    in_pds[i] = in_mem->get_primitive_desc();
    fprintf(stderr, "MKLDNNSum8\n");
    scales[i] = 1;
  }
  fprintf(stderr, "MKLDNNSum9\n");
  mkldnn::sum::primitive_desc pdesc(scales, in_pds);
  fprintf(stderr, "MKLDNNSum10\n");

  auto out_mem = CreateMKLDNNMem(out_data, pdesc.dst_primitive_desc(), req);
  fprintf(stderr, "MKLDNNSum11\n");
  MKLDNNStream *stream = MKLDNNStream::Get();
  fprintf(stderr, "MKLDNNSum12\n");
  stream->RegisterPrim(mkldnn::sum(pdesc, in_prims, *out_mem.second));
  fprintf(stderr, "MKLDNNSum13\n");
  CommitOutput(out_data, out_mem);
  fprintf(stderr, "MKLDNNSum14\n");
  stream->Submit();
  fprintf(stderr, "MKLDNNSum15\n");
}

}  // namespace op
}  // namespace mxnet
#endif
