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
 * \file mkldnn_softmax.cc
 * \brief
 * \author Da Zheng
 * \author Wenting Jiang (wenting.jiang@intel.com)        
*/

#if MXNET_USE_MKLDNN == 1
#include "../softmax-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNSoftmaxFwd {
  std::shared_ptr<mkldnn::softmax_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> out;

 public:
  MKLDNNSoftmaxFwd(const SoftmaxParam& param,
                   bool is_train,
                   const NDArray &in_data,
                   const NDArray &out_data,
                   const OpReqType &req):
                   fwd(nullptr), data(nullptr), out(nullptr) {
                   _Init(param, is_train, in_data, out_data, req);
  }
  ~MKLDNNSoftmaxFwd() {}
  void SetDataHandle(const NDArray &in_data,
                     const NDArray &out_data) {
    this->data->set_data_handle(in_data.GetMKLDNNData()->get_data_handle());
    auto out_mem = const_cast<NDArray&>(out_data).CreateMKLDNNData(this->out->get_primitive_desc());
    this->out->set_data_handle(out_mem->get_data_handle());
  }
  void Execute() {
    MKLDNNStream *stream = MKLDNNStream::Get();
    stream->RegisterPrim(*fwd);
    stream->Submit();
  }

 private:
  void _Init(const SoftmaxParam& param,
             bool is_train,
             const NDArray &in_data,
             const NDArray &out_data,
             const OpReqType &req) {
    // mkldnn::softmax_forward::primitive_desc
    auto input_mem = in_data.GetMKLDNNData();
    mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
    mkldnn::memory::desc data_md = data_mpd.desc();
    auto cpu_engine = CpuEngine::Get()->get_engine();
    auto prop = is_train
                ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
    mkldnn::softmax_forward::desc desc = mkldnn::softmax_forward::desc(prop, data_md, param.axis);
    mkldnn::softmax_forward::primitive_desc pdesc(desc, cpu_engine);
    // mkldnn::memory
    this->data.reset(new mkldnn::memory(data_mpd));
    this->out.reset(new mkldnn::memory(data_mpd));
    // mkldnn::softmax_forward
    this->fwd = std::shared_ptr<mkldnn::softmax_forward>(
                new mkldnn::softmax_forward(pdesc, *(this->data), *(this->out)));
  }
};

typedef MKLDNNParamOpSign<SoftmaxParam> MKLDNNSmSignature;

static MKLDNNSoftmaxFwd &GetSoftmaxFwd(const SoftmaxParam& param,
                                       const OpContext &ctx,
                                       const NDArray &in_data,
                                       const NDArray &out_data,
                                       const OpReqType &req) {
  static thread_local std::unordered_map<MKLDNNSmSignature, MKLDNNSoftmaxFwd, MKLDNNOpHash> fwds;
  MKLDNNSmSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(param.axis);
  // TODO(huang jin): add NDArray as key uniformly
  key.AddSign(*(in_data.GetMKLDNNData()));

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSoftmaxFwd fwd(param, ctx.is_train, in_data, out_data, req);
    auto ins_ret = fwds.insert(std::pair<MKLDNNSmSignature, MKLDNNSoftmaxFwd>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNSoftmaxCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const NDArray &in_data,
                          const OpReqType &req,
                          const NDArray &out_data) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  MKLDNNSoftmaxFwd &fwd = GetSoftmaxFwd(param, ctx, in_data, out_data, req);
  fwd.SetDataHandle(in_data, out_data);
  fwd.Execute();
}

}   // namespace op
}   // namespace mxnet
#endif
