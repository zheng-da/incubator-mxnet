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
 * \file mkldnn_convolution_relu.cc
 * \brief
 * \author Zhang Rong A (rong.a.zhang@intel.com)
*/

#if MXNET_USE_MKLDNN == 1

#include "../convolution_relu-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {


static memory::primitive_desc GetConvParaPd(
    const mkldnn::convolution_relu_forward::primitive_desc primitive_desc,
    enum query query_pd, int index = 0) {

    memory::primitive_desc adesc;
    mkldnn_primitive_desc_t cdesc;
    const_mkldnn_primitive_desc_t const_cdesc =
        mkldnn_primitive_desc_query_pd(primitive_desc.get(), mkldnn::convert_to_c(query_pd), index);
    error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
            "could not clone a ConvPara primitive descriptor");
    adesc.reset(cdesc);
    return adesc;
}

static mkldnn::convolution_relu_forward::primitive_desc GetConvReluFwdImpl(
    const ConvolutionReluParam& param, bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  auto negative_slope = param.slope;

  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
  }

  if (param.dilate.ndim() == 0) {
    if (bias == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
      auto conv_relu_desc = convolution_relu_forward::desc(desc, negative_slope);
      return mkldnn::convolution_relu_forward::primitive_desc(conv_relu_desc, engine);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, bias_md, out_md, strides, padding, padding,
          mkldnn::padding_kind::zero);
      auto conv_relu_desc = convolution_relu_forward::desc(desc, negative_slope);
      return mkldnn::convolution_relu_forward::primitive_desc(conv_relu_desc, engine);
    }
  } else {
    mkldnn::memory::dims dilates{0, 0};
    if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    }
    if (bias == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      auto conv_relu_desc = convolution_relu_forward::desc(desc, negative_slope);
      return mkldnn::convolution_relu_forward::primitive_desc(conv_relu_desc, engine);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
                                             data_md, weight_md, bias_md, out_md, strides,
                                             dilates, padding, padding,
                                             mkldnn::padding_kind::zero);
      auto conv_relu_desc = convolution_relu_forward::desc(desc, negative_slope);
      return mkldnn::convolution_relu_forward::primitive_desc(conv_relu_desc, engine);
    }
  }
}

class MKLDNNConvReluForward {
  std::shared_ptr<mkldnn::convolution_relu_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::convolution_relu_forward::primitive_desc fwd_pd;

  MKLDNNConvReluForward(const ConvolutionReluParam& param, bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvReluFwdImpl(param, is_train, data, weights, bias, output)) {
  }

  void SetDataHandle(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output) {
    if (this->data == nullptr) {
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              GetConvParaPd(fwd_pd, src_pd), data.get_data_handle()));
    } else {
      this->data->set_data_handle(data.get_data_handle());
    }

    if (this->weight == nullptr) {
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              GetConvParaPd(fwd_pd, weights_pd), weight.get_data_handle()));
    } else {
      this->weight->set_data_handle(weight.get_data_handle());
    }

    if (this->out == nullptr) {
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              GetConvParaPd(fwd_pd, dst_pd), output.get_data_handle()));
    } else {
      this->out->set_data_handle(output.get_data_handle());
    }

    if (bias != nullptr) {
      int bias_idx = 1;
      if (this->bias == nullptr) {
        this->bias = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                GetConvParaPd(fwd_pd, weights_pd, bias_idx), bias->get_data_handle()));
      } else {
        this->bias->set_data_handle(bias->get_data_handle());
      }
      if (this->fwd == nullptr) {
        this->fwd = std::shared_ptr<mkldnn::convolution_relu_forward>(
            new mkldnn::convolution_relu_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                            mkldnn::primitive::at(*this->weight),
                                            mkldnn::primitive::at(*this->bias),
                                            *this->out));
      }
    } else {
      if (this->fwd == nullptr) {
        this->fwd = std::shared_ptr<mkldnn::convolution_relu_forward>(
          new mkldnn::convolution_relu_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                          mkldnn::primitive::at(*this->weight),
                                          *this->out));
      }
    }
  }

  const mkldnn::convolution_relu_forward &GetFwd() const {
    return *fwd;
  }
};

typedef MKLDNNParamOpSign<ConvolutionReluParam> MKLDNNConvReluSignature;

static inline MKLDNNConvReluForward &GetConvReluFwd(
    const nnvm::NodeAttrs& attrs, bool is_train,
    const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
  static thread_local
         std::unordered_map<MKLDNNConvReluSignature, MKLDNNConvReluForward, MKLDNNOpHash> fwds;
  const ConvolutionReluParam& param = nnvm::get<ConvolutionReluParam>(attrs.parsed);
  MKLDNNConvReluSignature key(param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias) {
    key.AddSign(*bias);
  }
  key.AddSign(param.slope);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConvReluForward fwd(param, is_train, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvReluSignature, MKLDNNConvReluForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNConvolutionReluForward(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<NDArray>& in_data,
    const std::vector<OpReqType>& req, const std::vector<NDArray>& out_data) {

    TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
    const ConvolutionReluParam& param = nnvm::get<ConvolutionReluParam>(attrs.parsed);
    MKLDNNConvReluForward &fwd = GetConvReluFwd(attrs,
        ctx.is_train, in_data[conv::kData], in_data[conv::kWeight],
        param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);

    auto conv_src_pd = GetConvParaPd(fwd.fwd_pd, src_pd);
    auto conv_dst_pd = GetConvParaPd(fwd.fwd_pd, dst_pd);
    auto conv_weights_pd = GetConvParaPd(fwd.fwd_pd, weights_pd);
    int bias_idx = 1;  // bias use weights_pd ,use idx 1; weights use weights_pd dflt idx 0
    auto conv_bias_pd = GetConvParaPd(fwd.fwd_pd, weights_pd, bias_idx);

    auto data_mem = in_data[conv::kData].GetMKLDNNDataReorder(conv_src_pd);
    auto engine = CpuEngine::Get()->get_engine();
    auto weight_mem = GetWeights(in_data[conv::kWeight], conv_weights_pd, param.num_group);
    auto out_mem = CreateMKLDNNMem(out_data[conv::kOut], conv_dst_pd, req[conv::kOut]);

    const mkldnn::memory *bias_mem = nullptr;
    if (!param.no_bias) {
      bias_mem = in_data[conv::kBias].GetMKLDNNDataReorder(conv_bias_pd);
    }
    fwd.SetDataHandle(*data_mem, *weight_mem, bias_mem, *out_mem.second);
    MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());

    CommitOutput(out_data[conv::kOut], out_mem);
    MKLDNNStream::Get()->Submit();
}


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1

