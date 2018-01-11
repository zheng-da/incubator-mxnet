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
 * Copyright (c) 2015 by Contributors
 * \file lrn.cc
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_lrn-inl.h"
#endif

namespace mxnet {
namespace op {

int GetNumOutputs(const LRNParam &param) {
#if MXNET_USE_MKLDNN == 1
  return SupportMKLDNNLRN(param) ? 3 : 2;
#else
  return 2;
#endif
}

static bool LRNShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_shape,
                     std::vector<TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
  const TShape &dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  out_shape->clear();
  out_shape->push_back(dshape);
  out_shape->push_back(dshape);
#if MXNET_USE_MKLDNN == 1
  // Create LRN primitive for getting the workspace size
  CHECK_EQ(dshape.ndim(), 4U);
  memory::dims src_tz_ = {static_cast<int>(dshape[0]),
                          static_cast<int>(dshape[1]),
                          static_cast<int>(dshape[2]),
                          static_cast<int>(dshape[3])};
  auto src_md = memory::desc({ src_tz_ }, memory::data_type::f32,
                              memory::format::nchw);
  const LRNParam &param = nnvm::get<LRNParam>(attrs.parsed);
  auto pdesc_fwd = GetLRNFwdDesc(param, 1, src_md);
  auto ws_size = pdesc_fwd.workspace_primitive_desc().get_size();
  TShape ws_shape(1);
  ws_shape[0] = ws_size;
  out_shape->push_back(ws_shape);
#endif
  return true;
}

static inline std::vector<std::string> ListArguments() {
  return {"data"};
}

static bool LRNType(const nnvm::NodeAttrs& attrs,
                    std::vector<int> *in_type,
                    std::vector<int> *out_type) {
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (index_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
    }
  }
#if MXNET_USE_MKLDNN == 1
  int n_out = 3;
#else
  int n_out = 2;
#endif
  out_type->clear();
  for (int i = 0; i < n_out; ++i ) out_type->push_back(dtype);
  return true;
}

struct LRNGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[0]);  // out_grad
    heads.push_back(n->inputs[lrn_enum::kData]);
    heads.emplace_back(nnvm::NodeEntry{n, lrn_enum::kTmpNorm, 0});
#if MXNET_USE_MKLDNN == 1
    heads.emplace_back(nnvm::NodeEntry{n, lrn_enum::kTmpSpace, 0});
#endif
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

inline static bool LRNForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                              const int dev_mask,
                                              DispatchMode* dispatch_mode,
                                              std::vector<int> *in_attrs,
                                              std::vector<int> *out_attrs) {
  CHECK(!in_attrs->empty());
  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  for (size_t i = 0; i < out_attrs->size(); i++) {
    (*out_attrs)[i] = kDefaultStorage;
  }
  return true;
}

inline static bool LRNBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                               const int dev_mask,
                                               DispatchMode* dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  CHECK(!in_attrs->empty());
  *dispatch_mode = DispatchMode::kFCompute;
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  for (size_t i = 0; i < out_attrs->size(); i++) 
    (*out_attrs)[i] = kDefaultStorage;
  return true;
}

void LRNComputeCPU(const nnvm::NodeAttrs &attrs,
                    const OpContext &ctx,
                    const std::vector<NDArray> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<NDArray> &outputs) {
#if MXNET_USE_MKLDNN == 1
  const LRNParam &param = nnvm::get<LRNParam>(attrs.parsed);
  if (SupportMKLDNN(inputs[lrn_enum::kData]) &&
      SupportMKLDNNLRN(param) &&
      inputs[lrn_enum::kData].dtype() == mshadow::kFloat32) {
    const NDArray *workspace = nullptr;
    if (ctx.is_train && (outputs.size() == 3U)) {
      workspace = &outputs[lrn_enum::kTmpSpace];
    }
    MKLDNNLRNCompute(ctx, param, inputs[lrn_enum::kData], req[lrn_enum::kData],
                     outputs[lrn_enum::kOut], workspace);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++)
    in_blobs[i] = inputs[i].data();

  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  LRNCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

void LRNGradComputeCPU(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
#if MXNET_USE_MKLDNN == 1
  const LRNParam &param = nnvm::get<LRNParam>(attrs.parsed);
  const NDArray &out_grad = inputs[0];
  const NDArray &in_data = inputs[1];
  const NDArray &in_grad = outputs[0];

  if (SupportMKLDNN(inputs[0]) &&
      SupportMKLDNNLRN(param) &&
      (inputs[0].dtype() == mshadow::kFloat32)) {
    const NDArray *workspace = nullptr;
    if (inputs.size() == 4U) {
      workspace = &inputs[3];
    } else {
      workspace = nullptr;
    }
    MKLDNNLRNGradCompute(ctx, param, out_grad, in_data,
                           req[0], in_grad, workspace);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++) {
    in_blobs[i] = inputs[i].data();
  }

  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();

  LRNGradCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

DMLC_REGISTER_PARAMETER(LRNParam);

NNVM_REGISTER_OP(LRN)
.describe(R"code(Applies local response normalization to the input.

The local response normalization layer performs "lateral inhibition" by normalizing
over local input regions.

If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
:math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
activity :math:`b_{x,y}^{i}` is given by the expression:

.. math::
   b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}

where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
number of kernels in the layer.

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs([](const NodeAttrs& attrs) {
  const LRNParam &param = nnvm::get<LRNParam>(attrs.parsed);
  return GetNumOutputs(param);
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
                                    [](const NodeAttrs& attrs) { return 1; })
.set_attr_parser(ParamParser<LRNParam>)
.set_attr<nnvm::FInferShape>("FInferShape", LRNShape)
.set_attr<nnvm::FInferType>("FInferType", LRNType)
.set_attr<FInferStorageType>("FInferStorageType", LRNForwardInferStorageType)
.set_attr<FCompute>("FCompute<cpu>", LRNCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", LRNComputeCPU)
.set_attr<nnvm::FGradient>("FGradient", LRNGrad{"_backward_LRN"})
.add_argument("data", "NDArray-or-Symbol", "Input data to LRN")
.add_arguments(LRNParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_LRN)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LRNParam>)
.set_attr<FInferStorageType>("FInferStorageType", LRNBackwardInferStorageType)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", LRNGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", LRNGradComputeCPU);
}  // namespace op
}  // namespace mxnet
