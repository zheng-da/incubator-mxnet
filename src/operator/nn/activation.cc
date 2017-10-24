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
 * \file activation.cc
 * \brief activation op
 * \author Bing Xu, Da Zheng
*/
#include "./activation-inl.h"
#include "../tensor/elemwise_unary_op.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_relu-inl.h"
#endif  // MXNET_USE_MKLDNN

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ActivationParam);

// This will determine the order of the inputs for backward computation.
struct ActivationGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.emplace_back(nnvm::NodeEntry{n, activation::kOut, 0});
#if MXNET_USE_CUDNN == 1
    heads.push_back(n->inputs[activation::kData]);
#endif
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

static void ActivationComputeEx_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
#if MXNET_USE_MKLDNN == 1
  if (param.act_type == activation::kReLU) {
    switch (inputs[0].dtype()) {
      case mshadow::kFloat32:
        MKLDNNRelu_Forward<float>(ctx, inputs[0], req[0], outputs[0]);
        return;
      default:
        break;
    }
  }
#endif
  _ActivationCompute<cpu>(param, ctx, inputs[0].data(), req[0],
      outputs[0].data());
}

void ActivationGradComputeEx_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
#if MXNET_USE_CUDNN == 1
  CHECK_EQ(inputs.size(), 3U);
#else
  CHECK_EQ(inputs.size(), 2U);
#endif
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
#if MXNET_USE_MKLDNN == 1
  if (param.act_type == activation::kReLU) {
    switch (inputs[0].dtype()) {
      case mshadow::kFloat32:
        MKLDNNRelu_Backward<float>(ctx, inputs[0], inputs[1], req[0],
            outputs[0]);
        return;
      default:
        break;
    }
  }
#endif
  _ActivationGradCompute<cpu>(param, ctx, inputs[0].data(), inputs[1].data(),
      req[0], outputs[0].data());
}

MXNET_OPERATOR_REGISTER_UNARY(Activation)
.describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", ActivationCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ActivationComputeEx_CPU)
.set_attr<nnvm::FGradient>("FGradient", ActivationGrad{"_backward_Activation"})
.add_arguments(ActivationParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Activation)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", ActivationGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ActivationGradComputeEx_CPU);

}  // namespace op
}  // namespace mxnet
