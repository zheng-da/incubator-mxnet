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
 * \file activation-inl.h
 * \brief Activation operator
 * \author Bing Xu, Da Zheng
*/
#ifndef MXNET_OPERATOR_NN_ACTIVATION_INL_H_
#define MXNET_OPERATOR_NN_ACTIVATION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace activation {
enum ActivationOpInputs {kData};
enum ActivationOpOutputs {kOut};
enum ActivationOpType {kReLU, kSigmoid, kTanh, kSoftReLU};
}  // activation

struct ActivationParam : public dmlc::Parameter<ActivationParam> {
  // use int for enumeration
  int act_type;
  DMLC_DECLARE_PARAMETER(ActivationParam) {
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .describe("Activation function to be applied.");
  }
};

/**
 * \brief This is the implementation of activation operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
class ActivationOp {
 public:
  virtual void Forward(const OpContext &ctx, const TBlob &in_data,
                       const OpReqType &req, const TBlob &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data.FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data.FlatTo2D<xpu, DType>(s);
    Assign(out, req, F<ForwardOp>(data));
  }

  virtual void Backward(const OpContext &ctx, const TBlob &out_grad,
                        const TBlob &out_data, const OpReqType &req,
                        const TBlob &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> m_out_grad = out_grad.FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_data = out_data.FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_grad = in_grad.FlatTo2D<xpu, DType>(s);
    Assign(m_in_grad, req, F<BackwardOp>(m_out_data) * m_out_grad);
  }
};  // class ActivationOp

template<typename xpu, typename ForwardOp, typename BackwardOp, typename DType>
ActivationOp<xpu, ForwardOp, BackwardOp, DType> &get_activation_op() {
  static thread_local ActivationOp<xpu, ForwardOp, BackwardOp, DType> op;
  return op;
}

template<typename xpu>
void _ActivationCompute(const ActivationParam &param, const OpContext &ctx,
    const TBlob &input, OpReqType req, const TBlob &output) {
  MSHADOW_REAL_TYPE_SWITCH(input.type_flag_, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        get_activation_op<xpu, mshadow_op::relu, mshadow_op::relu_grad, DType>().Forward(
            ctx, input, req, output);
        break;
      case activation::kSigmoid:
        get_activation_op<xpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>().Forward(
            ctx, input, req, output);
        break;
      case activation::kTanh:
        get_activation_op<xpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>().Forward(
            ctx, input, req, output);
        break;
      case activation::kSoftReLU:
        get_activation_op<xpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>().Forward(
            ctx, input, req, output);
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  });
}

template<typename xpu>
void _ActivationGradCompute(const ActivationParam &param, const OpContext &ctx,
    const TBlob &out_grad, const TBlob &out_data, OpReqType req,
    const TBlob &output) {
  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        get_activation_op<xpu, mshadow_op::relu, mshadow_op::relu_grad, DType>().Backward(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kSigmoid:
        get_activation_op<xpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>().Backward(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kTanh:
        get_activation_op<xpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>().Backward(
            ctx, out_grad, out_data, req, output);
        break;
      case activation::kSoftReLU:
        get_activation_op<xpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>().Backward(
            ctx, out_grad, out_data, req, output);
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  });
}

template<typename xpu>
void ActivationCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  _ActivationCompute<xpu>(param, ctx, inputs[0], req[0], outputs[0]);
}

template<typename xpu>
void ActivationGradCompute(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
#if MXNET_USE_CUDNN == 1
  CHECK_EQ(inputs.size(), 3U);
#else
  CHECK_EQ(inputs.size(), 2U);
#endif
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  _ActivationGradCompute<xpu>(param, ctx, inputs[0], inputs[1], req[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_ACTIVATION_INL_H_
