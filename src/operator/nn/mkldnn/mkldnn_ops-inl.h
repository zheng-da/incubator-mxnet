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
 * \file mkldnn_ops-inl.h
 * \brief
 * \author Da Zheng
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <vector>
#include <mkldnn.hpp>

namespace mxnet {
namespace op {

/* For fully connected. */
void MKLDNNFCForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                     const std::vector<NDArray> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data);
void MKLDNNFCBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<NDArray> &outputs);

/* For convolution. */
void MKLDNNConvolutionForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                              const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data);
void MKLDNNConvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs);

/* For deconvolution */
void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data);
void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs);

/* For softmax */
void MKLDNNSoftmaxCompute(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const NDArray &in_data, const OpReqType &req,
                          const NDArray &out_data);

/* For sum */
void MKLDNNSumForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs, const OpReqType &req,
                      const NDArray &out_data);

/* For copy */
void MKLDNNCopy(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const NDArray &in_data, const OpReqType &req,
    const NDArray &out_data);

/* For concat */
void MKLDNNConcatForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                         const std::vector<NDArray> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &out_data);
void MKLDNNConcatBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs);

/* For activation */
void MKLDNNActivationForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const NDArray &in_data, const OpReqType &req,
                             const NDArray &out_data);
void MKLDNNActivationBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                              const NDArray &out_grad, const NDArray &in_data,
                              const OpReqType &req, const NDArray &in_grad);

void Sum(const mkldnn::memory &arr1, const mkldnn::memory &arr2,
         const mkldnn::memory &out);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1

#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_OPS_INL_H_
