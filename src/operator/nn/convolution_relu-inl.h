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
 * \file convolution_relu-inl.h
 * \brief
 * \author Zhang Rong A (rong.a.zhang@intel.com)
*/
#ifndef MXNET_OPERATOR_NN_CONVOLUTION_RELU_INL_H_
#define MXNET_OPERATOR_NN_CONVOLUTION_RELU_INL_H_


#if MXNET_USE_MKLDNN == 1

#include "./convolution-inl.h"


namespace mxnet {
namespace op {

struct ConvolutionReluParam : public dmlc::Parameter<ConvolutionReluParam> {
  // use for conv
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> layout;

  // use  for relu
  float slope;

  DMLC_DECLARE_PARAMETER(ConvolutionReluParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("Convolution stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("Convolution dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("Zero pad for convolution: (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temporary workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
    .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
    DMLC_DECLARE_FIELD(slope).set_default(0)
    .describe("Init slope for the activation. ");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const ConvolutionReluParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->workspace == other.workspace &&
           this->no_bias == other.no_bias &&
           this->layout == other.layout &&
           this->slope == other.slope;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_CONVOLUTION_RELU_INL_H_

