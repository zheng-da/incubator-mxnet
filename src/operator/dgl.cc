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

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "../imperative/imperative_utils.h"
#include "./subgraph_op_common.h"

namespace mxnet {
namespace op {

struct SendRecvParam : public dmlc::Parameter<SendRecvParam> {
  int num_args;
  int num_outputs;
  nnvm::Tuple<dim_t> msg_index;
  // This index indicates the location of the reduce function inputs in the input array
  // of the operator. However, some of the inputs are not the operator inputs. Instead,
  // they are the outputs of the message function. We'll index these output arrays as if
  // they are at the end of the operator input array.
  nnvm::Tuple<dim_t> red_index;
  // This index indicates the location of the update function inputs in the input array
  // of the operator. The same indexing as red_index works here as well.
  nnvm::Tuple<dim_t> update_index;
  DMLC_DECLARE_PARAMETER(SendRecvParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(3)
    .describe("Number of input arguments, including all symbol inputs.");
    DMLC_DECLARE_FIELD(num_outputs).set_lower_bound(1)
    .describe("The number of outputs of the operator.");
    DMLC_DECLARE_FIELD(msg_index)
    .describe("The locations of message function inputs in the given inputs.");
    DMLC_DECLARE_FIELD(red_index)
    .describe("The locations of reduce function inputs in the given inputs.");
    DMLC_DECLARE_FIELD(update_index)
    .describe("The locations of vertex update function inputs in the given inputs.");
  }
};  // struct SendRecvParam

DMLC_REGISTER_PARAMETER(SendRecvParam);

//const std::string msg_idx = "msg_idx";
//const std::string pattern = "dummy_msgs_";
//
//static int get_msg_idx_loc(const nnvm::Symbol &red_sym) {
//  std::vector<std::string> input_names = red_sym.ListInputNames(nnvm::Symbol::kAll);
//  for (size_t i = 0; i < input_names.size(); i++)
//    if (input_names[i] == msg_idx)
//      return i;
//  return -1;
//}
//
//static std::vector<int> get_dummy_msgs_locs(const nnvm::Symbol &red_sym) {
//  std::vector<std::string> input_names = red_sym.ListInputNames(nnvm::Symbol::kAll);
//  std::vector<int> locs;
//  for (size_t i = 0; i < input_names.size(); i++) {
//    if (input_names[i].substr(0, pattern.length()) == pattern) {
//      locs.push_back(i);
//    }
//  }
//  return locs;
//}
//
//static std::vector<std::pair<int, int>> get_dummy_msg_loc_map() {
//  std::vector<std::string> input_names = red_sym.ListInputNames(nnvm::Symbol::kAll);
//  std::vector<int> dummy_locs;
//  std::vector<std::string> msg_names;
//  for (size_t i = 0; i < input_names.size(); i++) {
//    if (input_names[i].substr(0, pattern.length()) == pattern) {
//      dummy_locs.push_back(i);
//      msg_names.push_back(input_names[i].substr(pattern.length()));
//    }
//  }
//  std::string msg_locs;
//  for (auto msg_name : msg_names) {
//    for (size_t i = 0; i < input_names.size(); i++) {
//      if (input_names[i] == msg_name)
//      msg_locs.push_back(i);
//    }
//  }
//  CHECK_EQ(msg_locs.size(), dummy_locs.size());
//  std::vector<std::pair<int, int>> pairs(msg_locs.size());
//  for (size_t i = 0; i < pairs.size(); i++)
//  pairs[i] = std::pair<int, int>(dummy_locs[i], msg_locs[i]);
//  return pairs;
//}

static bool SendRecvStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  static const std::function<bool(const int &)> is_udf = is_stype_udf;
  CHECK_EQ(in_attrs->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  CHECK_EQ(attrs.subgraphs[2]->outputs.size(), params.num_outputs);

  *dispatch_mode = DispatchMode::kFComputeEx;
  std::vector<int> internal_in_attrs = *in_attrs;

  std::vector<int> msg_in_attrs;
  extract_by_loc(*in_attrs, params.msg_index, &msg_in_attrs);
  DispatchMode msg_mode = DispatchMode::kUndefined;
  std::vector<int> msg_out_attrs(attrs.subgraphs[0]->outputs.size(), kUndefinedStorage);
  bool succ_0 = InferSubgraphStorage(*attrs.subgraphs[0], dev_mask, \
                                     &msg_mode, &msg_in_attrs, &msg_out_attrs);
  CHECK(sync_in_in(params.msg_index, &internal_in_attrs, &msg_in_attrs, is_udf));
  // We index the outputs of the message function as if they are in the end of
  // the operator input array.
  for (size_t i = 0; i < msg_out_attrs.size(); i++)
    internal_in_attrs.push_back(msg_out_attrs[i]);

  // TODO construct storages for msg_idx and dummy_msgs.

  std::vector<int> red_in_attrs;
  extract_by_loc(internal_in_attrs, params.red_index, &red_in_attrs);
  DispatchMode red_mode = DispatchMode::kUndefined;
  std::vector<int> red_out_attrs(attrs.subgraphs[1]->outputs.size(), kUndefinedStorage);
  bool succ_1 = InferSubgraphStorage(*attrs.subgraphs[1], dev_mask, \
                                     &red_mode, &red_in_attrs, &red_out_attrs);
  CHECK(sync_in_in(params.red_index, &internal_in_attrs, &red_in_attrs, is_udf));
  // We index the outputs of the reduce function as if they are in the end of
  // the operator input array.
  for (size_t i = 0; i < red_out_attrs.size(); i++)
    internal_in_attrs.push_back(red_out_attrs[i]);

  std::vector<int> update_in_attrs;
  extract_by_loc(internal_in_attrs, params.update_index, &update_in_attrs);
  DispatchMode update_mode = DispatchMode::kUndefined;
  bool succ_2 = InferSubgraphStorage(*attrs.subgraphs[2], dev_mask, \
                                     &update_mode, &update_in_attrs, out_attrs);
  CHECK(sync_in_in(params.update_index, &internal_in_attrs, &update_in_attrs, is_udf));

  for (size_t i = 0; i < in_attrs->size(); i++)
    in_attrs->at(i) = internal_in_attrs[i];

  return succ_0 && succ_1 && succ_2;
}

static bool SendRecvType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  static const std::function<bool(const int &)> is_udf = is_type_udf;
  CHECK_EQ(in_attrs->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(out_attrs->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  CHECK_EQ(attrs.subgraphs[2]->outputs.size(), params.num_outputs);
  std::vector<int> internal_in_attrs = *in_attrs;

  std::vector<int> msg_in_attrs;
  extract_by_loc(*in_attrs, params.msg_index, &msg_in_attrs);
  std::vector<int> msg_out_attrs(attrs.subgraphs[0]->outputs.size(), -1);
  bool succ_0 = InferSubgraphDataType(*attrs.subgraphs[0], &msg_in_attrs, &msg_out_attrs);
  CHECK(sync_in_in(params.msg_index, &internal_in_attrs, &msg_in_attrs, is_udf));
  // We index the outputs of the message function as if they are in the end of
  // the operator input array.
  for (size_t i = 0; i < msg_out_attrs.size(); i++)
    internal_in_attrs.push_back(msg_out_attrs[i]);

  // TODO construct types for msg_idx and dummy_msgs.

  std::vector<int> red_in_attrs;
  extract_by_loc(internal_in_attrs, params.red_index, &red_in_attrs);
  std::vector<int> red_out_attrs(attrs.subgraphs[1]->outputs.size(), -1);
  bool succ_1 = InferSubgraphDataType(*attrs.subgraphs[1], &red_in_attrs, &red_out_attrs);
  CHECK(sync_in_in(params.red_index, &internal_in_attrs, &red_in_attrs, is_udf));
  // We index the outputs of the reduce function as if they are in the end of
  // the operator input array.
  for (size_t i = 0; i < red_out_attrs.size(); i++)
    internal_in_attrs.push_back(red_out_attrs[i]);

  std::vector<int> update_in_attrs;
  extract_by_loc(internal_in_attrs, params.update_index, &update_in_attrs);
  bool succ_2 = InferSubgraphDataType(*attrs.subgraphs[2], &update_in_attrs, out_attrs);
  CHECK(sync_in_in(params.update_index, &internal_in_attrs, &update_in_attrs, is_udf));

  for (size_t i = 0; i < in_attrs->size(); i++)
    in_attrs->at(i) = internal_in_attrs[i];

  return succ_0 && succ_1 && succ_2;
}

static bool SendRecvShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_shapes,
                          std::vector<TShape> *out_shapes) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  static const std::function<bool(const TShape &)> is_udf = is_shape_udf;
  CHECK_EQ(in_shapes->size() + 3U, (size_t) params.num_args);
  CHECK_EQ(out_shapes->size(), (size_t) params.num_outputs);
  CHECK_EQ(attrs.subgraphs.size(), 3U);
  CHECK_EQ(attrs.subgraphs[2]->outputs.size(), params.num_outputs);
  std::vector<TShape> internal_in_shapes = *in_shapes;

  // The third input array is a vector to indicate the reduce vertices.
  size_t num_reduce_vertices = in_shapes->at(2)[0];

  std::vector<TShape> msg_in_shapes;
  extract_by_loc(*in_shapes, params.msg_index, &msg_in_shapes);
  std::vector<TShape> msg_out_shapes(attrs.subgraphs[0]->outputs.size());
  bool succ_0 = InferSubgraphShape(*attrs.subgraphs[0], &msg_in_shapes, &msg_out_shapes);
  CHECK(sync_in_in(params.msg_index, &internal_in_shapes, &msg_in_shapes, is_udf));
  // We index the outputs of the message function as if they are in the end of
  // the operator input array.
  for (size_t i = 0; i < msg_out_shapes.size(); i++) {
    TShape s(msg_out_shapes[i].ndim() + 1);
    s[0] = num_reduce_vertices;
    s[1] = 2;     // In practice, this dimension indicates the number of messages in the neighborhood.
    for (size_t j = 2; j < msg_out_shapes[i].ndim(); j++)
      s[j] = msg_out_shapes[i][j];
    internal_in_shapes.push_back(s);
  }

  // TODO construct shapes for msg_idx and dummy_msgs.

  std::vector<TShape> red_in_shapes;
  extract_by_loc(internal_in_shapes, params.red_index, &red_in_shapes);
  std::vector<int> red_out_shapes(attrs.subgraphs[1]->outputs.size(), -1);
  bool succ_1 = InferSubgraphShape(*attrs.subgraphs[1], &red_in_shapes, out_shapes);
  CHECK(sync_in_in(params.red_index, &internal_in_shapes, &red_in_shapes, is_udf));
  // We index the outputs of the reduce function as if they are in the end of
  // the operator input array.
  for (size_t i = 0; i < red_out_shapes.size(); i++)
    internal_in_shapes.push_back(red_out_shapes[i]);

  std::vector<TShape> update_in_shapes;
  extract_by_loc(internal_in_shapes, params.update_index, &update_in_shapes);
  bool succ_2 = InferSubgraphShape(*attrs.subgraphs[2], &update_in_shapes, out_shapes);
  CHECK(sync_in_in(params.update_index, &internal_in_shapes, &update_in_shapes, is_udf));

  for (size_t i = 0; i < in_shapes->size(); i++)
    in_shapes->at(i) = internal_in_shapes[i];

  return succ_0 && succ_1 && succ_2;
}

class SendRecvState {
 public:
  SendRecvParam params;
  CachedOpPtr msg_op;
  CachedOpPtr red_op;
  CachedOpPtr update_op;
  OpStatePtr msg_state;
  OpStatePtr red_state;
  OpStatePtr update_state;

  static CachedOpPtr MakeSharedOp(const Symbol &sym) {
    // We turn on static_alloc for two reasons.
    // It avoids the overhead of unnecessary memory allocation.
    // only static_alloc supports nested call of CachedOp.
    std::vector<std::pair<std::string, std::string> > kwargs = {
      {"inline_limit", "0"},
      {"static_alloc", "1"}
    };
    return std::make_shared<CachedOp>(sym, kwargs);
  }

  SendRecvState(const SendRecvParam &params,
                const Symbol &msg_sym,
                const Symbol &red_sym,
                const Symbol &update_sym): params(params) {
    msg_op = MakeSharedOp(msg_sym);
    red_op = MakeSharedOp(red_sym);
    update_op = MakeSharedOp(update_sym);
  }

};

static OpStatePtr CreateSendRecvState(const NodeAttrs& attrs,
                                      Context ctx,
                                      const std::vector<TShape>& ishape,
                                      const std::vector<int>& itype) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  return OpStatePtr::Create<SendRecvState>(
    params,
    *attrs.subgraphs[0],
    *attrs.subgraphs[1],
    *attrs.subgraphs[2]);
}

static void SendRecvComputeExCPU(const OpStatePtr& state_ptr, const OpContext& ctx, const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req, const std::vector<NDArray>& outputs) {
  SendRecvState &state = state_ptr.get_state<SendRecvState>();
  const SendRecvParam &params = state.params;
  // These are the vertex Id vectors.
  CHECK_EQ(inputs[0].shape().ndim(), 1U);
  CHECK_EQ(inputs[1].shape().ndim(), 1U);
  CHECK_EQ(inputs[2].shape().ndim(), 1U);
  CHECK_EQ(inputs[0].dtype(), mshadow::kInt64);
  CHECK_EQ(inputs[1].dtype(), mshadow::kInt64);
  CHECK_EQ(inputs[2].dtype(), mshadow::kInt64);
  size_t num_receivers = inputs[2].shape()[0];
//  const int64_t *send_ids = inputs[0].data().dptr<int64_t>();
  const int64_t *recv_ids = inputs[1].data().dptr<int64_t>();
  const int64_t *uniq_recv = inputs[2].data().dptr<int64_t>();

  // Execute the message function.
  std::vector<NDArray> msg_inputs;
  extract_by_loc(inputs, params.msg_index, &msg_inputs);
  // TODO do we have to do shape inference here?
  std::vector<NDArray> msg_outputs(state.msg_op->num_outputs());
  std::vector<NDArray *> msg_input_ptrs(msg_inputs.size());
  for (size_t i = 0; i < msg_input_ptrs.size(); i++)
    msg_input_ptrs[i] = &msg_inputs[i];
  std::vector<NDArray *> msg_output_ptrs(msg_outputs.size());
  for (size_t i = 0; i < msg_output_ptrs.size(); i++)
    msg_output_ptrs[i] = &msg_outputs[i];
  state.msg_state = state.msg_op->Forward(nullptr, msg_input_ptrs, msg_output_ptrs);

  // Execute the reduce function.
  // TODO Let's run this step in the most naive way: run one neighborhood at a time.
  std::vector<NDArray> red_outputs(state.red_op->num_outputs());
  off_t recv_start = 0;
  for (size_t i = 0; i < num_receivers; i++) {
    // First we need to read the messages from the right location.
    off_t recv_end = recv_start;
    for (; recv_ids[recv_end] != uniq_recv[i]; recv_end++);
    recv_start = recv_end;

    size_t num_recvs = 1;
    std::vector<NDArray> internal_inputs = inputs;
    for (size_t i = 0; i < msg_outputs.size(); i++) {
      NDArray slice = msg_outputs[i].Slice(recv_start, recv_end);
      TShape shape = slice.shape();
      TShape new_shape(shape.ndim() + 1);
      new_shape[0] = num_recvs;
      for (size_t j = 1; j < new_shape.ndim(); j++)
        new_shape[j] = shape[j - 1];
      internal_inputs.push_back(slice.Reshape(new_shape));
    }
    std::vector<NDArray> red_inputs;
    extract_by_loc(internal_inputs, params.red_index, &red_inputs);

    std::vector<NDArray> red_outputs(state.red_op->num_outputs());
    std::vector<NDArray *> red_input_ptrs(red_inputs.size());
    for (size_t i = 0; i < red_input_ptrs.size(); i++)
      red_input_ptrs[i] = &red_inputs[i];
    std::vector<NDArray *> red_output_ptrs(red_outputs.size());
    // TODO the outputs should be sliced from the global output arrays.
    for (size_t i = 0; i < red_output_ptrs.size(); i++)
      red_output_ptrs[i] = &red_outputs[i];
    state.red_state = state.red_op->Forward(nullptr, red_input_ptrs, red_output_ptrs);
  }

  std::vector<NDArray> internal_inputs = inputs;
  for (size_t i = 0; i < msg_outputs.size(); i++) {
    // These are the placeholders for the outputs of the message functions.
    internal_inputs.push_back(NDArray());
  }
  for (size_t i = 0; i < red_outputs.size(); i++) {
    internal_inputs.push_back(red_outputs[i]);
  }

  // Execute the vertex update function.
  std::vector<NDArray> update_inputs;
  extract_by_loc(internal_inputs, params.update_index, &update_inputs);
  std::vector<NDArray> update_outputs(state.update_op->num_outputs());
  std::vector<NDArray *> update_input_ptrs(update_inputs.size());
  for (size_t i = 0; i < update_input_ptrs.size(); i++)
    update_input_ptrs[i] = &update_inputs[i];
  std::vector<NDArray *> update_output_ptrs(update_outputs.size());
  for (size_t i = 0; i < update_output_ptrs.size(); i++)
    update_output_ptrs[i] = &update_outputs[i];
  state.update_state = state.update_op->Forward(nullptr, update_input_ptrs, update_output_ptrs);
}

NNVM_REGISTER_OP(_send_and_recv)
.MXNET_DESCRIBE("")
.set_attr_parser(ParamParser<SendRecvParam>)
.set_attr<FInferStorageType>("FInferStorageType", SendRecvStorageType)
.set_num_inputs([](const NodeAttrs& attrs) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  return params.num_args;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  return params.num_outputs;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
  std::vector<std::string> names;
  names.reserve(params.num_args);
  names.push_back("message_func");
  names.push_back("reduce_func");
  names.push_back("vertex_update_func");
  for (int i = 3; i < params.num_args; ++i)
    names.push_back("data" + std::to_string(i - 3));
  return names;
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return std::vector<uint32_t>{0, 1, 2};
})
//.set_attr<nnvm::FGradient>("FGradient", SendRecvGradient)
.set_attr<FCreateOpState>("FCreateOpState", CreateSendRecvState)
.set_attr<nnvm::FInferShape>("FInferShape", SendRecvShape)
.set_attr<nnvm::FInferType>("FInferType", SendRecvType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SendRecvComputeExCPU)
.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
  return ExecType::kSubgraphExec;
})
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", SendRecvComputeExCPU)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("message_func", "Symbol", "Input graph for the message function.")
.add_argument("reduce_func", "Symbol", "Input graph for the reduce function.")
.add_argument("vertex_update_func", "Symbol", "Input graph for the vertex update function.")
.add_argument("data", "NDArray-or-Symbol[]",
              "The input arrays that include data arrays and states.")
.add_arguments(SendRecvParam::__FIELDS__());

//NNVM_REGISTER_OP(_backward_cond)
//.set_num_inputs([](const NodeAttrs& attrs){
//  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
//  return params.num_outputs * 2 + params.num_args - 3;
//})
//.set_num_outputs([](const NodeAttrs& attrs){
//  const SendRecvParam& params = nnvm::get<SendRecvParam>(attrs.parsed);
//  return params.num_args - 3;
//})
//.set_attr<FExecType>("FExecType", [](const NodeAttrs& attrs) {
//  return ExecType::kSubgraphExec;
//})
//.set_attr<FInferStorageType>("FInferStorageType", BackwardSendRecvStorageType)
//.set_attr_parser(ParamParser<SendRecvParam>)
//.set_attr<bool>("TIsLayerOpBackward", true)
//.set_attr<nnvm::TIsBackward>("TIsBackward", true)
//.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SendRecvGradComputeExCPU)
//.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", SendRecvGradComputeExCPU);

}  // namespace op
}  // namespace mxnet
