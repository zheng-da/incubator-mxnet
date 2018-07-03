# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from mxnet import gluon
import numpy as np
import copy
from numpy.testing import assert_allclose
import unittest
from mxnet.test_utils import almost_equal, default_context
from numpy.testing import assert_allclose as assert_almost_equal  # This is more restrictive


def test_simple_add():

    class _TestBlock(gluon.HybridBlock):

        def __init__(self, cond, func, max_iterations):
            super(_TestBlock, self).__init__()
            self.cond = cond
            self.func = func
            self.max_iterations = max_iterations

        def hybrid_forward(self, F, *loop_vars):
            return F.contrib.while_loop(
                cond=self.cond,
                func=self.func,
                loop_vars=loop_vars,
                max_iterations=self.max_iterations
            )

    for hybridize in [False, True]:
        # Case 1.1: result should be sum([1, 2, 3 ... 100])
        model = _TestBlock(
            cond=lambda i, s: i <= 5,
            func=lambda i, s: (None, (i + 1, s + i)),
            max_iterations=10,
        )
        if hybridize:
            model.hybridize()
        _, result = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
        )
        assert result[0].asscalar() == 6
        assert result[1].asscalar() == 15
        # Case 1.2: result should be sum([1, 2, 3 ... 1000])
        model = _TestBlock(
            cond=lambda i, s, true: true,
            func=lambda i, s, true: (None, (i + 1, s + i, true)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, result = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([1], dtype="int64"), # true
        )
        assert result[0].asscalar() == 1001
        assert result[1].asscalar() == 500500
        assert result[2].asscalar() == 1
        # Case 1.3: result should be sum([])
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (None, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, result = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([0], dtype="int64"), # false
        )
        assert result[0].asscalar() == 1
        assert result[1].asscalar() == 0
        assert result[2].asscalar() == 0
        # Case 2.1: result should be sum([1, 2, 3 ... 100])
        model = _TestBlock(
            cond=lambda i, s: i <= 100,
            func=lambda i, s: (i, (i + 1, s + i)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        (outputs, ), (result_i, result_s) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
        )
        assert all(outputs.asnumpy()[ : 100] == np.arange(1, 101).reshape(100, 1))
        assert result_i.asscalar() == 101
        assert result_s.asscalar() == 5050
        # Case 2.2: result should be sum([1, 2, 3 ... 1000])
        model = _TestBlock(
            cond=lambda i, s, true: true,
            func=lambda i, s, true: (i, (i + 1, s + i, true)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        (outputs, ), (result_i, result_s, _) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([1], dtype="int64"), # true
        )
        assert all(outputs.asnumpy() == np.arange(1, 1001).reshape(1000, 1))
        assert result_i.asscalar() == 1001
        assert result_s.asscalar() == 500500
        # Case 2.3: very corner case
        model = _TestBlock(
            cond=lambda i, s, false: false,
            func=lambda i, s, false: (i, (i + 1, s + i, false)),
            max_iterations=1000,
        )
        if hybridize:
            model.hybridize()
        _, (result_i, result_s, _) = model(
            mx.nd.array([1], dtype="int64"), # i
            mx.nd.array([0], dtype="int64"), # s
            mx.nd.array([0], dtype="int64"), # false
        )
        assert result_i.asscalar() == 1
        assert result_s.asscalar() == 0


def _verify_while_loop(cond, func, loop_var_shapes, free_var_shapes, is_train, max_iterations, is_for):

    def _create_vars(num, prefix):
        return [mx.sym.var(prefix + str(i)) for i in range(num)]

    def _create_arrays(shapes):
        return [mx.nd.random.uniform(-1.0, 1.0, shape=x) for x in shapes]

    def _create_dict(prefix, shapes):
        return {prefix + str(i): mx.nd.random.uniform(-1.0, 1.0, shape=x) for i, x in enumerate(shapes)}

    def _merge_dict(*dicts):
        result = {}
        for item in dicts:
            result.update(item)
        return result

    def _to_numpy_list(arrays):
        return [x.asnumpy() if x is not None else x for x in arrays]

    def _get_imperative_result():
        free_vars = [args["FreeVar" + str(i)].copy() for i, _ in enumerate(free_var_shapes)]
        loop_vars = [args["LoopVar" + str(i)].copy() for i, _ in enumerate(loop_var_shapes)]
        loop_var_start = int(is_for)
        if is_train:
            for var in free_vars + loop_vars[loop_var_start: ]:
                var.attach_grad()
        with mx.autograd.record(train_mode=is_train):
            outputs, final_loop_vars = mx.nd.contrib.while_loop(
                cond=lambda *_loop_vars: cond(_loop_vars, free_vars),
                func=lambda *_loop_vars: func(_loop_vars, free_vars),
                loop_vars=loop_vars,
                max_iterations=max_iterations,
            )
            n_steps = outputs[0].shape[0] if outputs else 0
            out_grads = _create_arrays(x.shape for x in outputs)  \
                      + _create_arrays(x.shape for x in final_loop_vars)
            loop_result_nd = [x for x in outputs] + [x for x in final_loop_vars]
            # loop_result_nd = [x * 2 for x in outputs] + [x * 3 for x in final_loop_vars]
            grads = []
            if is_train:
                cat_out = mx.nd.concat(*[x.reshape(-1) for x in loop_result_nd], dim=0)
                cat_out.backward(out_grad=mx.nd.concat(*[x.reshape(-1) for x in out_grads], dim=0))
                grads = [free_vars[i].grad for i, _ in enumerate(free_var_shapes)] \
                      + [loop_vars[i].grad for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
            return _to_numpy_list(loop_result_nd), _to_numpy_list(grads), out_grads, n_steps

    def _get_symbolic_result(out_grads, n_steps):

        def _copy_args_dict(name_list):
            return {name: args[name].copy() for name in name_list}

        def _zeros_like_dict(name_list):
            return {name: mx.nd.zeros_like(args[name]) for name in name_list}

        free_syms = _create_vars(len(free_var_shapes), "FreeVar")
        loop_syms = _create_vars(len(loop_var_shapes), "LoopVar")
        outputs, final_loop_syms = mx.sym.contrib.while_loop(
            cond=lambda *_loop_vars: cond(_loop_vars, free_syms),
            func=lambda *_loop_vars: func(_loop_vars, free_syms),
            loop_vars=loop_syms,
            max_iterations=max_iterations,
        )
        if n_steps == 0:
            outputs = []
        else:
            outputs = [x.slice_axis(axis=0, begin=0, end=n_steps) for x in outputs]
        # loop_result_sym = [x * 2 for x in outputs] + [x * 3 for x in final_loop_syms]
        loop_result_sym = [x for x in outputs] + [x for x in final_loop_syms]
        loop_result_sym = mx.sym.Group(loop_result_sym)

        loop_var_start = int(is_for)
        args_names = ["FreeVar" + str(i) for i, _ in enumerate(free_var_shapes)] \
                   + ["LoopVar" + str(i) for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
        args_grad = None if not is_train else _zeros_like_dict(x for x in args_names)
        executor = loop_result_sym.bind(
            ctx=default_context(),
            args=_copy_args_dict(loop_result_sym.list_inputs()),
            args_grad=args_grad,
        )
        loop_result_nd = executor.forward(is_train=is_train)
        grads = []
        if is_train:
            executor.backward(out_grads=out_grads)
            grads = [executor.grad_dict.get("FreeVar" + str(i), None) for i, _ in enumerate(free_var_shapes)] \
                  + [executor.grad_dict.get("LoopVar" + str(i), None) for i, _ in enumerate(loop_var_shapes) if i >= loop_var_start]
            print(args_names)
        return _to_numpy_list(loop_result_nd), _to_numpy_list(grads)

    args = _merge_dict(
        _create_dict("FreeVar", free_var_shapes),
        _create_dict("LoopVar", loop_var_shapes),
    )
    if is_for:
        assert loop_var_shapes[0] == (1, )
        args["LoopVar0"] = mx.nd.array([0])
    imp_outs, imp_grads, out_grads, n_steps = _get_imperative_result()
    print "Inputs"
    for name, arg in args.items():
        print name, arg
    print
    print "out_grads", out_grads
    sym_outs, sym_grads = _get_symbolic_result(out_grads, n_steps)
    for imp_out, sym_out in zip(imp_outs, sym_outs):
        if imp_out is None or sym_out is None:
            continue
        assert_almost_equal(imp_out, sym_out)
    print "imp_outs:"
    for a in imp_outs:
        print a
    print
    print "sym_outs:"
    for a in sym_outs:
        print a
    print
    print "imp_grads:"
    for a in imp_grads:
        print a
    print
    print "sym_grads:"
    for a in sym_grads:
        print a
    print
    for imp_grad, sym_grad in zip(imp_grads, sym_grads):
        if imp_grad is None or sym_grad is None:
            continue
        assert_almost_equal(imp_grad, sym_grad)


def test_while_loop_for_foreach():

    def make_true_cond():
        return lambda loop_vars, _: (loop_vars[0] < 1e9).prod()
 
    def make_false_cond():
        return lambda loop_vars, _: (loop_vars[0] > 1e9).prod()

    def make_for_cond(length):
        return lambda loop_vars, _: loop_vars[0] < length

    def case_1(**params):
        step_funcs = [
            lambda a, b, s: a * 1.5 + b * 2.5 - s * 3.5,
            lambda a, b, s: a * 1.5 - s * 3.5 + b * 2.5,
            lambda a, b, s: b * 2.5 + a * 1.5 - s * 3.5,
            lambda a, b, s: b * 2.5 - s * 3.5 + a * 1.5,
            lambda a, b, s: s * -3.5 + a * 1.5 + b * 2.5,
            lambda a, b, s: s * -3.5 + b * 2.5 + a * 1.5,
            lambda a, b, s: a * 2.5 * b + s * 0.3,
            lambda a, b, s: b * 2.5 * a + s * 0.3,
            lambda a, b, s: 2.5 * a * b + s * 0.3,
            lambda a, b, s: b * a * 2.5 + s * 0.3,
            lambda a, b, s: 2.5 * b * a + s * 0.3,
            lambda a, b, s: b * a * 2.5 + s * 0.3,
            lambda a, b, s: s * 0.3 + a * 2.5 * b,
            lambda a, b, s: s * 0.3 + b * 2.5 * a,
            lambda a, b, s: s * 0.3 + 2.5 * a * b,
            lambda a, b, s: s * 0.3 + b * a * 2.5,
            lambda a, b, s: s * 0.3 + 2.5 * b * a,
            lambda a, b, s: s * 0.3 + b * a * 2.5,
        ]
        def make_func(step_func):
            def step(loop, free):
                (s, ), (a, b) = loop, free
                out = step_func(a, b, s)
                return (out, out)
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                print "Case", case_id
                _verify_while_loop(
                    func=make_func(step_func),
                    is_train=is_train,
                    is_for=False,
                    **params
                )

    def case_2(**params):
        step_funcs = [
            lambda in_, s, f_1: in_ * f_1,
            # lambda in_, s, f_1: in_ * 2 + s + f_1,
            # lambda in_, s, f_1: s + in_ * 2 + f_1,
            # lambda in_, s, f_1: s + f_1 + in_ * 2,
            # lambda in_, s, f_1: f_1 + in_ * 2 + s,
            # lambda in_, s, f_1: f_1 + s + in_ * 2,
            # lambda in_, s, f_1: 2 * in_ + s + f_1,
            # lambda in_, s, f_1: 2 * in_ + f_1 + s,
            # lambda in_, s, f_1: s + 2 * in_ + f_1,
            # lambda in_, s, f_1: s + f_1 + 2 * in_,
            # lambda in_, s, f_1: f_1 + 2 * in_ + s,
            # lambda in_, s, f_1: f_1 + s + 2 * in_,
        ]
        def make_func(step_func):
            """This simulates:
            def compute(s, inputs, f_1, length):
                outputs = []
                for i in range(length):
                    s += inputs[i] * 2 + f_1
                    outputs.append(s)
                return outputs, s
            """
            def step(loop, free):
                (i, ), (scanned, f_1) = loop, free
                # (i, s), (scanned, f_1, _) = loop, free
                in_ = scanned.take(i).squeeze(axis=0)
                out = step_func(in_, None, f_1)
                return (out, i + 1)
            return step
        case_id = 0
        for is_train in [True, False]:
            for step_func in step_funcs:
                case_id += 1
                print "Case", case_id
                _verify_while_loop(
                    func=make_func(step_func),
                    max_iterations=1000,
                    is_train=is_train,
                    is_for=True,
                    **params
                )
    # # Case 0: the simpest case
    # print("Testing Case 0")
    # def _simple_func(loop, free):
    #     (i, ), (scanned, ) = loop, free
    #     in_ = scanned.take(i).squeeze(axis=0)
    #     return (in_, i + 1)
    # _verify_while_loop(
    #     cond=make_true_cond(),
    #     func=_simple_func,
    #     max_iterations=1,
    #     is_train=True,
    #     is_for=True,
    #     loop_var_shapes=[
    #         (1, ),          # i
    #     ],
    #     free_var_shapes=[
    #         (1, 3),         # scanned
    #     ],
    # )
    # # Case 1.1.*
    # print("Testing Case 1.1")
    # case_1(
    #     cond=make_true_cond(),
    #     loop_var_shapes=[
    #         (1, ),          # s
    #     ],
    #     free_var_shapes=[
    #         (1, ),          # a
    #         (1, ),          # b
    #     ],
    #     max_iterations=23
    # )
    # # Case 1.2.*
    # print("Testing Case 1.2")
    # case_1(
    #     cond=make_true_cond(),
    #     loop_var_shapes=[
    #         (2, 3, 4),      # s
    #     ],
    #     free_var_shapes=[
    #         (2, 3, 4),      # a
    #         (2, 3, 4),      # b
    #     ],
    #     max_iterations=31
    # )
    # # # Case 1.3.*
    # # print("Testing Case 1.3")
    # # case_1(
    # #     cond=make_true_cond(),
    # #     loop_var_shapes=[
    # #         (2, 3, 4),      # s
    # #     ],
    # #     free_var_shapes=[
    # #         (2, 3, 4),      # a
    # #         (2, 3, 4),      # b
    # #     ],
    # #     max_iterations=20
    # # )
    # Case 2.1.*
    print("Testing Case 2.1")
    length = 2
    case_2(
        cond=make_for_cond(length=length),
        loop_var_shapes=[
            (1, ),          # i
            # (1, ),          # s
        ],
        free_var_shapes=[
            (length, 1),    # scanned
            (1, ),          # f_1
            # (3, 4, 5, 6),   # f_2, unused
        ],
    )
    # Case 2.2.*
    # print("Testing Case 2.2")
    # case_2(
    #     cond=make_for_cond(length=1),
    #     loop_var_shapes=[
    #         (1, ),          # i
    #         (2, ),          # s
    #     ],
    #     free_var_shapes=[
    #         (3, 2),         # scanned
    #         (2, ),          # f_1
    #         (3, 4, 5, 6),   # f_2, unused
    #     ],
    # )


if __name__ == '__main__':
    # import nose
    # nose.runmodule()
    # test_simple_add()
    test_while_loop_for_foreach()
