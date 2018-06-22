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
from mxnet.test_utils import almost_equal, assert_almost_equal


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

    # Case 1.1: result should be sum([1, 2, 3 ... 100])
    model = _TestBlock(
        cond=lambda i, s: i <= 5,
        func=lambda i, s: (i + 1, s + i),
        max_iterations=10,
    )
    model.hybridize()
    result = model(
        mx.nd.array([1], dtype="int64"), # i
        mx.nd.array([0], dtype="int64"), # s
    )
    assert result[0].asscalar() == 6
    assert result[1].asscalar() == 15
    # Case 1.2: result should be sum([1, 2, 3 ... 1000])
    model = _TestBlock(
        cond=lambda i, s, true: true,
        func=lambda i, s, true: (i + 1, s + i, true),
        max_iterations=1000,
    )
    model.hybridize()
    result = model(
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
        func=lambda i, s, false: (i + 1, s + i, false),
        max_iterations=1000,
    )
    model.hybridize()
    result = model(
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
    model.hybridize()
    outputs, result_i, result_s = model(
        mx.nd.array([1], dtype="int64"), # i
        mx.nd.array([0], dtype="int64"), # s
    )
    assert all(outputs.asnumpy() == np.arange(1, 101).reshape(100, 1))
    assert result_i.asscalar() == 101
    assert result_s.asscalar() == 5050
    # Case 2.2: result should be sum([1, 2, 3 ... 1000])
    model = _TestBlock(
        cond=lambda i, s, true: true,
        func=lambda i, s, true: (i, (i + 1, s + i, true)),
        max_iterations=1000,
    )
    model.hybridize()
    outputs, result_i, result_s, _ = model(
        mx.nd.array([1], dtype="int64"), # i
        mx.nd.array([0], dtype="int64"), # s
        mx.nd.array([1], dtype="int64"), # s
    )
    assert all(outputs.asnumpy() == np.arange(1, 1001).reshape(1000, 1))
    assert result_i.asscalar() == 1001
    assert result_s.asscalar() == 500500
    # Case 2.3: very corner case
    # TODO(Junru, Da): in this case, the current implementation returns only loop_vars,
    # which causes inconsistency between symbolic and imperative mode.
    # We should discuss this.
    # model = _TestBlock(
    #     cond=lambda i, s: False,
    #     func=lambda i, s: (i, (i + 1, s + i)),
    #     max_iterations=1000,
    # )
    # result = model(
    #     mx.nd.array([1], dtype="int64"), # i
    #     mx.nd.array([0], dtype="int64"), # s
    # )
    # assert result[0].asscalar() == 1
    # assert result[1].asscalar() == 0


def test_simple_batched_add():
    pass


if __name__ == '__main__':
    # import nose
    # nose.runmodule()
    test_simple_add()
    test_simple_batched_add()
