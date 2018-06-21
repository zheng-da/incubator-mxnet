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

# coding: utf-8
# pylint: disable=wildcard-import, unused-wildcard-import
"""Contrib NDArray API of MXNet."""
import math
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
try:
    from .gen_contrib import *
except ImportError:
    pass

__all__ = ["rand_zipfian"]

# pylint: disable=line-too-long
def rand_zipfian(true_classes, num_sampled, range_max, ctx=None):
    """Draw random samples from an approximately log-uniform or Zipfian distribution.

    This operation randomly samples *num_sampled* candidates the range of integers [0, range_max).
    The elements of sampled_candidates are drawn with replacement from the base distribution.

    The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

    P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

    This sampler is useful when the true classes approximately follow such a distribution.
    For example, if the classes represent words in a lexicon sorted in decreasing order of \
    frequency. If your classes are not ordered by decreasing frequency, do not use this op.

    Additionaly, it also returns the number of times each of the \
    true classes and the sampled classes is expected to occur.

    Parameters
    ----------
    true_classes : NDArray
        A 1-D NDArray of the target classes.
    num_sampled: int
        The number of classes to randomly sample.
    range_max: int
        The number of possible classes.
    ctx : Context
        Device context of output. Default is current context. Overridden by
        `mu.context` when `mu` is an NDArray.

    Returns
    -------
    samples: NDArray
        The sampled candidate classes in 1-D `int64` dtype.
    expected_count_true: NDArray
        The expected count for true classes in 1-D `float64` dtype.
    expected_count_sample: NDArray
        The expected count for sampled candidates in 1-D `float64` dtype.

    Examples
    --------
    >>> true_cls = mx.nd.array([3])
    >>> samples, exp_count_true, exp_count_sample = mx.nd.contrib.rand_zipfian(true_cls, 4, 5)
    >>> samples
    [1 3 3 3]
    <NDArray 4 @cpu(0)>
    >>> exp_count_true
    [ 0.12453879]
    <NDArray 1 @cpu(0)>
    >>> exp_count_sample
    [ 0.22629439  0.12453879  0.12453879  0.12453879]
    <NDArray 4 @cpu(0)>
    """
    if ctx is None:
        ctx = current_context()
    log_range = math.log(range_max + 1)
    rand = uniform(0, log_range, shape=(num_sampled,), dtype='float64', ctx=ctx)
    # make sure sampled_classes are in the range of [0, range_max)
    sampled_classes = (rand.exp() - 1).astype('int64') % range_max

    true_cls = true_classes.as_in_context(ctx).astype('float64')
    expected_count_true = ((true_cls + 2.0) / (true_cls + 1.0)).log() / log_range * num_sampled
    # cast sampled classes to fp64 to avoid interget division
    sampled_cls_fp64 = sampled_classes.astype('float64')
    expected_prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
    expected_count_sampled = expected_prob_sampled * num_sampled
    return sampled_classes, expected_count_true, expected_count_sampled
# pylint: enable=line-too-long

def foreach(body, data, init_states):
    """Run a for loop with user-defined computation over NDArrays on dimension 0.

    This operator simulates a for loop and body has the computation for an iteration
    of the for loop. It runs the computation in body on each slice from the input
    NDArrays.

    body takes two arguments as input and outputs a tuple of two elements,
    as illustrated below:

    out, states = body(data1, states)

    data1 can be either an NDArray or a list of NDArrays. If data is an NDArray,
    data1 is an NDArray. Otherwise, data1 is a list of NDArrays and has the same
    size as data. states is a list of NDArrays and have the same size as init_states.
    Similarly, out can be either an NDArray or a list of NDArrays, which are concatenated
    as the first output of foreach; states from the last execution of body
    are the second output of foreach.

    The computation done by this operator is equivalent to the pseudo code below
    when the input data is NDArray:

    states = init_states
    outs = []
    for i in data.shape[0]:
        s = data[i]
        out, states = body(s, states)
        outs.append(out)
    outs = stack(*outs)


    Parameters
    ----------
    body : a Python function.
        Define computation in an iteration.
    data: an NDArray or a list of NDArrays.
        The input data.
    init_states: an NDArray or a list of NDArrays.
        The initial values of the loop states.
    name: string.
        The name of the operator.

    Returns
    -------
    outputs: an NDArray or a list of NDArrays.
        The output data concatenated from the output of all iterations.
    states: a list of NDArrays.
        The loop states in the last iteration.

    Examples
    --------
    >>> step = lambda data, states: (data + states[0], [states[0] * 2])
    >>> data = mx.nd.random.uniform(shape=(2, 10))
    >>> states = [mx.nd.random.uniform(shape=(10))]
    >>> outs, states = mx.nd.contrib.foreach(step, data, states)
    """

    def check_input(inputs, in_type, msg):
        is_NDArray_or_list = True
        if isinstance(inputs, list):
            for i in inputs:
                if not isinstance(i, in_type):
                    is_NDArray_or_list = False
                    break
        else:
            is_NDArray_or_list = isinstance(inputs, in_type)
        assert is_NDArray_or_list, msg

    check_input(data, ndarray.NDArray, "data should be an NDArray or a list of NDArrays")
    check_input(init_states, ndarray.NDArray,
                "init_states should be an NDArray or a list of NDArrays")

    not_data_list = isinstance(data, ndarray.NDArray)
    num_iters = data.shape[0] if not_data_list else data[0].shape[0]
    states = init_states
    outputs = []
    for i in range(num_iters):
        if not_data_list:
            eles = data[i]
        else:
            eles = [d[i] for d in data]
        outs, states = body(eles, states)
        outs = _as_list(outs)
        outputs.append(outs)
    outputs = zip(*outputs)
    tmp_outputs = []
    for out in outputs:
        tmp_outputs.append(ndarray.op.stack(*out))
    outputs = tmp_outputs

    if not_data_list and len(outputs) == 1:
        outputs = outputs[0]
    return (outputs, states)


def while_loop(cond, func, loop_vars, max_iterations):
    """Run a while loop with user-defined computation and loop condition.

    This operator simulates a while loop and body has the computation for an iteration
    of the while loop. It runs the computation in body using loop variables,
    until the loop condition is not satisfied.

    ``cond'' is a user-defined loop condition function that takes ``loop_vars''
    as input and return a boolean scalar ndarray to determine the termination of the loop.
    The loop terminates when ``cond'' is false.

    ``func'' is a user-defined loop body function that takes ``loop_vars'' as input,
    performs computation and returns:
    1) a list of NDArrays, which has the same number of NDArrays and the same types as ``loop_vars''
    2) optionally outputs for each step
    The signature of ``fun'' can be:
       * def func(loop_vars): new_loop_vars
    or * def func(loop_vars): (output, new_loop_vars)

    ``loop_vars'' is a list of NDArrays that represent loop variables.

    ``max_iterations'' is a python or NDArray scalar that defines the maximal number of iterations.
    The maximal number of iterations is defined statically when the computation is constructed.

    The computation done by this operator is equivalent to the pseudo code below
    when the loop variables are NDArray:

    # ``func'' returns only the new_loop_vars
    steps = 0
    while steps < max_iterations and cond(loop_vars):
        loop_vars = func(loop_vars)
        steps += 1
    return loop_vars

    # ``func'' returns (output, new_loop_vars)
    steps = 0
    outputs = []
    while steps < max_iterations and cond(loop_vars):
        output, loop_vars = func(loop_vars)
        outputs.append(output)
        steps += 1
    return stack(outputs), loop_vars

    TODO(Junru): review the documentation, it has been out-of-date.
    Corner case #1: what if cond is always false? Do we return "outputs"?
    """
    def _to_python_type(inputs, type, name):
        """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
        if isinstance(inputs, ndarray.NDArray):
            inputs = inputs.asscalar()
        try:
            inputs = type(inputs)
        except:
            raise ValueError("Cannot convert %s to python %s" % (name, type.__name__))
        return inputs

    def _to_ndarray_tuple(inputs, name):
        """Converts "inputs", possibly a single mxnet NDArray, a list of mxnet NDArray,
        a tuple of mxnet NDArray, into a tuple of NDArray
        """
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if isinstance(inputs, ndarray.NDArray):
            inputs = (inputs, )
        if not isinstance(inputs, tuple):
            raise ValueError("%s must be an NDArray, or a tuple or list of NDArrays" % (name, ))
        for item in inputs:
            if not isinstance(item, ndarray.NDArray):
                raise ValueError("%s must be an NDArray, or a tuple or list of NDArrays" % (name, ))
        return inputs

    def _func_wrapper(loop_vars):
        """This wrapper unifies
             "func: loop_vars -> new_loop_vars"
         and "func: loop_vars -> (step_output, new_loop_vars)"
        into "func: loop_vars -> (None or tuple of step_outputs, tuple of new_loop_vars)
        """
        result = func(*loop_vars)
        if isinstance(result, ndarray.NDArray):
            result = (result, )
        if isinstance(result, list):
            result = tuple(result)
        if not isinstance(result, tuple):
            raise ValueError("Invalid return type of func: %s" % (type(result).__name__))
        if len(result) == 2 and (isinstance(result[1], list) or isinstance(result[1], tuple) or len(loop_vars) == 1):
            step_output, new_loop_vars = result
            step_output = _to_ndarray_tuple(step_output, "step_output")
        else:
            step_output, new_loop_vars = None, result
        new_loop_vars = _to_ndarray_tuple(new_loop_vars, "new_loop_vars")
        if len(loop_vars) != len(new_loop_vars):
            raise ValueError("The number of loop_vars should be consistent during the loop")
        return step_output, new_loop_vars

    max_iterations = _to_python_type(max_iterations, int, "max_iteration")
    loop_vars = _to_ndarray_tuple(loop_vars, "loop_vars")
    if len(loop_vars) == 0:
        raise ValueError("loop_vars should contain at least one element")

    steps = 0
    outputs = None
    while steps < max_iterations and \
            _to_python_type(cond(*loop_vars), bool, "Return value of cond"): # loop condition
        step_output, loop_vars = _func_wrapper(loop_vars)
        loop_vars = _to_ndarray_tuple(loop_vars, "loop_vars produced by func")
        if step_output is not None:
            outputs = outputs or []
            outputs.append(step_output)
        steps += 1
        if outputs is not None and len(outputs) != steps:
            raise ValueError("Whether func produces step_output is inconsistent in the loop")
    if len(loop_vars) == 1:
        loop_vars,  = loop_vars
    if outputs is not None:
        outputs = tuple(ndarray.op.stack(*item) for item in zip(*outputs))
        if len(outputs) == 1:
            outputs, = outputs
        return (outputs, loop_vars)
    return loop_vars
