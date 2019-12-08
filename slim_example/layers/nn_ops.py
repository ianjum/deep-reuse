# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.framework import load_library
user_conv_module = load_library.load_op_library('../user_ops/user_conv.so')

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access


def _non_atrous_convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    data_format=None,  # pylint: disable=redefined-builtin
    quantized=False,
    quantization_params=None,
    strides=None,
    name=None):
  """Computes sums of N-D convolutions (actually cross correlation).
  It is required that 1 <= N <= 3.
  This is used to implement the more generic `convolution` function, which
  extends the interface of this function with a `dilation_rate` parameter.
  Args:
    input: Rank N+2 tensor of type T of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if `data_format`
      does not start with `"NC"`, or
      `[batch_size, in_channels] + input_spatial_shape` if `data_format` starts
      with `"NC"`.
    filter: Rank N+2 tensor of type T of shape
      `filter_spatial_shape + [in_channels, out_channels]`.  Rank of either
      `input` or `filter` must be known.
    padding: Padding method to use, must be either "VALID" or "SAME".
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    strides: Sequence of N positive integers, defaults to `[1] * N`.
    name: Name prefix to use.
  Returns:
    Rank N+2 tensor of type T of shape
    `[batch_size] + output_spatial_shape + [out_channels]`, where
    if padding == "SAME":
      output_spatial_shape = input_spatial_shape
    if padding == "VALID":
      output_spatial_shape = input_spatial_shape - filter_spatial_shape + 1.
  Raises:
    ValueError: if ranks are incompatible.
  """
  with ops.name_scope(name, "non_atrous_convolution", [input, filter]) as scope:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
    input_shape = input.get_shape()
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
    filter_shape = filter.get_shape()
    op = _NonAtrousConvolution(
        input_shape,
        filter_shape=filter_shape,
        padding=padding,
        data_format=data_format,
        quantized=quantized,
        quantization_params=quantization_params,
        strides=strides,
        name=scope)
    return op(input, filter)


class _NonAtrousConvolution(object):
  """Helper class for _non_atrous_convolution.
  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.
  Arguments:
    input_shape: static input shape, i.e. input.get_shape().
    filter_shape: static filter shape, i.e. filter.get_shape().
    padding: see _non_atrous_convolution.
    data_format: see _non_atrous_convolution.
    strides: see _non_atrous_convolution.
    name: see _non_atrous_convolution.
  """

  def __init__(
      self,
      input_shape,
      filter_shape,  # pylint: disable=redefined-builtin
      padding,
      data_format=None,
      quantized=False,
      quantization_params=None,
      strides=None,
      name=None):
    print('======================Initilaize]]]]]]]]]]]]]]]]]]]]', quantized)
    filter_shape = filter_shape.with_rank(input_shape.ndims)
    self.padding = padding
    self.name = name
    self.quantized=quantized
    self.quantization_params=quantization_params
    input_shape = input_shape.with_rank(filter_shape.ndims)
    if input_shape.ndims is None:
      raise ValueError("Rank of convolution must be known")
    if input_shape.ndims < 3 or input_shape.ndims > 5:
      raise ValueError(
          "`input` and `filter` must have rank at least 3 and at most 5")
    conv_dims = input_shape.ndims - 2
    if strides is None:
      strides = [1] * conv_dims
    elif len(strides) != conv_dims:
      raise ValueError("len(strides)=%d, but should be %d" % (len(strides),
                                                              conv_dims))
    if conv_dims == 1:
      # conv1d uses the 2-d data format names
      if data_format is None:
        data_format = "NWC"
      elif data_format not in {"NCW", "NWC", "NCHW", "NHWC"}:
        raise ValueError("data_format must be \"NWC\" or \"NCW\".")
      self.strides = strides[0]
      self.data_format = data_format
      self.conv_op = self._conv1d
    elif conv_dims == 2:
      if data_format is None or data_format == "NHWC":
        data_format = "NHWC"
        strides = [1] + list(strides) + [1]
      elif data_format == "NCHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")
      self.strides = strides
      self.data_format = data_format
      print('==========',conv_dims)
      if quantized:
        print("---------------quantization_params:", quantization_params[1],"---------")
        self.conv_op = user_conv_module.user_conv
      else:
        print('==========+++++',conv_dims)
        self.conv_op = gen_nn_ops.conv2d
    elif conv_dims == 3:
      if data_format is None or data_format == "NDHWC":
        strides = [1] + list(strides) + [1]
      elif data_format == "NCDHW":
        strides = [1, 1] + list(strides)
      else:
        raise ValueError("data_format must be \"NDHWC\" or \"NCDHW\". Have: %s"
                         % data_format)
      self.strides = strides
      self.data_format = data_format
      self.conv_op = gen_nn_ops.conv3d

  # Note that we need this adapter since argument names for conv1d don't match
  # those for gen_nn_ops.conv2d and gen_nn_ops.conv3d.
  # pylint: disable=redefined-builtin
  def _conv1d(self, input, filter, strides, padding, data_format, name):
    return conv1d(
        value=input,
        filters=filter,
        stride=strides,
        padding=padding,
        data_format=data_format,
        name=name)

  # pylint: enable=redefined-builtin

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    if self.quantized == True:
      return self.conv_op(
          input=inp,
          filter=filter,
          quantization_params=self.quantization_params,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          name=self.name)
    else:
      return self.conv_op(
          input=inp,
          filter=filter,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          name=self.name)


# @tf_export("nn.with_space_to_batch")
def with_space_to_batch(
    input,  # pylint: disable=redefined-builtin
    dilation_rate,
    padding,
    op,
    filter_shape=None,
    spatial_dims=None,
    data_format=None):
  """Performs `op` on the space-to-batch representation of `input`.
  This has the effect of transforming sliding window operations into the
  corresponding "atrous" operation in which the input is sampled at the
  specified `dilation_rate`.
  In the special case that `dilation_rate` is uniformly 1, this simply returns:
    op(input, num_spatial_dims, padding)
  Otherwise, it returns:
    batch_to_space_nd(
      op(space_to_batch_nd(input, adjusted_dilation_rate, adjusted_paddings),
         num_spatial_dims,
         "VALID")
      adjusted_dilation_rate,
      adjusted_crops),
  where:
    adjusted_dilation_rate is an int64 tensor of shape [max(spatial_dims)],
    adjusted_{paddings,crops} are int64 tensors of shape [max(spatial_dims), 2]
  defined as follows:
  We first define two int64 tensors `paddings` and `crops` of shape
  `[num_spatial_dims, 2]` based on the value of `padding` and the spatial
  dimensions of the `input`:
  If `padding = "VALID"`, then:
    paddings, crops = required_space_to_batch_paddings(
      input_shape[spatial_dims],
      dilation_rate)
  If `padding = "SAME"`, then:
    dilated_filter_shape =
      filter_shape + (filter_shape - 1) * (dilation_rate - 1)
    paddings, crops = required_space_to_batch_paddings(
      input_shape[spatial_dims],
      dilation_rate,
      [(dilated_filter_shape - 1) // 2,
       dilated_filter_shape - 1 - (dilated_filter_shape - 1) // 2])
  Because `space_to_batch_nd` and `batch_to_space_nd` assume that the spatial
  dimensions are contiguous starting at the second dimension, but the specified
  `spatial_dims` may not be, we must adjust `dilation_rate`, `paddings` and
  `crops` in order to be usable with these operations.  For a given dimension,
  if the block size is 1, and both the starting and ending padding and crop
  amounts are 0, then space_to_batch_nd effectively leaves that dimension alone,
  which is what is needed for dimensions not part of `spatial_dims`.
  Furthermore, `space_to_batch_nd` and `batch_to_space_nd` handle this case
  efficiently for any number of leading and trailing dimensions.
  For 0 <= i < len(spatial_dims), we assign:
    adjusted_dilation_rate[spatial_dims[i] - 1] = dilation_rate[i]
    adjusted_paddings[spatial_dims[i] - 1, :] = paddings[i, :]
    adjusted_crops[spatial_dims[i] - 1, :] = crops[i, :]
  All unassigned values of `adjusted_dilation_rate` default to 1, while all
  unassigned values of `adjusted_paddings` and `adjusted_crops` default to 0.
  Note in the case that `dilation_rate` is not uniformly 1, specifying "VALID"
  padding is equivalent to specifying `padding = "SAME"` with a filter_shape of
  `[1]*N`.
  Advanced usage. Note the following optimization: A sequence of
  `with_space_to_batch` operations with identical (not uniformly 1)
  `dilation_rate` parameters and "VALID" padding
    net = with_space_to_batch(net, dilation_rate, "VALID", op_1)
    ...
    net = with_space_to_batch(net, dilation_rate, "VALID", op_k)
  can be combined into a single `with_space_to_batch` operation as follows:
    def combined_op(converted_input, num_spatial_dims, _):
      result = op_1(converted_input, num_spatial_dims, "VALID")
      ...
      result = op_k(result, num_spatial_dims, "VALID")
    net = with_space_to_batch(net, dilation_rate, "VALID", combined_op)
  This eliminates the overhead of `k-1` calls to `space_to_batch_nd` and
  `batch_to_space_nd`.
  Similarly, a sequence of `with_space_to_batch` operations with identical (not
  uniformly 1) `dilation_rate` parameters, "SAME" padding, and odd filter
  dimensions
    net = with_space_to_batch(net, dilation_rate, "SAME", op_1, filter_shape_1)
    ...
    net = with_space_to_batch(net, dilation_rate, "SAME", op_k, filter_shape_k)
  can be combined into a single `with_space_to_batch` operation as follows:
    def combined_op(converted_input, num_spatial_dims, _):
      result = op_1(converted_input, num_spatial_dims, "SAME")
      ...
      result = op_k(result, num_spatial_dims, "SAME")
    net = with_space_to_batch(net, dilation_rate, "VALID", combined_op)
  Args:
    input: Tensor of rank > max(spatial_dims).
    dilation_rate: int32 Tensor of *known* shape [num_spatial_dims].
    padding: str constant equal to "VALID" or "SAME"
    op: Function that maps (input, num_spatial_dims, padding) -> output
    filter_shape: If padding = "SAME", specifies the shape of the convolution
      kernel/pooling window as an integer Tensor of shape [>=num_spatial_dims].
      If padding = "VALID", filter_shape is ignored and need not be specified.
    spatial_dims: Monotonically increasing sequence of `num_spatial_dims`
      integers (which are >= 1) specifying the spatial dimensions of `input`
      and output.  Defaults to: `range(1, num_spatial_dims+1)`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
  Returns:
    The output Tensor as described above, dimensions will vary based on the op
    provided.
  Raises:
    ValueError: if `padding` is invalid or the arguments are incompatible.
    ValueError: if `spatial_dims` are invalid.
  """
  input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
  input_shape = input.get_shape()

  def build_op(num_spatial_dims, padding):
    return lambda inp, _: op(inp, num_spatial_dims, padding)

  new_op = _WithSpaceToBatch(
      input_shape,
      dilation_rate,
      padding,
      build_op,
      filter_shape=filter_shape,
      spatial_dims=spatial_dims,
      data_format=data_format)
  return new_op(input, None)


class _WithSpaceToBatch(object):
  """Helper class for with_space_to_batch.
  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.
  Arguments
    input_shape: static shape of input. i.e. input.get_shape().
    dilation_rate: see with_space_to_batch
    padding: see with_space_to_batch
    build_op: Function that maps (num_spatial_dims, paddings) -> (function that
      maps (input, filter) -> output).
    filter_shape: see with_space_to_batch
    spatial_dims: see with_space_to_batch
    data_format: see with_space_to_batch
  """

  def __init__(self,
               input_shape,
               dilation_rate,
               padding,
               build_op,
               filter_shape=None,
               spatial_dims=None,
               data_format=None):
    """Helper class for _with_space_to_batch."""
    dilation_rate = ops.convert_to_tensor(
        dilation_rate, dtypes.int32, name="dilation_rate")
    try:
      rate_shape = dilation_rate.get_shape().with_rank(1)
    except ValueError:
      raise ValueError("rate must be rank 1")

    if not dilation_rate.get_shape().is_fully_defined():
      raise ValueError("rate must have known shape")

    num_spatial_dims = rate_shape[0].value

    if data_format is not None and data_format.startswith("NC"):
      starting_spatial_dim = 2
    else:
      starting_spatial_dim = 1

    if spatial_dims is None:
      spatial_dims = range(starting_spatial_dim,
                           num_spatial_dims + starting_spatial_dim)
    orig_spatial_dims = list(spatial_dims)
    spatial_dims = sorted(set(int(x) for x in orig_spatial_dims))
    if spatial_dims != orig_spatial_dims or any(x < 1 for x in spatial_dims):
      raise ValueError(
          "spatial_dims must be a montonically increasing sequence of positive "
          "integers")  # pylint: disable=line-too-long

    if data_format is not None and data_format.startswith("NC"):
      expected_input_rank = spatial_dims[-1]
    else:
      expected_input_rank = spatial_dims[-1] + 1

    try:
      input_shape.with_rank_at_least(expected_input_rank)
    except ValueError:
      ValueError("input tensor must have rank %d at least" %
                 (expected_input_rank))

    const_rate = tensor_util.constant_value(dilation_rate)
    rate_or_const_rate = dilation_rate
    if const_rate is not None:
      rate_or_const_rate = const_rate
      if np.any(const_rate < 1):
        raise ValueError("dilation_rate must be positive")
      if np.all(const_rate == 1):
        self.call = build_op(num_spatial_dims, padding)
        return

    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.

    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      if filter_shape is None:
        raise ValueError("filter_shape must be specified for SAME padding")
      filter_shape = ops.convert_to_tensor(filter_shape, name="filter_shape")
      const_filter_shape = tensor_util.constant_value(filter_shape)
      if const_filter_shape is not None:
        filter_shape = const_filter_shape
        self.base_paddings = _with_space_to_batch_base_paddings(
            const_filter_shape, num_spatial_dims, rate_or_const_rate)
      else:
        self.num_spatial_dims = num_spatial_dims
        self.rate_or_const_rate = rate_or_const_rate
        self.base_paddings = None
    elif padding == "VALID":
      self.base_paddings = np.zeros([num_spatial_dims, 2], np.int32)
    else:
      raise ValueError("Invalid padding method %r" % padding)

    self.input_shape = input_shape
    self.spatial_dims = spatial_dims
    self.dilation_rate = dilation_rate
    self.data_format = data_format
    self.op = build_op(num_spatial_dims, "VALID")
    self.call = self._with_space_to_batch_call

  def _with_space_to_batch_call(self, inp, filter):  # pylint: disable=redefined-builtin
    """Call functionality for with_space_to_batch."""
    # Handle input whose shape is unknown during graph creation.
    input_spatial_shape = None
    input_shape = self.input_shape
    spatial_dims = self.spatial_dims
    if input_shape.ndims is not None:
      input_shape_list = input_shape.as_list()
      input_spatial_shape = [input_shape_list[i] for i in spatial_dims]
    if input_spatial_shape is None or None in input_spatial_shape:
      input_shape_tensor = array_ops.shape(inp)
      input_spatial_shape = array_ops.stack(
          [input_shape_tensor[i] for i in spatial_dims])

    base_paddings = self.base_paddings
    if base_paddings is None:
      # base_paddings could not be computed at build time since static filter
      # shape was not fully defined.
      filter_shape = array_ops.shape(filter)
      base_paddings = _with_space_to_batch_base_paddings(
          filter_shape, self.num_spatial_dims, self.rate_or_const_rate)
    paddings, crops = array_ops.required_space_to_batch_paddings(
        input_shape=input_spatial_shape,
        base_paddings=base_paddings,
        block_shape=self.dilation_rate)

    dilation_rate = _with_space_to_batch_adjust(self.dilation_rate, 1,
                                                spatial_dims)
    paddings = _with_space_to_batch_adjust(paddings, 0, spatial_dims)
    crops = _with_space_to_batch_adjust(crops, 0, spatial_dims)
    input_converted = array_ops.space_to_batch_nd(
        input=inp, block_shape=dilation_rate, paddings=paddings)

    result = self.op(input_converted, filter)

    result_converted = array_ops.batch_to_space_nd(
        input=result, block_shape=dilation_rate, crops=crops)

    # Recover channel information for output shape if channels are not last.
    if self.data_format is not None and self.data_format.startswith("NC"):
      if not result_converted.shape[1].value and filter is not None:
        output_shape = result_converted.shape.as_list()
        output_shape[1] = filter.shape[-1]
        result_converted.set_shape(output_shape)

    return result_converted

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.call(inp, filter)


def _with_space_to_batch_base_paddings(filter_shape, num_spatial_dims,
                                       rate_or_const_rate):
  """Helper function to compute base_paddings."""
  # Spatial dimensions of the filters and the upsampled filters in which we
  # introduce (rate - 1) zeros between consecutive filter values.
  filter_spatial_shape = filter_shape[:num_spatial_dims]
  dilated_filter_spatial_shape = (
      filter_spatial_shape + (filter_spatial_shape - 1) *
      (rate_or_const_rate - 1))
  pad_extra_shape = dilated_filter_spatial_shape - 1

  # When full_padding_shape is odd, we pad more at end, following the same
  # convention as conv2d.
  pad_extra_start = pad_extra_shape // 2
  pad_extra_end = pad_extra_shape - pad_extra_start
  base_paddings = array_ops.stack(
      [[pad_extra_start[i], pad_extra_end[i]] for i in range(num_spatial_dims)])
  return base_paddings


def _with_space_to_batch_adjust(orig, fill_value, spatial_dims):
  """Returns an `adjusted` version of `orig` based on `spatial_dims`.
  Tensor of the same type as `orig` and with shape
  `[max(spatial_dims), ...]` where:
    adjusted[spatial_dims[i] - 1, ...] = orig[i, ...]
  for 0 <= i < len(spatial_dims), and
    adjusted[j, ...] = fill_value
  for j != spatial_dims[i] - 1 for some i.
  If `orig` is a constant value, then the result will be a constant value.
  Args:
    orig: Tensor of rank > max(spatial_dims).
    fill_value: Numpy scalar (of same data type as `orig) specifying the fill
      value for non-spatial dimensions.
    spatial_dims: See with_space_to_batch.
  Returns:
    `adjusted` tensor.
  """
  fill_dims = orig.get_shape().as_list()[1:]
  dtype = orig.dtype.as_numpy_dtype
  parts = []
  const_orig = tensor_util.constant_value(orig)
  const_or_orig = const_orig if const_orig is not None else orig
  prev_spatial_dim = 0
  i = 0
  while i < len(spatial_dims):
    start_i = i
    start_spatial_dim = spatial_dims[i]
    if start_spatial_dim > 1:
      # Fill in any gap from the previous spatial dimension (or dimension 1 if
      # this is the first spatial dimension) with `fill_value`.
      parts.append(
          np.full(
              [start_spatial_dim - 1 - prev_spatial_dim] + fill_dims,
              fill_value,
              dtype=dtype))
    # Find the largest value of i such that:
    #   [spatial_dims[start_i], ..., spatial_dims[i]]
    #     == [start_spatial_dim, ..., start_spatial_dim + i - start_i],
    # i.e. the end of a contiguous group of spatial dimensions.
    while (i + 1 < len(spatial_dims) and
           spatial_dims[i + 1] == spatial_dims[i] + 1):
      i += 1
    parts.append(const_or_orig[start_i:i + 1])
    prev_spatial_dim = spatial_dims[i]
    i += 1
  if const_orig is not None:
    return np.concatenate(parts)
  else:
    return array_ops.concat(parts, 0)


def _get_strides_and_dilation_rate(num_spatial_dims, strides, dilation_rate):
  """Helper function for verifying strides and dilation_rate arguments.
  This is used by `convolution` and `pool`.
  Args:
    num_spatial_dims: int
    strides: Optional.  List of N ints >= 1.  Defaults to [1]*N.  If any value
      of strides is > 1, then all values of dilation_rate must be 1.
    dilation_rate: Optional.  List of N ints >= 1.  Defaults to [1]*N.  If any
      value of dilation_rate is > 1, then all values of strides must be 1.
  Returns:
    Normalized (strides, dilation_rate) as int32 numpy arrays of shape
    [num_spatial_dims].
  Raises:
    ValueError: if the parameters are invalid.
  """
  if dilation_rate is None:
    dilation_rate = [1] * num_spatial_dims
  elif len(dilation_rate) != num_spatial_dims:
    raise ValueError("len(dilation_rate)=%d but should be %d" %
                     (len(dilation_rate), num_spatial_dims))
  dilation_rate = np.array(dilation_rate, dtype=np.int32)
  if np.any(dilation_rate < 1):
    raise ValueError("all values of dilation_rate must be positive")

  if strides is None:
    strides = [1] * num_spatial_dims
  elif len(strides) != num_spatial_dims:
    raise ValueError("len(strides)=%d but should be %d" % (len(strides),
                                                           num_spatial_dims))
  strides = np.array(strides, dtype=np.int32)
  if np.any(strides < 1):
    raise ValueError("all values of strides must be positive")

  if np.any(strides > 1) and np.any(dilation_rate > 1):
    raise ValueError(
        "strides > 1 not supported in conjunction with dilation_rate > 1")
  return strides, dilation_rate


# @tf_export("nn.convolution")
def convolution(
    input,  # pylint: disable=redefined-builtin
    filter,  # pylint: disable=redefined-builtin
    padding,
    strides=None,
    dilation_rate=None,
    quantized=False,
    quantization_params=True,
    name=None,
    data_format=None):
  # pylint: disable=line-too-long
  """Computes sums of N-D convolutions (actually cross-correlation).
  This also supports either output striding via the optional `strides` parameter
  or atrous convolution (also known as convolution with holes or dilated
  convolution, based on the French word "trous" meaning holes in English) via
  the optional `dilation_rate` parameter.  Currently, however, output striding
  is not supported for atrous convolutions.
  Specifically, in the case that `data_format` does not start with "NC", given
  a rank (N+2) `input` Tensor of shape
    [num_batches,
     input_spatial_shape[0],
     ...,
     input_spatial_shape[N-1],
     num_input_channels],
  a rank (N+2) `filter` Tensor of shape
    [spatial_filter_shape[0],
     ...,
     spatial_filter_shape[N-1],
     num_input_channels,
     num_output_channels],
  an optional `dilation_rate` tensor of shape [N] (defaulting to [1]*N)
  specifying the filter upsampling/input downsampling rate, and an optional list
  of N `strides` (defaulting [1]*N), this computes for each N-D spatial output
  position (x[0], ..., x[N-1]):
  ```
    output[b, x[0], ..., x[N-1], k] =
        sum_{z[0], ..., z[N-1], q}
            filter[z[0], ..., z[N-1], q, k] *
            padded_input[b,
                         x[0]*strides[0] + dilation_rate[0]*z[0],
                         ...,
                         x[N-1]*strides[N-1] + dilation_rate[N-1]*z[N-1],
                         q]
  ```
  where b is the index into the batch, k is the output channel number, q is the
  input channel number, and z is the N-D spatial offset within the filter. Here,
  `padded_input` is obtained by zero padding the input using an effective
  spatial filter shape of `(spatial_filter_shape-1) * dilation_rate + 1` and
  output striding `strides` as described in the
  [comment here](https://tensorflow.org/api_guides/python/nn#Convolution).
  In the case that `data_format` does start with `"NC"`, the `input` and output
  (but not the `filter`) are simply transposed as follows:
    convolution(input, data_format, **kwargs) =
      tf.transpose(convolution(tf.transpose(input, [0] + range(2,N+2) + [1]),
                               **kwargs),
                   [0, N+1] + range(1, N+1))
  It is required that 1 <= N <= 3.
  Args:
    input: An N-D `Tensor` of type `T`, of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    filter: An N-D `Tensor` with the same type as `input` and shape
      `spatial_filter_shape + [in_channels, out_channels]`.
    padding: A string, either `"VALID"` or `"SAME"`. The padding algorithm.
    strides: Optional.  Sequence of N ints >= 1.  Specifies the output stride.
      Defaults to [1]*N.  If any value of strides is > 1, then all values of
      dilation_rate must be 1.
    dilation_rate: Optional.  Sequence of N ints >= 1.  Specifies the filter
      upsampling/input downsampling rate.  In the literature, the same parameter
      is sometimes called `input stride` or `dilation`.  The effective filter
      size used for the convolution will be `spatial_filter_shape +
      (spatial_filter_shape - 1) * (rate - 1)`, obtained by inserting
      (dilation_rate[i]-1) zeros between consecutive elements of the original
      filter in each spatial dimension i.  If any value of dilation_rate is > 1,
      then all values of strides must be 1.
    name: Optional name for the returned tensor.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
  Returns:
    A `Tensor` with the same type as `input` of shape
        `[batch_size] + output_spatial_shape + [out_channels]`
    if data_format is None or does not start with "NC", or
        `[batch_size, out_channels] + output_spatial_shape`
    if data_format starts with "NC",
    where `output_spatial_shape` depends on the value of `padding`.
    If padding == "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding == "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] -
              (spatial_filter_shape[i]-1) * dilation_rate[i])
             / strides[i]).
  Raises:
    ValueError: If input/output depth does not match `filter` shape, if padding
      is other than `"VALID"` or `"SAME"`, or if data_format is invalid.
  """
  # pylint: enable=line-too-long
  with ops.name_scope(name, "convolution", [input, filter]) as name:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
   # input_shape = input.get_shape()
    if input.get_shape()[-1] is not None:
      input_shape = input.get_shape()[-1]
    else:
      input_shape = array_ops.shape(input)[-1]
    filter = ops.convert_to_tensor(filter, name="filter")  # pylint: disable=redefined-builtin
   # filter_shape = filter.get_shape()
    if filter.get_shape()[-1] is not None:
      filter_shape = filter.get_shape()[-1]
    else:
      filter_shape = array_ops.shape(filter)[-1]
    op = Convolution(
        input_shape,
        filter_shape,
        padding,
        strides=strides,
        dilation_rate=dilation_rate,
	quantized=quantized,
	quantization_params=quantization_params,
        name=name,
        data_format=data_format)
    return op(input, filter)


class Convolution(object):
  """Helper class for convolution.
  Note that this class assumes that shapes of input and filter passed to
  __call__ are compatible with input_shape and filter_shape passed to the
  constructor.
  Arguments
    input_shape: static shape of input. i.e. input.get_shape().
    filter_shape: static shape of the filter. i.e. filter.get_shape().
    padding:  see convolution.
    strides: see convolution.
    dilation_rate: see convolution.
    name: see convolution.
    data_format: see convolution.
  """

  def __init__(self,
               input_shape,
               filter_shape,
               padding,
               strides=None,
               dilation_rate=None,
               quantized=False,
               quantization_params=True,
               name=None,
               data_format=None):
    """Helper function for convolution."""
    print("+++++++++++3", quantized)
    num_total_dims = filter_shape.ndims
    if num_total_dims is None:
      num_total_dims = input_shape.ndims
    if num_total_dims is None:
      raise ValueError("rank of input or filter must be known")

    num_spatial_dims = num_total_dims - 2

    try:
      input_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("input tensor must have rank %d" % (num_spatial_dims + 2))

    try:
      filter_shape.with_rank(num_spatial_dims + 2)
    except ValueError:
      ValueError("filter tensor must have rank %d" % (num_spatial_dims + 2))

    if data_format is None or not data_format.startswith("NC"):
      input_channels_dim = input_shape[num_spatial_dims + 1]
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      input_channels_dim = input_shape[1]
      spatial_dims = range(2, num_spatial_dims + 2)

    if not input_channels_dim.is_compatible_with(
        filter_shape[num_spatial_dims]):
      raise ValueError(
          "number of input channels does not match corresponding dimension of "
          "filter, {} != {}".format(input_channels_dim,
                                    filter_shape[num_spatial_dims]))

    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)

    self.input_shape = input_shape
    self.filter_shape = filter_shape
    self.data_format = data_format
    self.strides = strides
    self.quantized = quantized
    self.quantization_params = quantization_params
    self.name = name
    self.conv_op = _WithSpaceToBatch(
        input_shape,
        dilation_rate=dilation_rate,
        padding=padding,
        build_op=self._build_op,
        filter_shape=filter_shape,
        spatial_dims=spatial_dims,
        data_format=data_format)

  def _build_op(self, _, padding):
    return _NonAtrousConvolution(
        self.input_shape,
        filter_shape=self.filter_shape,
        padding=padding,
        data_format=self.data_format,
        strides=self.strides,
	quantized=self.quantized,
	quantization_params=self.quantization_params,
        name=self.name)

  def __call__(self, inp, filter):  # pylint: disable=redefined-builtin
    return self.conv_op(inp, filter)
