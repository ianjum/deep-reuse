# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=g-short-docstring-punctuation
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

#FIXX: commenting out add_arg_scope since this is not needed
#from tensorflow.contrib.framework.python.ops import add_arg_scope
#FIXX: commenting out variables and replace variables.model_variable with tf.get_variable
#from tensorflow.contrib.framework.python.ops import variables
import tensorflow as tf
#FIXX: replace initializers.xavier_initializer() with tf.Variable(tf.initializers.GlorotUniform()(shape=inputs.get_shape()))
#from tensorflow.contrib.layers.python.layers import initializers
#FIXX: implement the utils functions here
#from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
import convolution as u_convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

# TODO(b/28426988): Replace legacy_* fns migrated from slim.
# TODO(b/28426988): Remove legacy_* when all uses have migrated to new API.
__all__ = [
    'avg_pool2d', 'avg_pool3d', 'batch_norm', 'bias_add', 'conv1d', 'conv2d',
    'conv3d', 'conv2d_in_plane', 'conv2d_transpose', 'conv3d_transpose',
    'convolution', 'convolution1d', 'convolution2d', 'convolution2d_in_plane',
    'convolution2d_transpose', 'convolution3d', 'convolution3d_transpose',
    'dense_to_sparse', 'dropout', 'elu', 'flatten', 'fully_connected', 'GDN',
    'gdn', 'images_to_sequence', 'layer_norm', 'linear', 'pool', 'max_pool2d',
    'max_pool3d', 'one_hot_encoding', 'relu', 'relu6', 'repeat',
    'scale_gradient', 'separable_conv2d', 'separable_convolution2d',
    'sequence_to_images', 'softmax', 'spatial_softmax', 'stack', 'unit_norm',
    'legacy_fully_connected', 'legacy_linear', 'legacy_relu', 'maxout'
]

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


# TODO(jbms): change `rate` parameter to `dilation_rate` for consistency with
# underlying op.
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer= None,
                #FIXX : initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
		        quantized=False,
		        quantization_params=None,
                scope=None,
                conv_dims=None):
  """Adds an N-D convolution followed by an optional batch_norm layer.

  It is required that 1 <= N <= 3.

  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
  provided (such as `batch_norm`), it is then applied. Otherwise, if
  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
  variable would be created and added the activations. Finally, if
  `activation_fn` is not `None`, it is applied to the activations as well.

  Performs atrous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.

  Args:
    inputs: A Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: Integer, the number of output filters.
    kernel_size: A sequence of N positive integers specifying the spatial
      dimensions of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: A sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: One of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    rate: A sequence of N positive integers specifying the dilation rate to use
      for atrous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.
    conv_dims: Optional convolution dimensionality, when set it would use the
      corresponding convolution (e.g. 2 for Conv 2D, 3 for Conv 3D, ..). When
      leaved to None it would select the convolution dimensionality based on
      the input rank (i.e. Conv ND, with N = input_rank - 2).

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If `data_format` is invalid.
    ValueError: Both 'rate' and `stride` are not uniformly 1.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = _build_variable_getter({
      'bias': 'biases',
      'kernel': 'weights'
  })

  with variable_scope.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims

    if conv_dims is not None and conv_dims + 2 != input_rank:
      raise ValueError('Convolution expects input with rank %d, got %d' %
                       (conv_dims + 2, input_rank))
    if input_rank == 3:
      layer_class = convolutional_layers.Convolution1D
    elif input_rank == 4:
      layer_class = u_convolutional_layers.Convolution2D
    elif input_rank == 5:
      layer_class = convolutional_layers.Convolution3D
    else:
      raise ValueError('Convolution not supported for input with rank',
                       input_rank)


    df = ('channels_first'
          if data_format and data_format.startswith('NC') else 'channels_last')
    layer = layer_class(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        data_format=df,
        dilation_rate=rate,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        #FIXX: replace weight_initializer as tf.Variable(tf.initializers.GlorotUniform()(shape=inputs.get_shape()))
        kernel_initializer=tf.Variable(tf.initializers.GlorotUniform()(shape=inputs.get_shape())),
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
	    quantized=quantized,
	    quantization_params=quantization_params,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return collect_named_outputs(outputs_collections, sc.name, outputs)

def _model_variable_getter(
    getter,
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    rename=None,
    use_resource=None,
    synchronization=tf_variables.VariableSynchronization.AUTO,
    aggregation=tf_variables.VariableAggregation.NONE,
    **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
    #FIXX : replace with tf.get_variable function
  # return tf.compat.v1.get_variable(
    return  model_variable(
          name,
          getter,
          shape=shape,
          dtype=dtypes.float32,
          initializer=initializer,
          regularizer=regularizer,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=partitioner,
          use_resource=use_resource,
          synchronization=synchronization,
          aggregation=aggregation)
  #   var = tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  #   return var

def model_variable(name,
                   getter,
                   shape=None,
                   dtype=dtypes.float32,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   collections=None,
                   caching_device=None,
                   device=None,
                   partitioner=None,
                   use_resource=None,
                   synchronization=tf_variables.VariableSynchronization.AUTO,
                   aggregation=tf_variables.VariableAggregation.NONE):

  collections = list(collections or [])
  collections += [ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES]
  var = variable(
      name,
      getter,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      device=device,
      partitioner=partitioner,
      use_resource=use_resource,
      synchronization=synchronization,
      aggregation=aggregation)
  return var

def variable(name,
             getter,
             shape=None,
             dtype=None,
             initializer=None,
             regularizer=None,
             trainable=True,
             collections=None,
             caching_device=None,
             device=None,
             partitioner=None,
             use_resource=None,
             synchronization=tf_variables.VariableSynchronization.AUTO,
             aggregation=tf_variables.VariableAggregation.NONE):
  collections = list(collections if collections is not None else
                     [ops.GraphKeys.GLOBAL_VARIABLES])

  # Remove duplicates
  collections = list(set(collections))
  if getter is not None:
     getter = functools.partial(
         tf_variables._make_getter(), reuse=variable_scope.get_variable_scope().reuse)
  with ops.device(device or ''):
    return getter(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation)


def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""
  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)
  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = get_variable_collections(collections_set,
                                               collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf_variables.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in ops.get_collection(collection):
        ops.add_to_collection(collection, var)

# Simple alias.
convolution2d = convolution
conv2d = convolution2d

#FIXX: adding the whole function implementation so we dont have to call unneessary packages
def collect_named_outputs(collections, alias, outputs):
    if alias[-1] == '/':
        alias = alias[:-1]
    outputs.alias = alias
    if collections:
        ops.add_to_collections(collections, outputs)
    return outputs

def get_variable_collections(variables_collections, name):
  if isinstance(variables_collections, dict):
    variable_collections = variables_collections.get(name, None)
  else:
    variable_collections = variables_collections
  return variable_collections
