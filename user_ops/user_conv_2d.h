/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef USER_CONV_2D_H_
#define USER_CONV_2D_H_

#include <cstdio>
// #include "../third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/padding.h"
// #include "tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
// #include "tensorflow/core/kernels/eigen_spatial_convolutions.h"
#include "tensorflow/core/util/tensor_format.h"

using namespace tensorflow; 

namespace user {
template <typename Device, typename T, typename IndexType, int NDIMS>
struct PadInput {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
		  const std::array<int, NDIMS -2>& padding_left,
		  const std::array<int, NDIMS -2>& padding_right,
		  typename TTypes<T, NDIMS, IndexType>::Tensor out,
		  TensorFormat format) {
  }
};      

template <typename Device, typename T>
struct im2col {
  void operator()(const Device& d,
                  typename TTypes<T, 4>::ConstTensor input,
                  int patch_rows, int patch_cols, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols,
                  const Eigen::PaddingType& padding,
//                   typename TTypes<T,4>::Tensor output) {
                  typename TTypes<T,2>::Tensor output) {
    // Need to swap row/col when calling Eigen, because our data is in
    // NHWC format while Eigen assumes NWHC format.
    // map the vector to 2d and transpose it.
    To32Bit(output).device(d) =
        To32Bit(input)
            .extract_image_patches(patch_cols, patch_rows, stride_cols,
                                   stride_rows, rate_cols, rate_rows, padding)
            .reshape(output.dimensions());
  }
};

template <typename Device, typename T>
struct remap_input{
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor transformed_input, 
	int param_H, int param_L, typename TTypes<T, 2>::Tensor remapped_input) {
  }
};

template <typename Device, typename T>
struct getIDs{
  void operator()(const Device& d, typename TTypes<T, 2>::Tensor ID_matrix, 
	typename TTypes<int, 2>::Tensor ID_vector, int param_H, int n_matrices) {
  }
};

template <typename Device, typename T>
struct getCentroidMatrix{
  void operator()(const Device& d, typename TTypes<int, 2>::Tensor ID_vector,
        typename TTypes<T, 2>::Tensor transformed_input, int param_H, int param_L, int N,
	typename TTypes<T, 2>::Tensor CenMatrix,
	typename TTypes<int, 2>::Tensor CenCount) {
  }
};

template <typename Device, typename T>
struct reconstruction{
  void operator()(const Device& d, typename TTypes<T, 2>::Tensor CenOutput,
        typename TTypes<int, 2>::Tensor ID_vector, int param_H,
	int N, int M, typename TTypes<T, 4>::Tensor output) {
  }
};

template <typename Device, typename T, typename IndexType, int NDIMS>
struct TransformFilter {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out) {
    // We want a 3, 2, 0, 1 shuffle. Merge the spatial dimensions together
    // to speed up the shuffle operation.
    Eigen::DSizes<IndexType, 3> merged_dims;
    merged_dims[0] = in.dimension(0);  // spatial dimensions
    for (int i = 1; i < NDIMS - 2; ++i) {
      merged_dims[0] *= in.dimension(i);
    }
    merged_dims[1] = in.dimension(NDIMS - 2);  // input filters
    merged_dims[2] = in.dimension(NDIMS - 1);  // output filters

    Eigen::DSizes<IndexType, NDIMS> expanded_dims;
    expanded_dims[0] = in.dimension(NDIMS - 1);  // output filters
    expanded_dims[1] = in.dimension(NDIMS - 2);  // input filters
    for (int i = 0; i < NDIMS; ++i) {            // spatial dimensions
      expanded_dims[i + 2] = in.dimension(i);
    }

    out.device(d) = in.reshape(merged_dims)
                        .shuffle(Eigen::DSizes<IndexType, 3>(2, 1, 0))
                        .reshape(expanded_dims);
  }
};

template <typename Device, typename T>
struct MaxMinRange {
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor input,
		  typename TTypes<T, 2>::Tensor maxminrange) {
//     int a = 1;
//     input;
//     maxminrange;
  }
};

template <typename Device, typename T>
struct AssignCluster {
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor input,
		  typename TTypes<T, 2>::Tensor maxminrange,
		  int channel_size, int dim_per_channel,
		  typename TTypes<int, 2>::Tensor clusterIDtable) {
  }
};

template <typename Device, typename T>
struct extractUniqueID {
  void operator()(const Device& d,
                  typename TTypes<int, 2>::Tensor input,
		  int n_cluster,
		  typename TTypes<int, 2>::Tensor temp_checktable,
		  typename TTypes<int, 2>::Tensor output) {
  }
};
}  // namespace user


#endif  // TENSORFLOW_KERNELS_CONV_2D_H_
