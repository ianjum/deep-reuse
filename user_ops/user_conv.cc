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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <string.h>
#include <map>
#include <vector>

#include "user_conv.h"
#include "user_conv_2d.h"
#include "conv_2d.h"
#include "ops_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
#include "tensorflow/core/kernels/xsmm_conv2d.h"
#endif

#if GOOGLE_CUDA
#include "conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("UserConv")
    .Input("input: T")
    .Input("filter: T")
//     .Input("quantization_params: int32") // when used in training, as Input
    .Output("output: T")
    .Attr("quantization_params: list(int)") // when used in inference, as Attr 
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
Computes a 2-D convolution given 4-D `input` and `filter` tensors.
Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`, this op
performs the following:
1. Flattens the filter to a 2-D matrix with shape
   `[filter_height * filter_width * in_channels, output_channels]`.
2. Extracts image patches from the input tensor to form a *virtual*
   tensor of shape `[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. For each patch, right-multiplies the filter matrix and the image patch
   vector.
In detail, with the default NHWC format,
    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]
Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
input: A 4-D tensor. The dimension order is interpreted according to the value
    of `data_format`, see below for details.
filter: A 4-D tensor of shape
    `[filter_height, filter_width, in_channels, out_channels]`
output: A 4-D tensor. The dimension order is determined by the value of
    `data_format`, see below for details.
strides: 1-D tensor of length 4.  The stride of the sliding window for each
  dimension of `input`. The dimension order is determined by the value of
    `data_format`, see below for details.
padding: The type of padding algorithm to use.
data_format: Specify the data format of the input and output data. With the

    default format "NHWC", the data is stored in the order of:
        [batch, height, width, channels].
    Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
)doc");

// CPU specialization of the convlution computation
// template <typename Device, typename T>
namespace {
template <typename Device, typename T>
struct LaunchUserConv {
  void operator()(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int row_stride, int col_stride,
                  int row_dilation, int col_dilation, const Padding& padding,
                  Tensor* output, TensorFormat data_format) {
    CHECK(data_format == FORMAT_NHWC) << "Generic conv implementation only "
                                         "supports NHWC tensor format for now.";
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && row_stride == 1 &&
        col_stride == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      //
      // TODO(vrv): We should be able to call SpatialConvolution
      // and it will produce the same result, but doing so
      // led to NaNs during training.  Using matmul instead for now.
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation == 1 &&
               col_dilation == 1 && padding == VALID) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair);
    } else {
      functor::SpatialConvolution<Device, T>()(
          ctx->eigen_device<Device>(), output->tensor<T, 4>(),
          input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride, col_stride,
          BrainPadding2EigenPadding(padding));
    }
  }
};
}  // namespace

template <typename T>
struct LaunchUserConvOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, 
                  std::vector<int> quan_params, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding, Tensor* output,
                  TensorFormat data_format) {
    if (data_format != FORMAT_NHWC) {
      ctx->SetStatus(
          errors::Unimplemented("Generic conv implementation only supports "
                                "NHWC tensor format for now."));
      return;
    }
    const int64 in_depth = GetTensorDim(input, data_format, 'C');
    OP_REQUIRES(ctx, in_depth == filter.dim_size(2),
                errors::Unimplemented("Generic conv implementation does not "
                                      "support grouped convolutions for now."));
    LaunchUserConv<CPUDevice, T>()(ctx, input, filter, row_stride, col_stride,
                                  row_dilation, col_dilation, padding, output,
                                  data_format);
  }
};


#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params) {
  TF_RETURN_IF_ERROR(context->GetAttr("quantization_params", &params->quant_params));
  TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
  TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));
  string data_format_string;
  TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));
  TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
              errors::InvalidArgument("Invalid data format"));

  const auto& strides = params->strides;
  const auto& dilations = params->dilations;
  const auto& data_format = params->data_format;

  TF_REQUIRES(dilations.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  TF_REQUIRES(strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64 stride_n = GetTensorDim(strides, data_format, 'N');
  const int64 stride_c = GetTensorDim(strides, data_format, 'C');
  const int64 stride_h = GetTensorDim(strides, data_format, 'H');
  const int64 stride_w = GetTensorDim(strides, data_format, 'W');
  TF_REQUIRES(
      stride_n == 1 && stride_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
  TF_REQUIRES(stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64 dilation_n = GetTensorDim(dilations, data_format, 'N');
  const int64 dilation_c = GetTensorDim(dilations, data_format, 'C');
  const int64 dilation_h = GetTensorDim(dilations, data_format, 'H');
  const int64 dilation_w = GetTensorDim(dilations, data_format, 'W');
  TF_REQUIRES(
      dilation_n == 1 && dilation_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "dilations in the batch and depth dimensions."));
  TF_REQUIRES(
      dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  return Status::OK();
}

Status ComputeConv2DDimension(const Conv2DParameters& params,
                              const Tensor& input, const Tensor& filter,
                              Conv2DDimensions* dimensions) {
  // Check that 2D convolution input and filter have exactly 4 dimensions.
  TF_REQUIRES(input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  TF_REQUIRES(filter.dims() == 4,
              errors::InvalidArgument("filter must be 4-dimensional: ",
                                      filter.shape().DebugString()));
  for (int i = 0; i < 3; i++) {
    TF_REQUIRES(
        FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
        errors::InvalidArgument("filter too large"));
  }

  // The last dimension for input is in_depth. Check that it is the same as the
  // filter's in_depth or it is evenly divisible by filter's in_depth.
  const int64 in_depth_raw = GetTensorDim(input, params.data_format, 'C');
  const int64 patch_depth_raw = filter.dim_size(2);
  TF_REQUIRES(FastBoundsCheck(in_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input depth too large"));
  TF_REQUIRES(FastBoundsCheck(patch_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Patch depth too large"));
  const int in_depth = static_cast<int>(in_depth_raw);
  const int patch_depth = static_cast<int>(patch_depth_raw);
  TF_REQUIRES(in_depth % patch_depth == 0,
              errors::InvalidArgument(
                  "input depth must be evenly divisible by filter depth: ",
                  in_depth, " vs ", patch_depth));

  // The last dimension for filter is out_depth.
  const int out_depth = static_cast<int>(filter.dim_size(3));

  // The second dimension for input is rows/height.
  // The first dimension for filter is rows/height.
  const int64 input_rows_raw = GetTensorDim(input, params.data_format, 'H');
  TF_REQUIRES(FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input rows too large"));
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(filter.dim_size(0));

  // The third dimension for input is columns/width.
  // The second dimension for filter is columns/width.
  const int64 input_cols_raw = GetTensorDim(input, params.data_format, 'W');
  TF_REQUIRES(FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input cols too large"));
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(filter.dim_size(1));

  // The first dimension for input is batch.
  const int64 batch_raw = GetTensorDim(input, params.data_format, 'N');
  TF_REQUIRES(FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("batch is too large"));
  const int batch = static_cast<int>(batch_raw);

  // Take the stride and dilation from the second and third dimensions only (we
  // do not support striding or dilation on the batch or depth dimension).
  const int stride_rows = GetTensorDim(params.strides, params.data_format, 'H');
  const int stride_cols = GetTensorDim(params.strides, params.data_format, 'W');
  const int dilation_rows =
      GetTensorDim(params.dilations, params.data_format, 'H');
  const int dilation_cols =
      GetTensorDim(params.dilations, params.data_format, 'W');

  // Compute windowed output sizes for rows and columns.
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
      input_rows, filter_rows, dilation_rows, stride_rows, params.padding,
      &out_rows, &pad_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
      input_cols, filter_cols, dilation_cols, stride_cols, params.padding,
      &out_cols, &pad_cols));

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->patch_depth = patch_depth;
  dimensions->out_depth = out_depth;
  dimensions->stride_rows = stride_rows;
  dimensions->stride_cols = stride_cols;
  dimensions->dilation_rows = dilation_rows;
  dimensions->dilation_cols = dilation_cols;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows = pad_rows;
  dimensions->pad_cols = pad_cols;

  return Status::OK();
}

#undef TF_REQUIRES

// OpKernel definition
template <typename Device, typename T>
// class UserConvOp : public BinaryOp<T> {1:
class UserConvOp : public OpKernel {
 public:
//   explicit UserConvOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
  explicit UserConvOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }
    launcher_(context, use_cudnn_, cudnn_use_autotune_, input,
              filter, params_.quant_params,
              dimensions.dilation_rows, dimensions.dilation_cols,
              dimensions.stride_rows, dimensions.stride_cols, 
	      params_.padding,
              output, params_.data_format);
  }

 private:
  Conv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;
  LaunchUserConvOp<Device, T> launcher_;

};

// Register the CPU kernels
#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("UserConv").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      UserConvOp<CPUDevice, T>);

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
#if !defined(USE_GEMM_FOR_CONV)
REGISTER_CPU(Eigen::half)
REGISTER_CPU(float)
#endif  // USE_GEMM_FOR_CONV

#if GOOGLE_CUDA
template <typename T>
void LaunchUserConvOp<GPUDevice, T>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter, 
    std::vector<int> quan_params, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    Tensor* output, TensorFormat data_format) {
  
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  Tensor input = input_param;
  const int64 in_batch = GetTensorDim(input, data_format, 'N');
  int64 in_rows = GetTensorDim(input, data_format, 'H');
  int64 in_cols = GetTensorDim(input, data_format, 'W');
  const int64 in_depths = GetTensorDim(input, data_format, 'C');
  const int64 patch_rows = filter.dim_size(0);
  const int64 patch_cols = filter.dim_size(1);
  const int64 patch_depths = filter.dim_size(2);

  int padding_rows = 0;
  int padding_cols = 0;
  const int64 out_batch = GetTensorDim(*output, data_format, 'N');
  const int64 out_rows = GetTensorDim(*output, data_format, 'H');
  const int64 out_cols = GetTensorDim(*output, data_format, 'W');
  const int64 out_depths = GetTensorDim(*output, data_format, 'C');
  
  int param_H = quan_params[0];
  int param_L = quan_params[1];
  const uint64 N = in_batch*out_rows*out_cols;
  const uint64 K = patch_rows*patch_cols*in_depths;
  const uint64 M = filter.dim_size(3);
  int n_matrices = K / param_L;
 
  Tensor transformed_input;
  OP_REQUIRES_OK(
         ctx, 
         ctx->allocate_temp(DataTypeToEnum<T>::value,
	             TensorShape({K, N}),
                     &transformed_input));
  
  user::im2col<GPUDevice, T>()(
        ctx->eigen_device<GPUDevice>(), input_param.tensor<T, 4>(), patch_rows,
        patch_cols, row_stride, col_stride, 1, 1,
        BrainPadding2EigenPadding(padding), transformed_input.tensor<T, 2>());

  Tensor hashing_matrix;
  OP_REQUIRES_OK(
        ctx,
	ctx->allocate_temp(DataTypeToEnum<T>::value,
	            TensorShape({param_H, param_L}),
		    &hashing_matrix));

  auto hashing_ptr = AsDeviceMemory(hashing_matrix.template flat<T>().data(),
                                    hashing_matrix.template flat<T>().size());
  stream ->ThenPopulateRandUniform(&hashing_ptr);

  Tensor ID_matrix;
  Tensor ID_vector;
  OP_REQUIRES_OK(
        ctx,
	ctx->allocate_temp(DataTypeToEnum<T>::value,
	            TensorShape({param_H*n_matrices, N}), 
		    &ID_matrix));
  OP_REQUIRES_OK(
        ctx,
	ctx->allocate_temp(DataTypeToEnum<int>::value,
	            TensorShape({n_matrices, N}), 
		    &ID_vector));
  
  auto input_ptr = AsDeviceMemory(transformed_input.template flat<T>().data(),
                                  transformed_input.template flat<T>().size());
  auto ID_ptr = AsDeviceMemory(ID_matrix.template flat<T>().data(),
                               ID_matrix.template flat<T>().size());
  auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;
  bool blas_launch_status =
      stream
        ->ThenBlasGemmStridedBatched(
	      no_transpose, no_transpose, param_H, N, param_L, 1.0f,
	      hashing_ptr, param_H, 0, input_ptr, param_L, param_L*N,
	      0.0f, &ID_ptr, param_H, param_H*N, n_matrices).ok();
  
  
  user::getIDs<GPUDevice, T>()(
        ctx->eigen_device<GPUDevice>(), ID_matrix.tensor<T, 2>(),
	ID_vector.tensor<int, 2>(), param_H, n_matrices);

  Tensor CenMatrix;
  Tensor CenCount;
  OP_REQUIRES_OK(
        ctx,
	ctx->allocate_temp(DataTypeToEnum<int>::value,
	            TensorShape({n_matrices, std::pow(2, param_H)}),&CenCount));
  OP_REQUIRES_OK(
        ctx,
	ctx->allocate_temp(DataTypeToEnum<T>::value,
	            TensorShape({n_matrices*param_L, std::pow(2, param_H)}),&CenMatrix));

  user::getCentroidMatrix<GPUDevice, T>()(
        ctx->eigen_device<GPUDevice>(), ID_vector.tensor<int, 2>(),
	transformed_input.tensor<T, 2>(), param_H, param_L, N, CenMatrix.tensor<T, 2>(),
	CenCount.tensor<int, 2>());

  Tensor remapped_filter;
  OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(DataTypeToEnum<T>::value,
	             TensorShape({M*n_matrices, param_L}),
                     &remapped_filter));

  user::remap_input<GPUDevice, T>()(
        ctx->eigen_device<GPUDevice>(), filter.tensor<T, 4>(),
	param_H, param_L, remapped_filter.tensor<T, 2>());
  
  Tensor CenOutput;
  OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(DataTypeToEnum<T>::value,
	             TensorShape({M*n_matrices, std::pow(2, param_H)}),
                     &CenOutput));
   
  const uint64 m =  N;
  const uint64 k = param_L;
  const uint64 n = M;
  const int64 stride_a = N * param_L;
  const int64 stride_b = M * param_L;
  const int64 stride_c = N * M;
  
  auto f_ptr = AsDeviceMemory(remapped_filter.template flat<T>().data(),
                              remapped_filter.template flat<T>().size());
  auto cen_ptr = AsDeviceMemory(CenMatrix.template flat<T>().data(),
                                CenMatrix.template flat<T>().size());
  auto out_ptr = AsDeviceMemory(CenOutput.template flat<T> ().data(),
                                CenOutput.template flat<T> ().size());
  blas_launch_status =
      stream
        ->ThenBlasGemmStridedBatched(
	      no_transpose, no_transpose, M, std::pow(2, param_H), param_L, 1.0f,
              f_ptr, M, M*param_L, cen_ptr, param_L, param_L*std::pow(2, param_H),
	      0.0f, &out_ptr, M, M*std::pow(2, param_H), n_matrices).ok();
  
  
  user::reconstruction<GPUDevice, T>()(
        ctx->eigen_device<GPUDevice>(), CenOutput.tensor<T, 2>(),
	ID_vector.tensor<int, 2>(), param_H, N, M, output->tensor<T, 4>());

  
  // test the blas version
  const uint64 mm =  N;
  const uint64 kk = K;
  const uint64 nn = M;
  input_ptr = AsDeviceMemory(transformed_input.template flat<T>().data(),
                             transformed_input.template flat<T>().size());
  auto filter_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                              filter.template flat<T>().size());
  auto output_ptr = AsDeviceMemory(output->template flat<T>().data(),
                              output->template flat<T>().size());
  no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;
  blas_launch_status = 
      stream
           ->ThenBlasGemm(no_transpose, no_transpose, nn, mm, kk, 1.0f, filter_ptr, nn,
	                  input_ptr, kk, 0.0f, &output_ptr, nn)
           .ok();
}

// Forward declarations of the functor specializations for GPU.

namespace user {
#define DECLARE_GPU_SPEC(T)                \
  template <>                                                                \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      typename TTypes<T, 4, int>::Tensor out);                               \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;              \
  template <>                                                                \
  void im2col<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,          \
      int patch_rows, int patch_cols, int stride_rows,                       \
      int stride_cols, int rate_rows, int rate_cols,                         \
      const Eigen::PaddingType& padding,                                     \
      typename TTypes<T, 2>::Tensor output);                                 \
  template <>                                                                \
  void remap_input<GPUDevice, T>::operator()(                                \
        const GPUDevice& d, typename TTypes<T, 4>::ConstTensor filter,            \
	int param_H, int param_L,                                            \
	typename TTypes<T, 2>::Tensor remapped_filter);                      \
  template <>                                                                \
  void getIDs<GPUDevice, T>::operator()(                                     \
        const GPUDevice& d, typename TTypes<T, 2>::Tensor ID_matrix,         \
	typename TTypes<int, 2>::Tensor ID_vector,                           \
	int param_H, int n_matrices);                                        \
  template <>                                                                \
  void getCentroidMatrix<GPUDevice, T>::operator()(                          \
        const GPUDevice& d, typename TTypes<int, 2>::Tensor ID_vector,       \
	typename TTypes<T, 2>::Tensor transformed_input,                     \
	int param_H, int param_L, int N,                                     \
	typename TTypes<T, 2>::Tensor CenMatrix,                             \
	typename TTypes<int, 2>::Tensor CenCount);                           \
  template <>                                                                \
  void reconstruction<GPUDevice, T>::operator()(                             \
        const GPUDevice& d, typename TTypes<T, 2>::Tensor CenOutput,         \
	typename TTypes<int, 2>::Tensor ID_vector,                           \
	int param_H, int N, int M,                                           \
	typename TTypes<T, 4>::Tensor output);                                \
  template <>                                                                \
  void MaxMinRange<GPUDevice, T>::operator()(                                \
      const GPUDevice& d,                                                    \
      typename TTypes<T, 4>::Tensor input,                                   \
      typename TTypes<T, 2>::Tensor maxminrange);                            \
  extern template struct MaxMinRange<GPUDevice, T>;                          \
  template <>                                                                \
  void AssignCluster<GPUDevice, T>::operator()(                              \
      const GPUDevice& d,                                                    \
      typename TTypes<T, 4>::Tensor input,                                   \
      typename TTypes<T, 2>::Tensor maxminrange,                             \
      int channel_size, int dim_per_channel,                                 \
      typename TTypes<int, 2>::Tensor clusterIDtable);                       \
  template <>                                                                \
  void extractUniqueID<GPUDevice, T>::operator()(                            \
       const GPUDevice& d,                                                   \
       typename TTypes<int, 2>::Tensor input,                                \
       int n_cluster,                                                        \
       typename TTypes<int, 2>::Tensor temp_checktable,                      \
       typename TTypes<int, 2>::Tensor output);                              \
  extern template struct extractUniqueID<GPUDevice, T>;                      \
  template <>                                                                \
  void PadInput<GPUDevice, T, int, 4>::operator()(                           \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      const std::array<int, 2>& padding_left,                                \
      const std::array<int, 2>& padding_right,                               \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format);     \
  extern template struct PadInput<GPUDevice, T, int, 4>

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
#undef DECLARE_GPU_SPEC
} //namespace user


// Registration of the GPU implementations.
// REGISTER_KERNEL_BUILDER(       
//       Name("UserConv").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),   
//       UserConvOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(       
      Name("UserConv").Device(DEVICE_GPU).TypeConstraint<float>("T"),   
      UserConvOp<GPUDevice, float>);

#endif  // GOOGLE_CUDA
