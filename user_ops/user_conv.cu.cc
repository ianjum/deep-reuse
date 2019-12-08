/* implementing the optimized Conv layer 
=================================================*/
#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include <string.h>
#include <map>
// #include <vector>
#include <cstdio>
#include <cmath>
#include <array>
#include "user_conv_2d.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>


using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;
namespace user {


// A simple array that contains data that can be passed between CPU and GPU.
template <typename T, int IndexCount, T DefaultValue>
struct Array {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator[](int index) const {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T& operator[](int index) {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array() {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0) {
    data[0] = a0;
    for (int i = 1; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0, T a1) {
    data[0] = a0;
    data[1] = a1;
    for (int i = 2; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0, T a1, T a2) {
    data[0] = a0;
    data[1] = a1;
    data[2] = a2;
    for (int i = 3; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_STRONG_INLINE Array(const std::array<T, IndexCount>& array) {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = array[i];
    }
  }
  T data[IndexCount];
};

// A dimension type with compile-time known size.
template <int IndexCount>
struct Dimension : Array<int, IndexCount, 0> {
  typedef Array<int, IndexCount, 0> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0) : Base(a0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0, int a1)
      : Base(a0, a1) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
  EIGEN_STRONG_INLINE Dimension(const std::array<int, IndexCount>& array)
      : Base(array) {}
};

template <typename T>
__global__ void MaxMinRangeKernel(int N, T* input, 
                                  Dimension<2> input_dims, T* output,
				  Dimension<2> output_dims) {
  CUDA_1D_KERNEL_LOOP(index, N) {
    int output_index = index;
    int input_index = index;
    int stride = input_dims[1];
    output[output_index] = input[input_index];
    output[output_index+stride] = input[input_index];
    for (int i=1; i<input_dims[0]; i++) {
      input_index += stride;
      if(input[input_index] > output[output_index]) 
        output[output_index] = input[input_index];
      if(input[input_index] < output[output_index+stride]) 
        output[output_index+stride] = input[input_index];
    }
    output[output_index+2*stride] = output[output_index] - output[output_index+stride];
  
  }
}

template <typename T>
__global__ void AssignClusterKernel(int N, T* input, Dimension<2> input_dims,
                                    T* maxminrange, int channel_size, 
				    int dim_per_channel,
				    int* output, Dimension<2> output_dims) {
  CUDA_1D_KERNEL_LOOP(index, N) {
    int output_index = index;
    int input_index = index * channel_size;
    int col = input_index % input_dims[1];
    int range_offset = 2*input_dims[1] + col;
    int min_offset = input_dims[1] + col;
    int max_offset = col;
    int range_index = 0;
    int min_index = 0;
    int max_index = 0;
    T interval = 0.0;
    T minvalue = 0.0;
    T maxvalue = 0.0;
    int id = 0;
    for (int i=0; i<channel_size; i++) {
      range_index = range_offset + i;
      min_index = min_offset + i;
      max_index = max_offset + i;
      interval = (maxminrange[range_index])/dim_per_channel;
      minvalue = (maxminrange[min_index]);
      maxvalue = (maxminrange[max_index]);
      id = id*dim_per_channel + std::floor((input[input_index+i]-minvalue)/interval);
    }
    output[output_index] = id;
  }
}

template <typename T>
__global__ void extractUniqueIDKernel(int* input, Dimension<2> input_dims, 
                          int* temp_checktable, int n_cluster,
			  int* output, Dimension<2> output_dims) {

  CUDA_1D_KERNEL_LOOP(index, input_dims[1]) {
    int col = index;
    int stride = input_dims[1];
    int table_id = index % 1000;
    
    int cluster_id = 0;
    int id = 0;
    for (int i=0; i<input_dims[0]; i++) {
      id = input[i*stride + col];
      if (temp_checktable[table_id+id] == 0) {
	input[i] = cluster_id;
      }
    }
  }
}

template <typename T>
__global__ void remapInputKernel(const T* input, Dimension<2> input_dims,
                                 T* output, Dimension<2> output_dims,
				 int Ntot, int K, int param_L) {
  CUDA_1D_KERNEL_LOOP(index, Ntot) {
    int in_col_idx = index % K;
    int in_row_idx = (int)index / K;
    int subvec_ID = (int)in_col_idx / param_L;
    int subvec_col_idx = in_col_idx % param_L;
    int stride = input_dims[0] * param_L;
    int out_offset = subvec_ID * stride + in_row_idx * param_L + index % param_L;
    output[out_offset] = input[index];
  }
}

template <typename T>
__global__ void getIDsKernel(T* input, Dimension<2> input_dims,
                             int* output, int param_H, int n_matrices) {
  // each thread compute one ID
  CUDA_1D_KERNEL_LOOP(index, input_dims[1]*n_matrices) {
    output[index] = 0;
    int in_offset = param_H * (index - index % input_dims[1]) + index % input_dims[1];
    for (int i=in_offset; i<in_offset + param_H*input_dims[1]; i+=input_dims[1]) {
      if (input[i] > 0) {
        output[index] = (output[index] << 1) | 1;
      }
      else {
        output[index] << 1;
      }
    }
  }
  // TODO(lning): could try each thread read only one value and add to the ID
  //              need to use atomic instruction
}

template <typename T>
__global__ void getCentroidMatrixKernel(int* ID_vector, T* transformed_input,
					Dimension<2> centroid_dims,
					Dimension<2> count_dims,
					Dimension<2> input_dims,
					int param_L, int param_H, int N,
					T* CenMatrix, int* CenCount) {

  int N_count = count_dims[0]*count_dims[1];
  CUDA_1D_KERNEL_LOOP(index, N_count+N_count*param_L ) {
    if (index < N_count) {
      CenCount[index] = 0;
    }
    else {
      CenMatrix[index - N_count] = 0.0;
    }
  }
  __syncthreads();

  CUDA_1D_KERNEL_LOOP(index, (N*count_dims[0] + input_dims[0]*input_dims[1])) {
    if (index < N*count_dims[0]) {
      int cluster_ID = ID_vector[index];
      int in_subvecID = (int) index / N;
      int cen_offset = in_subvecID * count_dims[1] + cluster_ID;
      atomicAdd(&CenCount[cen_offset], 1);
    }
    else {
      int in_row_idx = (int)(index - N*count_dims[0]) / N;
      int in_col_idx = (index - N*count_dims[0]) % N;
      int in_subvecID = (int) in_row_idx / param_L;
      int cluster_ID = ID_vector[in_subvecID*N+in_col_idx];
      int cen_offset = in_row_idx * count_dims[1] + cluster_ID;
      atomicAdd(&CenMatrix[cen_offset], transformed_input[index - N*count_dims[0]]);
    }
  }
  __syncthreads();

  CUDA_1D_KERNEL_LOOP(index, centroid_dims[0]*centroid_dims[1]) {
    int cen_row_idx = (int) index / centroid_dims[1];
    int cen_col_idx = index % centroid_dims[1];
    int cen_subvecID = (int) cen_row_idx / param_L;
    int count_offset = cen_subvecID * centroid_dims[1] + cen_col_idx;
    if (CenCount[count_offset] > 0) {
      CenMatrix[index] /= CenCount[count_offset];
    }
  } 
}

template <typename T>
__global__ void reconstructionKernel(T* CenOutput, Dimension<2> CenOutput_dims,
                                     int* ID_vector, int N, int M, T* output) {
  CUDA_1D_KERNEL_LOOP(index, N*M) {
    output[index] = 0.0;
  }
  __syncthreads();

  CUDA_1D_KERNEL_LOOP(index, N*CenOutput_dims[0]) {
    int id_row_idx = (int)index / (N*M);
    int id_col_idx = index % N;
    int cluster_ID = ID_vector[id_row_idx * N + id_col_idx];
    int cen_row_idx = (int)index / N;
    int cen_offset = cen_row_idx * CenOutput_dims[1] + cluster_ID;
    int out_offset = index % (N * M);
    atomicAdd(&output[out_offset], CenOutput[cen_offset]);
  }
}

template <typename T>
struct MaxMinRange<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor input,
		  typename TTypes<T, 2>::Tensor maxminrange) {
    Dimension<2> input_dims;
    input_dims[0] = input.dimension(0)*input.dimension(1)*input.dimension(2);
    input_dims[1] = input.dimension(3);
    Dimension<2> output_dims;
    for (int i=0; i<2; ++i) {
      output_dims[i] = maxminrange.dimension(i);
    }
    CudaLaunchConfig config = GetCudaLaunchConfig(output_dims[1], d);
    
    MaxMinRangeKernel<T>
         <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
	    config.virtual_thread_count, input.data(), input_dims, 
	    maxminrange.data(), output_dims);
  }
};

template <typename T>
struct AssignCluster<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor input,
		  typename TTypes<T, 2>::Tensor maxminrange,
		  int channel_size, int dim_per_channel,
		  typename TTypes<int, 2>::Tensor clusterIDtable) {
    Dimension<2> input_dims;
    input_dims[0] = input.dimension(0)*input.dimension(1)*input.dimension(2);
    input_dims[1] = input.dimension(3);
    Dimension<2> output_dims;
    for (int i=0; i<2; ++i) {
      output_dims[i] = clusterIDtable.dimension(i);
    }
    CudaLaunchConfig config = GetCudaLaunchConfig(clusterIDtable.size(), d);

    AssignClusterKernel<T>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
	     config.virtual_thread_count, input.data(), input_dims, 
	     maxminrange.data(), channel_size, dim_per_channel,
	     clusterIDtable.data(), output_dims);

  }
};

template <typename T>
struct extractUniqueID<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<int, 2>::Tensor input,
		  int n_cluster,
                  typename TTypes<int, 2>::Tensor temp_checktable,
		  typename TTypes<int, 2>::Tensor output) {
    
    Dimension<2> input_dims;
    input_dims[0] = input.dimension(0);
    input_dims[1] = input.dimension(1);
    Dimension<2> output_dims;
    for (int i=0; i<2; ++i) {
      output_dims[i] = output.dimension(i);
    }
    extractUniqueIDKernel<T>
         <<<4, 250, 0, d.stream()>>>(
	    input.data(), input_dims, temp_checktable.data(), 
	    n_cluster, output.data(), output_dims);
  }
};

template <typename T>
struct remap_input<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<T,4>::ConstTensor input,
		  int param_H, int param_L,
		  typename TTypes<T,2>::Tensor output) {
  // input dimension is M * a * b * c with a * b * c = K 
  // n_matrices = K / param_L
  // reshape the input to output with dimension (M * n_matrices, param_L)
  Dimension<2> input_dims;
  input_dims[0] = input.dimension(0);
  input_dims[1] = input.dimension(1);
  input_dims[2] = input.dimension(2);
  input_dims[3] = input.dimension(3);
  int K = input_dims[1]*input_dims[2]*input_dims[3];
  int Ntot = input_dims[0]*K;  
  Dimension<2> output_dims;
  output_dims[0] = output.dimension(0);
  output_dims[1] = output.dimension(1);
  
  CudaLaunchConfig config = GetCudaLaunchConfig(Ntot, d);
  remapInputKernel<T>
       <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          input.data(), input_dims, output.data(), output_dims, Ntot, K, param_L);
  }
};

template <typename T>
struct getIDs<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d,
                  typename TTypes<T, 2>::Tensor ID_matrix, 
	          typename TTypes<int, 2>::Tensor ID_vector,
		  int param_H, int n_matrices) {
  
  Dimension<2> input_dims;
  input_dims[0] = ID_matrix.dimension(0);
  input_dims[1] = ID_matrix.dimension(1);

  CudaLaunchConfig config = GetCudaLaunchConfig(input_dims[1]*n_matrices, d);
  getIDsKernel<T>
       <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          ID_matrix.data(), input_dims, ID_vector.data(), param_H, n_matrices);
  }
};

template <typename T>
struct getCentroidMatrix<GPUDevice, T>{
  typedef GPUDevice Device;
  void operator()(const Device& d,
        typename TTypes<int, 2>::Tensor ID_vector,
        typename TTypes<T, 2>::Tensor transformed_input,
	int param_H, int param_L, int N,
	typename TTypes<T, 2>::Tensor CenMatrix, 
	typename TTypes<int, 2>::Tensor CenCount) {

    Dimension<2> centroid_dims;
    centroid_dims[0] = CenMatrix.dimension(0);
    centroid_dims[1] = CenMatrix.dimension(1);
    Dimension<2> count_dims;
    count_dims[0] = CenCount.dimension(0);
    count_dims[1] = CenCount.dimension(1);
    Dimension<2> input_dims;
    input_dims[0] = transformed_input.dimension(0);
    input_dims[1] = transformed_input.dimension(1);
    CudaLaunchConfig config = GetCudaLaunchConfig(input_dims[0] * input_dims[1], d);
    getCentroidMatrixKernel<T>
         <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
	    ID_vector.data(), transformed_input.data(), 
	    centroid_dims, count_dims, input_dims,
	    param_L, param_H, N, CenMatrix.data(), CenCount.data());
  }
};

template <typename T>
struct reconstruction<GPUDevice, T>{
  typedef GPUDevice Device;
  void operator()(const Device& d, 
                  typename TTypes<T, 2>::Tensor CenOutput,
                  typename TTypes<int, 2>::Tensor ID_vector,
		  int param_H, int N, int M,
		  typename TTypes<T, 4>::Tensor output) {
    Dimension<2> CenOutput_dims;
    CenOutput_dims[0] = CenOutput.dimension(0);
    CenOutput_dims[1] = CenOutput.dimension(1);
    CudaLaunchConfig config = GetCudaLaunchConfig(N * CenOutput_dims[0], d);
    reconstructionKernel<T>
         <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
	    CenOutput.data(), CenOutput_dims, ID_vector.data(), N, M, output.data());
  }
};
} // user
template struct user::PadInput<GPUDevice, float, int, 4>;              
// template struct user::PadInput<GPUDevice, Eigen::half, int, 4>;              
template struct user::im2col<GPUDevice, float>;
// template struct user::im2col<GPUDevice, Eigen::half>;
template struct user::remap_input<GPUDevice, float>;
// template struct user::remap_input<GPUDevice, Eigen::half>;
template struct user::getIDs<GPUDevice, float>;
// template struct user::getIDs<GPUDevice, Eigen::half>;
template struct user::getCentroidMatrix<GPUDevice, float>;
// template struct user::getCentroidMatrix<GPUDevice, Eigen::half>;
template struct user::reconstruction<GPUDevice, float>;
// template struct user::reconstruction<GPUDevice, Eigen::half>;
template struct user::TransformFilter<GPUDevice, float, int, 4>;
// template struct user::TransformFilter<GPUDevice, Eigen::half, int, 4>;
template struct user::MaxMinRange<GPUDevice, float>;
// template struct user::MaxMinRange<GPUDevice, Eigen::half>;
template struct user::AssignCluster<GPUDevice, float>;
// template struct user::AssignCluster<GPUDevice, Eigen::half>;
template struct user::extractUniqueID<GPUDevice, float>;
// template struct user::extractUniqueID<GPUDevice, Eigen::half>;

#endif // GOOGLE_CUDA

