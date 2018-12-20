#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cublas_v2.h>

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <cstdio>

extern THCState *state;

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename DType>
__device__ DType deformable_im2col_bilinear(const DType *bottom_data,
                                            const int data_width,
                                            const int height, const int width,
                                            DType h, DType w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (DType)h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (DType)w_low;
  } else {
    w_high = w_low + 1;
  }

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType v1 = bottom_data[h_low * data_width + w_low];
  DType v2 = bottom_data[h_low * data_width + w_high];
  DType v3 = bottom_data[h_high * data_width + w_low];
  DType v4 = bottom_data[h_high * data_width + w_high];
  DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename DType>
__device__ DType get_gradient_weight(DType argmax_h, DType argmax_w,
                                     const int h, const int w, const int height,
                                     const int width) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  argmax_h = max(argmax_h, (DType)0.0f);
  argmax_w = max(argmax_w, (DType)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (DType)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (DType)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}

template <typename DType>
__device__ DType get_coordinate_weight(DType argmax_h, DType argmax_w,
                                       const int height, const int width,
                                       const DType *im_data,
                                       const int data_width, const int bp_dir) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  if (argmax_h < 0)
    argmax_h = 0;
  if (argmax_w < 0)
    argmax_w = 0;

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (DType)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (DType)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;

  if (bp_dir == 0) {
    weight += -1 * (argmax_w_low + 1 - argmax_w) *
              im_data[argmax_h_low * data_width + argmax_w_low];
    weight += -1 * (argmax_w - argmax_w_low) *
              im_data[argmax_h_low * data_width + argmax_w_high];
    weight += (argmax_w_low + 1 - argmax_w) *
              im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_w - argmax_w_low) *
              im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    weight += -1 * (argmax_h_low + 1 - argmax_h) *
              im_data[argmax_h_low * data_width + argmax_w_low];
    weight += (argmax_h_low + 1 - argmax_h) *
              im_data[argmax_h_low * data_width + argmax_w_high];
    weight += -1 * (argmax_h - argmax_h_low) *
              im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_h - argmax_h_low) *
              im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename DType>
__global__ void deformable_im2col_gpu_kernel(
    const int n, const DType *data_im, const DType *data_offset,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, DType *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    DType *data_col_ptr =
        data_col + (c_col * height_col + h_col) * width_col + w_col;
    const DType *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const DType *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                                     kernel_h * kernel_w *
                                                     height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        DType val = static_cast<DType>(0);
        const DType h_im = h_in + i * dilation_h + offset_h;
        const DType w_im = w_in + j * dilation_w + offset_w;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          const DType map_h = i * dilation_h + offset_h;
          const DType map_w = j * dilation_w + offset_w;
          const int cur_height = height - h_in;
          const int cur_width = width - w_in;
          val = deformable_im2col_bilinear(data_im_ptr, width, cur_height,
                                           cur_width, map_h, map_w);
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

// template <typename DType>
// void deformable_im2col(cudaStream_t stream, const DType *data_im,
//                        const DType *data_offset, const int channels,
//                        const int height, const int width, const int ksize_h,
//                        const int ksize_w, const int pad_h, const int pad_w,
//                        const int stride_h, const int stride_w,
//                        const int dilation_h, const int dilation_w,
//                        const int deformable_group, DType *data_col) {
//   // We are going to launch channels * height_col * width_col kernels, each
//   // kernel responsible for copying a single-channel grid.
//   int height_col =
//       (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
//   int width_col =
//       (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
//   int num_kernels = channels * height_col * width_col;
//   int channel_per_deformable_group = channels / deformable_group;
//   // Launch
//   deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
//                                  stream>>>(
//       num_kernels, data_im, data_offset, height, width, ksize_h, ksize_w, pad_h,
//       pad_w, stride_h, stride_w, dilation_h, dilation_w,
//       channel_per_deformable_group, height_col, width_col, data_col);
//
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
//     // TODO(BZ) panic
//   }
// }

at::Tensor deformable_im2col(const at::Tensor& data_in,
                       const at::Tensor& data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;

  // create output tensor
  auto data_num = data_in.size(0);
  auto data_col = at::zeros({data_num, channels, height_col, width_col}, data_in.options());

  // get cuda stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Launch
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "deformable_im2col", [&] {
  deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels,
      data_in.contiguous().data<scalar_t>(),
      data_offset.contiguous().data<scalar_t>(),
      height, width, ksize_h, ksize_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col,
      data_col.data<scalar_t>());
    });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }

  return data_col;
}

// template void deformable_im2col<float>(
//     cudaStream_t stream, const float *data_im, const float *data_offset,
//     const int channels, const int height, const int width, const int ksize_h,
//     const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
//     const int stride_w, const int dilation_h, const int dilation_w,
//     const int deformable_group, float *data_col);

template <typename DType>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const DType *data_col, const DType *data_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, DType *grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const DType *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                                     kernel_h * kernel_w *
                                                     height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const DType offset_h = data_offset_ptr[data_offset_h_ptr];
    const DType offset_w = data_offset_ptr[data_offset_w_ptr];
    const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const DType cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              (c * height + cur_h + dy) * width + cur_w + dx;
          DType weight =
              get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy,
                                  cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

// template <typename DType>
// void deformable_col2im(cudaStream_t stream, const DType *data_col,
//                        const DType *data_offset, const int channels,
//                        const int height, const int width, const int ksize_h,
//                        const int ksize_w, const int pad_h, const int pad_w,
//                        const int stride_h, const int stride_w,
//                        const int dilation_h, const int dilation_w,
//                        const int deformable_group, DType *grad_im) {
//
//   int height_col =
//       (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
//   int width_col =
//       (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
//   int num_kernels = channels * ksize_h * ksize_w * height_col * width_col;
//   int channel_per_deformable_group = channels / deformable_group;
//   // To avoid involving atomic operations, we will launch one kernel per
//   // bottom dimension, and then in the kernel add up the top dimensions.
//   deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,
//                                  stream>>>(
//       num_kernels, data_col, data_offset, channels, height, width, ksize_h,
//       ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
//       channel_per_deformable_group, height_col, width_col, grad_im);
//
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
//     // TODO(BZ) panic
//   }
// }

at::Tensor deformable_col2im(const at::Tensor& data_col,
                       const at::Tensor& data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group) {

  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;

  // create output tensor
  auto data_num = data_col.size(0);
  auto grad_im = at::zeros({data_num, channels, height, width}, data_col.options());

  // get cuda stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  AT_DISPATCH_FLOATING_TYPES(data_col.type(), "deformable_col2im", [&] {
  deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels,
      data_col.contiguous().data<scalar_t>(),
      data_offset.contiguous().data<scalar_t>(),
      channels, height, width, ksize_h,
      ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col,
      grad_im.contiguous().data<scalar_t>());
    });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }

  return grad_im;
}

// template void deformable_col2im<float>(
//     cudaStream_t stream, const float *data_col, const float *data_offset,
//     const int channels, const int height, const int width, const int ksize_h,
//     const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
//     const int stride_w, const int dilation_h, const int dilation_w,
//     const int deformable_group, float *grad_im);

template <typename DType>
__global__ void deformable_col2im_coord_gpu_kernel(
    const int n, const DType *data_col, const DType *data_im,
    const DType *data_offset, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, DType *grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    DType val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = index / width_col / height_col;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const DType *data_col_ptr = data_col + deformable_group_index *
                                               channel_per_deformable_group *
                                               width_col * height_col;
    const DType *data_im_ptr =
        data_im + deformable_group_index * channel_per_deformable_group /
                      kernel_h / kernel_w * height * width;
    const DType *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                                     kernel_h * kernel_w *
                                                     height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos = ((col_c * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col) % kernel_w;
      int i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      DType inv_h = h_in + i * dilation_h + offset_h;
      DType inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h < 0 || inv_w < 0 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -1;
      }
      const DType weight = get_coordinate_weight(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

// template <typename DType>
// void deformable_col2im_coord(cudaStream_t stream, const DType *data_col,
//                              const DType *data_im, const DType *data_offset,
//                              const int channels, const int height,
//                              const int width, const int ksize_h,
//                              const int ksize_w, const int pad_h,
//                              const int pad_w, const int stride_h,
//                              const int stride_w, const int dilation_h,
//                              const int dilation_w, const int deformable_group,
//                              DType *grad_offset) {
//
//   int height_col =
//       (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
//   int width_col =
//       (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
//   int num_kernels =
//       height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;
//   int channel_per_deformable_group =
//       channels * ksize_h * ksize_w / deformable_group;
//   // To avoid involving atomic operations, we will launch one kernel per
//   // bottom dimension, and then in the kernel add up the top dimensions.
//   deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels),
//                                        CUDA_NUM_THREADS, 0, stream>>>(
//       num_kernels, data_col, data_im, data_offset, channels, height, width,
//       ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
//       dilation_w, channel_per_deformable_group, height_col, width_col,
//       grad_offset);
//
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
//     // TODO(BZ) panic
//   }
// }

at::Tensor deformable_col2im_coord(const at::Tensor& data_col,
                             const at::Tensor& data_im, const at::Tensor& data_offset,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int deformable_group) {

  int height_col =
      (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;
  int channel_per_deformable_group =
      channels * ksize_h * ksize_w / deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.

  // create output tensor
  auto data_num = data_col.size(0);
  auto grad_offset = at::zeros({data_num, 2 * ksize_h * ksize_w * deformable_group,
    height_col, width_col}, data_col.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(data_col.type(), "deformable_col2im_coord", [&] {
  deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels,
      data_col.contiguous().data<scalar_t>(),
      data_im.contiguous().data<scalar_t>(),
      data_offset.contiguous().data<scalar_t>(),
      channels, height, width,
      ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, channel_per_deformable_group, height_col, width_col,
      grad_offset.contiguous().data<scalar_t>());
  });
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }

  return grad_offset;
}

// template void
// deformable_col2im_coord(cudaStream_t stream, const float *data_col,
//                         const float *data_im, const float *data_offset,
//                         const int channels, const int height, const int width,
//                         const int ksize_h, const int ksize_w, const int pad_h,
//                         const int pad_w, const int stride_h, const int stride_w,
//                         const int dilation_h, const int dilation_w,
//                         const int deformable_group, float *grad_offset);


int deform_conv_forward_cuda(
  at::Tensor& input,
  at::Tensor& weight,
  at::Tensor& offset,
  at::Tensor& output,
  at::Tensor& columns,
  at::Tensor& ones,
  int kW, int kH, int dW, int dH, int padW, int padH,
  int dilationH, int dilationW, int deformable_group) {

  // THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, weight, offset,
  //                                        output, columns, ones));
  //
  // shape_check(state, input, offset, NULL, weight, kH, kW, dH, dW, padH, padW,
  //             dilationH, dilationW, deformable_group);
  //
  // input = THCudaTensor_newContiguous(state, input);
  // offset = THCudaTensor_newContiguous(state, offset);
  // weight = THCudaTensor_newContiguous(state, weight);

  // int batch = 1;
  // if (input->nDimension == 3) {
  //   // Force batch
  //   batch = 0;
  //
  //   THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
  //                         input->size[2]);
  //   THCudaTensor_resize4d(state, offset, 1, offset->size[0], offset->size[1],
  //                         offset->size[2]);
  // }

  // TODO: jw: find alternative code for the following two commands
  // THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, weight, offset,
  //                                        output, columns, ones));
  //
  // shape_check(state, input, offset, NULL, weight, kH, kW, dH, dW, padH, padW,
  //             dilationH, dilationW, deformable_group);

  // input = input.contiguous();
  // offset = offset.contiguous();
  // weight = weight.contiguous();

  int batch = 1;
  if (input.dim() == 3) {
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    offset = offset.view({1, offset.size(0), offset.size(1), offset.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  // bias = bias ? THCudaTensor_newContiguous(state, bias) : bias;
  // THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight,
  //                       outputWidth);
  //
  // THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
  //                       outputHeight * outputWidth);

  output.resize_({nOutputPlane, outputHeight, outputWidth});
  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  // if (ones->nDimension != 2 ||
  //     ones->size[0] * ones->size[1] < outputHeight * outputWidth) {
  //   THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
  //   THCudaTensor_fill(state, ones, 1);
  // }

  if (ones.dim() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones.resize_({outputHeight, outputWidth});
    ones.fill_(1);
  }

  // THCudaTensor *input_n = THCudaTensor_new(state);
  // THCudaTensor *offset_n = THCudaTensor_new(state);
  // THCudaTensor *output_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {

    // THCudaTensor_select(state, input_n, input, 0, elt);
    // THCudaTensor_select(state, offset_n, offset, 0, elt);
    // THCudaTensor_select(state, output_n, output, 0, elt);

    // long m_ = nOutputPlane;
    // long n_ = outputHeight * outputWidth;
    // long k_ = 1;

    // TODO(BZ) add bias term
    // if (bias) {
    //   THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
    //                    THCudaTensor_data(state, ones), k_,
    //                    THCudaTensor_data(state, bias), k_, 0.0f,
    //                    THCudaTensor_data(state, output_n), n_);
    // } else {
    //   THCudaTensor_zero(state, output_n);
    // }

    // THCudaTensor_zero(state, output_n);

    columns = deformable_im2col(input[elt], offset[elt],
        nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group);

    long m = nOutputPlane;
    long n = columns.size(1);
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     columns.contiguous().data<float>(), n,
                     weight.contiguous().data<float>(), k, 1.0f,
                     output[elt].contiguous().data<float>(), n);
  }

  // THCudaTensor_free(state, input_n);
  // THCudaTensor_free(state, offset_n);
  // THCudaTensor_free(state, output_n);

  // if (batch == 0) {
  //   THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight,
  //                         outputWidth);
  //   THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  //   THCudaTensor_resize3d(state, offset, offset->size[1], offset->size[2],
  //                         offset->size[3]);
  // }

  if (batch == 0) {
    output.resize_({nOutputPlane, outputHeight, outputWidth});
    input.resize_({nInputPlane, inputHeight, inputWidth});
    offset.resize_({offset.size(1), offset.size(2), offset.size(3)});
  }

  // THCudaTensor_free(state, input);
  // THCudaTensor_free(state, offset);
  // THCudaTensor_free(state, weight);
  // if (bias) THCudaTensor_free(state, bias);

  return 1;
}

int deform_conv_backward_input_cuda(
    at::Tensor& input,
    at::Tensor& offset,
    at::Tensor& gradOutput,
    at::Tensor& gradInput,
    at::Tensor& gradOffset,
    at::Tensor& weight,
    at::Tensor& columns,
    int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW, int deformable_group) {

  // THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, gradOutput, weight,
  //                                        offset, columns, gradInput));
  //
  // shape_check(state, input, offset, gradOutput, weight, kH, kW, dH, dW, padH,
  //             padW, dilationH, dilationW, deformable_group);

  // input = THCudaTensor_newContiguous(state, input);
  // offset = THCudaTensor_newContiguous(state, offset);
  // gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  // weight = THCudaTensor_newContiguous(state, weight);

  // input = input.contiguous();
  // offset = offset.contiguous();
  // gradOutput = gradOutput.contiguous();
  // weight = weight.contiguous();

  // int batch = 1;
  // if (input->nDimension == 3) {
  //   // Force batch
  //   batch = 0;
  //   THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
  //                         input->size[2]);
  //   THCudaTensor_resize4d(state, offset, 1, offset->size[0], offset->size[1],
  //                         offset->size[2]);
  //   THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0],
  //                         gradOutput->size[1], gradOutput->size[2]);
  // }

  int batch = 1;
  if (input.dim() == 3) {
    // Force batch
    batch = 0;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    offset.resize_({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  // THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight,
  //                       inputWidth);
  // THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
  //                       outputHeight * outputWidth);

  gradInput.resize_({batchSize, nInputPlane, inputHeight, inputWidth});
  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  // THCudaTensor *gradInput_n = THCudaTensor_new(state);
  // THCudaTensor *gradOffset_n = THCudaTensor_new(state);
  // THCudaTensor *input_n = THCudaTensor_new(state);
  // THCudaTensor *offset_n = THCudaTensor_new(state);
  // THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    // THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    // THCudaTensor_select(state, gradOffset_n, gradOffset, 0, elt);
    // THCudaTensor_select(state, input_n, input, 0, elt);
    // THCudaTensor_select(state, offset_n, offset, 0, elt);
    // THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    long m = nInputPlane * kW * kH;
    long n = columns.size(1);
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     gradOutput[elt].contiguous().data<float>(), n,
                     weight.contiguous().data<float>(), m, 0.0f,
                     columns.contiguous().data<float>(), n);

    gradOffset[elt] = deformable_col2im_coord(
        columns,
        input[elt],
        offset[elt],
        nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        dilationH, dilationW, deformable_group);

    gradInput[elt] = deformable_col2im(
        columns,
        offset[elt], nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group);
  }

  // THCudaTensor_free(state, gradInput_n);
  // THCudaTensor_free(state, gradOffset_n);
  // THCudaTensor_free(state, input_n);
  // THCudaTensor_free(state, offset_n);
  // THCudaTensor_free(state, gradOutput_n);

  // if (batch == 0) {
  //   THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
  //                         outputWidth);
  //   THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  //   THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight,
  //                         inputWidth);
  //   THCudaTensor_resize3d(state, offset, offset->size[1], offset->size[2],
  //                         offset->size[3]);
  //   THCudaTensor_resize3d(state, gradOffset, offset->size[1], offset->size[2],
  //                         offset->size[3]);
  // }

  if (batch == 0) {
    gradOutput.resize_({nOutputPlane, outputHeight, outputWidth});
    input.resize_({nInputPlane, inputHeight, inputWidth});
    gradInput.resize_({nInputPlane, inputHeight, inputWidth});
    offset.resize_({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset.resize_({offset.size(1), offset.size(2), offset.size(3)});
  }

  // THCudaTensor_free(state, input);
  // THCudaTensor_free(state, offset);
  // THCudaTensor_free(state, gradOutput);
  // THCudaTensor_free(state, weight);

  return 1;
}

int deform_conv_backward_parameters_cuda(
    at::Tensor& input,
    at::Tensor& offset,
    at::Tensor& gradOutput,
    at::Tensor& gradWeight, /*THCudaTensor *gradBias, */
    at::Tensor& columns,
    at::Tensor& ones,
    int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW, int deformable_group,
    float scale) {

  // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, offset, gradOutput,
  //                                        gradWeight, columns));
  //
  // shape_check(state, input, offset, gradOutput, gradWeight, kH, kW, dH, dW,
  //             padH, padW, dilationH, dilationW, deformable_group);

  // input = THCudaTensor_newContiguous(state, input);
  // offset = THCudaTensor_newContiguous(state, offset);
  // gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  // input = input.contiguous();
  // offset = offset.contiguous();
  // gradOutput = gradOutput.contiguous();

  // int batch = 1;
  // if (input->nDimension == 3) {
  //   // Force batch
  //   batch = 0;
  //   THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1],
  //                         input->size[2]);
  //   THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0],
  //                         gradOutput->size[1], gradOutput->size[2]);
  // }

  int batch = 1;
  if (input.dim() == 3) {
    // Force batch
    batch = 0;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  // THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
  //                       outputHeight * outputWidth);
  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  // THCudaTensor *input_n = THCudaTensor_new(state);
  // THCudaTensor *offset_n = THCudaTensor_new(state);
  // THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  for (int elt = 0; elt < batchSize; elt++) {
    // THCudaTensor_select(state, input_n, input, 0, elt);
    // THCudaTensor_select(state, offset_n, offset, 0, elt);
    // THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    columns = deformable_im2col(
        input[elt],
        offset[elt], nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group);

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns.size(1);

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     columns.contiguous().data<float>(), k,
                     gradOutput[elt].contiguous().data<float>(), k, 1.0f,
                     gradWeight.contiguous().data<float>(), n);
  }

  // THCudaTensor_free(state, input_n);
  // THCudaTensor_free(state, offset_n);
  // THCudaTensor_free(state, gradOutput_n);

  // if (batch == 0) {
  //   THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
  //                         outputWidth);
  //   THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  // }

  if (batch == 0) {
    gradOutput.resize_({nOutputPlane, outputHeight, outputWidth});
    input.resize_({nInputPlane, inputHeight, inputWidth});
  }

  // THCudaTensor_free(state, input);
  // THCudaTensor_free(state, offset);
  // THCudaTensor_free(state, gradOutput);
  return 1;
}
