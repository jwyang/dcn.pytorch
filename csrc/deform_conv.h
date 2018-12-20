// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

// #include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
int deform_conv_forward(at::Tensor& input,
                               at::Tensor& weight,
                               at::Tensor& offset,
                               at::Tensor& output,
                               at::Tensor& columns,
                               at::Tensor& ones,
                               const int kW,
                               const int kH,
                               const int dW,
                               const int dH,
                               const int padW,
                               const int padH,
                               const int dilationH,
                               const int dilationW,
                               const int deformable_group) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv_forward_cuda(input, weight, offset, output, columns, ones,
                                 kW, kH, dW, dH, padW, padH, dilationH, dilationW, deformable_group);

#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
  // return ROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

int deform_conv_backward_input(at::Tensor& input,
                                      at::Tensor& offset,
                                      at::Tensor& gradOutput,
                                      at::Tensor& gradInput,
                                      at::Tensor& gradOffset,
                                      at::Tensor& weight,
                                      at::Tensor& columns,
                                      const int kW,
                                      const int kH,
                                      const int dW,
                                      const int dH,
                                      const int padW,
                                      const int padH,
                                      const int dilationH,
                                      const int dilationW,
                                      const int deformable_group) {
  if (gradInput.type().is_cuda() && gradOffset.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv_backward_input_cuda(input, offset, gradOutput, gradInput, gradOffset, weight, columns,
      kW, kH, dW, dH, padW, padH, dilationH, dilationW, deformable_group);

#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

int deform_conv_backward_parameters(at::Tensor& input,
                                           at::Tensor& offset,
                                           at::Tensor& gradOutput,
                                           at::Tensor& gradWeight, /*THCudaTensor *gradBias, */
                                           at::Tensor& columns,
                                           at::Tensor& ones,
                                           const int kW,
                                           const int kH,
                                           const int dW,
                                           const int dH,
                                           const int padW,
                                           const int padH,
                                           const int dilationH,
                                           const int dilationW,
                                           const int deformable_group,
                                           const float scale) {
  if (gradWeight.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_conv_backward_parameters_cuda(input, offset, gradOutput, gradWeight, columns, ones,
      kW, kH, dW, dH, padW, padH, dilationH, dilationW, deformable_group, scale);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
