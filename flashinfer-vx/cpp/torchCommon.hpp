/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Discard any no-conversion macros defined by TorchExt
// This will mess up TRTLLM compilation
#ifdef __CUDA_NO_HALF_OPERATORS__ 
#undef __CUDA_NO_HALF_OPERATORS__ 
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__ 
#undef __CUDA_NO_HALF2_OPERATORS__ 
#endif
#ifdef __CUDA_NO_FP4_CONVERSIONS__
#undef __CUDA_NO_FP4_CONVERSIONS__
#endif

// TorchExt headers
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// Checks
#define CHECK_CUDA(x) { TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); }
#define CHECK_STRIDE(x) { TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); }
