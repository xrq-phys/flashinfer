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

#pragma once

#include <bit>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "dlpack/dlpack.h"

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;


constexpr DLDataType DL_INT8 = DLDataType{kDLInt, 8, 1};
constexpr DLDataType DL_FP16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType DL_FP32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType DL_FP64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType DL_E4M3 = DLDataType{kDLFloat8_e4m3fn, 8, 1};
constexpr DLDataType DL_E5M2 = DLDataType{kDLFloat8_e5m2, 8, 1};
constexpr DLDataType DL_BF16 = DLDataType{kDLBfloat, 16, 1};

constexpr uint32_t DL_INT8_CODE = std::bit_cast<uint32_t>(DL_INT8);
constexpr uint32_t DL_FP16_CODE = std::bit_cast<uint32_t>(DL_FP16);
constexpr uint32_t DL_FP32_CODE = std::bit_cast<uint32_t>(DL_FP32);
constexpr uint32_t DL_FP64_CODE = std::bit_cast<uint32_t>(DL_FP64);
constexpr uint32_t DL_E4M3_CODE = std::bit_cast<uint32_t>(DL_E4M3);
constexpr uint32_t DL_E5M2_CODE = std::bit_cast<uint32_t>(DL_E5M2);
constexpr uint32_t DL_BF16_CODE = std::bit_cast<uint32_t>(DL_BF16);

#define CHECK_CUDA(x) \
  TVM_FFI_ICHECK_EQ(x.device().device_type, kDLCUDA) << #x " must be a CUDA tensor";
