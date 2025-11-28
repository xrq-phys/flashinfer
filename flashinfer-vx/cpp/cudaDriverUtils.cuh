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

#include <cuda.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>


#define FLASHINFER_CUDA_DRIVER_CALL(func, ...)                                             \
{                                                                                          \
  CUresult e = (func);                                                                     \
  if (e != CUDA_SUCCESS) {                                                                 \
    char const* errorName = nullptr;                                                       \
    char const* errorString = nullptr;                                                     \
    cuGetErrorName(e, &errorName);                                                         \
    cuGetErrorString(e, &errorString);                                                     \
    std::ostringstream oss;                                                                \
    oss << "CUDA Driver Error: " << errorName << ": " << errorString << " (" << e  << ") " \
        << __FILE__ << ": line " << __LINE__ << " at function " << STR(func);              \
    const std::string message = oss.str();                                                 \
    std::cerr << message << std::endl;                                                     \
    throw std::runtime_error(message);                                                     \
  }                                                                                        \
}

#define FLASHINFER_CUDA_RUNTIME_CALL(func, ...)                                            \
{                                                                                          \
  cudaError_t e = (func);                                                                  \
  if (e != cudaSuccess) {                                                                  \
    char const* errorName = cudaGetErrorName(e);                                           \
    char const* errorString = cudaGetErrorString(e);                                       \
    std::ostringstream oss;                                                                \
    oss << "CUDA Runtime Error: " << errorName << ": " << errorString << " (" << e << ") " \
        << __FILE__ << ": line " << __LINE__ << " at function " << STR(func);              \
    const std::string message = oss.str();                                                 \
    std::cerr << message << std::endl;                                                     \
    throw std::runtime_error(message);                                                     \
  }                                                                                        \
}

