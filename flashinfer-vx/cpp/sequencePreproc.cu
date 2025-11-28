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

/// @brief Sequence preprocessor

#include "sequencePreproc.hpp"
#include "cudaDriverUtils.cuh"
#include "flashinfer/utils.cuh"

__global__ void sequencePreprocessKernel(
    int batchSize, int maxSeqLenQ, int maxSeqLenKv, int constSeqKv, int *seqLenQ, int *seqLenKv, int *cumSeqLenQ, int *cumSeqLenKv) {

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Populate cumSumLenQ
    if (seqLenQ != nullptr) {
      // Perform cumsum.
      int sum = 0;
      for (int i = 0; i < batchSize; ++i) {
        cumSeqLenQ[i] = sum;
        sum += seqLenQ[i];
      }
      cumSeqLenQ[batchSize] = sum;
    } else {
      // Populate scaled sequence
      for (int i = 0; i < batchSize + 1; ++i) {
        cumSeqLenQ[i] = i * maxSeqLenQ;
      }
    }

    // Optionally populate seqLenKv
    if (constSeqKv) {
      for (int i = 0; i < batchSize; ++i) {
        seqLenKv[i] = maxSeqLenKv;
      }
    }

    // Populate cumSumSeqKv
    int sum = 0;
    for (int i = 0; i < batchSize; ++i) {
      cumSeqLenKv[i] = sum;
      sum += seqLenKv[i];
    }
    cumSeqLenKv[batchSize] = sum;
  }
}

cudaError_t sequencePreprocess(int batchSize, int maxSeqLenQ, int maxSeqLenKv, int constSeqKv,
                               int *seqLenQ, int *seqLenKv, int *cumSeqLenQ, int *cumSeqLenKv,
                               int *hostSumOfSeqLensQ, int *hostSumOfSeqLensKv, cudaStream_t stream) {

  void* kernelArgs[] = {&batchSize, &maxSeqLenQ, &maxSeqLenKv, &constSeqKv, &seqLenQ, &seqLenKv, &cumSeqLenQ, &cumSeqLenKv};
  auto const kernelFunc = reinterpret_cast<const void*>(sequencePreprocessKernel);
  FLASHINFER_CUDA_RUNTIME_CALL(cudaLaunchKernel(kernelFunc, dim3{1}, dim3{32}, kernelArgs, 0, stream));

  if (seqLenQ == nullptr) {
    *hostSumOfSeqLensQ = batchSize * maxSeqLenQ;
  } else {
    FLASHINFER_CUDA_RUNTIME_CALL(cudaMemcpy(hostSumOfSeqLensQ, &cumSeqLenQ[batchSize], sizeof(int), cudaMemcpyDeviceToHost));
  }
  if (constSeqKv) {
    *hostSumOfSeqLensKv = batchSize * maxSeqLenKv;
  } else {
    FLASHINFER_CUDA_RUNTIME_CALL(cudaMemcpy(hostSumOfSeqLensKv, &cumSeqLenKv[batchSize], sizeof(int), cudaMemcpyDeviceToHost));
  }

  return cudaSuccess;
}
