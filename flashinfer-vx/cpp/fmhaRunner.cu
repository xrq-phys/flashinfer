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

#include "tvmFfiUtils.h"
#include "flashInferMetaInfoVx.h"
#include "flashinfer/trtllm/fmha/fmhaRunnerParams.h"
#include "flashinfer/logging.h"
#include "flashinfer/utils.cuh"
#include "cudaDriverUtils.cuh"
#include "sequencePreproc.hpp"

#include <bit>
#include <tuple>
#include <memory>
#include <algorithm>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>


namespace flashinfer_vx {

namespace impl {

class FmhaKernelStore {
  /// TODO: Use mDriver from TRTLLM
  struct KernelFunc {
    char mFuncName[256];
    CUmodule mCuModule = nullptr;
    CUfunction mCuFunction = nullptr;
  };
  static constexpr int numKernels =
      sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfosVx) / sizeof(tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfoVx);
  KernelFunc kernels[numKernels];
  static bool initialized;

  static
  Data_type convert_dtype_to_tllm(DLDataType dtype) {
    using namespace tensorrt_llm::kernels;
    uint32_t dtype_code = std::bit_cast<uint32_t>(dtype);
    if (dtype_code == DL_FP32_CODE) {
      return DATA_TYPE_FP32;
    }
    else if (dtype_code == DL_FP16_CODE) {
      return DATA_TYPE_FP16;
    }
    else if (dtype_code == DL_BF16_CODE) {
      return DATA_TYPE_BF16;
    }
    else if (dtype_code == DL_E4M3_CODE) {
      return DATA_TYPE_E4M3;
    }
    else {
      FLASHINFER_CHECK(false, "Type not supported for now");
    }
  }

public:
  struct FmhaKernelItem {
    tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfoVx const *mKernelMeta;
    CUfunction mFunc;
  };

  FmhaKernelItem find_kernel(
    TllmGenFmhaRunnerParams const &options,
    DLDataType dtypeQ,
    DLDataType dtypeKv,
    DLDataType dtypeO,
    int numEltsPerSageAttnBlkQ,
    int numEltsPerSageAttnBlkK,
    int numEltsPerSageAttnBlkP,
    int numEltsPerSageAttnBlkV,
    cudaDeviceProp const &deviceProps
  ) const {
    using namespace tensorrt_llm::kernels;
    int32_t smVer = deviceProps.major * 10 + deviceProps.minor;

    // Currently only one kernel would match so no need for heuristics.
    for (int i = 0; i < numKernels; ++i)
    {
      auto &kernelMetainfo = sTllmGenFmhaKernelMetaInfosVx[i];
      if (
        kernelMetainfo.mDataTypeQ == convert_dtype_to_tllm(dtypeQ) &&
        kernelMetainfo.mDataTypeKv == convert_dtype_to_tllm(dtypeKv) &&
        kernelMetainfo.mDataTypeO == convert_dtype_to_tllm(dtypeO) &&
        kernelMetainfo.mHeadDimQk == options.mHeadDimQk &&
        kernelMetainfo.mHeadDimV == options.mHeadDimV &&
        kernelMetainfo.mSM == smVer &&
        kernelMetainfo.mSharedMemBytes <= deviceProps.sharedMemPerMultiprocessor &&
        kernelMetainfo.mQkvLayout == int(options.mQkvLayout) &&
        kernelMetainfo.mMaskType == int(options.mMaskType) &&
        kernelMetainfo.mKernelType == int(options.mKernelType) &&
        kernelMetainfo.mTileScheduler == int(options.mTileScheduler) &&
        kernelMetainfo.mNumEltsPerSageAttnBlkQ == numEltsPerSageAttnBlkQ &&
        kernelMetainfo.mNumEltsPerSageAttnBlkK == numEltsPerSageAttnBlkK &&
        kernelMetainfo.mNumEltsPerSageAttnBlkP == numEltsPerSageAttnBlkP &&
        kernelMetainfo.mNumEltsPerSageAttnBlkV == numEltsPerSageAttnBlkV &&
        kernelMetainfo.mGroupsHeadsQ == (options.mNumHeadsQPerKv > 1) &&
        kernelMetainfo.mSparseMla == false
      ) {
        FmhaKernelItem kernelItem{&kernelMetainfo, kernels[i].mCuFunction};
        FLASHINFER_CHECK(kernelItem.mFunc != nullptr,
                         "A compatible kernel was found, but it was not loaded to the driver during init.\n"
                         "This is probably caused by a different device being used during kernel load & kernel exec.");
        return kernelItem;
      }
    }
    FLASHINFER_CHECK(false, "Available kernel not found");
  }

  FmhaKernelStore(FmhaKernelStore const&) = delete;
  FmhaKernelStore(FmhaKernelStore&&) = delete;
  FmhaKernelStore& operator=(FmhaKernelStore const&) = delete;
  FmhaKernelStore& operator=(FmhaKernelStore&&) = delete;

  FmhaKernelStore() {
    using namespace tensorrt_llm::kernels;

    // Obtain SM version for current device
    int deviceIndex;
    cudaDeviceProp deviceProps;
    FLASHINFER_CUDA_RUNTIME_CALL(cudaGetDevice(&deviceIndex));
    FLASHINFER_CUDA_RUNTIME_CALL(cudaGetDeviceProperties(&deviceProps, deviceIndex));
    int32_t smVer = deviceProps.major * 10 + deviceProps.minor;

    // Only allow one instance of this object
    FLASHINFER_CHECK(!initialized, "One process should construct FmhaKernelStore only once.");

    // Obtain kernels from cubin
    for (int i = 0; i < numKernels; ++i)
    {
      auto &kernelMeta = sTllmGenFmhaKernelMetaInfosVx[i];
      strcpy(kernels[i].mFuncName, kernelMeta.mFuncName);
      // Driver only recognizes image for the installed device
      if (smVer == kernelMeta.mSM) {
        FLASHINFER_CUDA_DRIVER_CALL(cuModuleLoadData(&kernels[i].mCuModule, kernelMeta.mCubin));
        FLASHINFER_CUDA_DRIVER_CALL(cuModuleGetFunction(&kernels[i].mCuFunction, kernels[i].mCuModule, kernelMeta.mFuncName));
        FLASHINFER_CUDA_DRIVER_CALL(cuFuncSetAttribute(kernels[i].mCuFunction,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes));
      }
    }

    // Set initialized flag to true
    initialized = true;
  }

  ~FmhaKernelStore() {
    for (int i = 0; i < numKernels; ++i) {
      if (kernels[i].mCuModule != nullptr) {
        FLASHINFER_CUDA_DRIVER_CALL(cuModuleUnload(kernels[i].mCuModule));
        kernels[i].mCuModule = nullptr;
        kernels[i].mCuFunction = nullptr;
      }
    }
    initialized = false;
  }
};

bool FmhaKernelStore::initialized = false;


void run_context_attn(
  FmhaKernelStore const &kernelStore,
  TensorView Otensor,
  TensorView Qtensor,
  TensorView Ktensor,
  TensorView Vtensor,
  TensorView seqLensQ,
  TensorView seqLensKv,
  TensorView scaleSoftmaxLog2,
  TensorView outputScale,
  TensorView sageAttnSfQ,
  TensorView sageAttnSfK,
  TensorView sageAttnSfP,
  TensorView sageAttnSfV,
  unsigned numToksPerSageAttnBlkQ,
  unsigned numToksPerSageAttnBlkK,
  unsigned numToksPerSageAttnBlkP,
  unsigned numToksPerSageAttnBlkV,
  float scaleQ,
  bool persistentScheduler,
  cudaStream_t stream
) {
  // Debug logging
  char logMessage[512];
  #define TLLM_LOG_DEBUG(...) { snprintf(logMessage, sizeof(logMessage), __VA_ARGS__); FLASHINFER_LOG_DEBUG(logMessage); }

  // Storage type
  CHECK_CUDA(Qtensor);
  CHECK_CUDA(Ktensor);
  CHECK_CUDA(Vtensor);
  CHECK_CUDA(Otensor);

  // Check input sizes
  TVM_FFI_ICHECK_EQ(Qtensor.size(3), Ktensor.size(3)) << "headDimQk mismatch";
  TVM_FFI_ICHECK_EQ(Vtensor.size(3), Otensor.size(3)) << "headDimV mismatch";
  TVM_FFI_ICHECK_EQ(Qtensor.size(2), Otensor.size(2)) << "numHeadsQ mismatch";
  TVM_FFI_ICHECK_EQ(Ktensor.size(2), Vtensor.size(2)) << "numHeadsKv mismatch";
  TVM_FFI_ICHECK_EQ(Ktensor.size(0), Qtensor.size(0)) << "batchSize mismatch (K vs Q)";
  TVM_FFI_ICHECK_EQ(Ktensor.size(0), Vtensor.size(0)) << "batchSize mismatch (K vs V)";
  TVM_FFI_ICHECK_EQ(Ktensor.size(0), Otensor.size(0)) << "batchSize mismatch (K vs O)";
  TVM_FFI_ICHECK_EQ(Ktensor.size(1), Vtensor.size(1)) << "maxSeqLenKv mismatch";

  // Tensor sizes
  int headDimQk   = Qtensor.size(3);
  int headDimV    = Vtensor.size(3);
  int numHeadsQ   = Qtensor.size(2);
  int numHeadsKv  = Ktensor.size(2);
  int batchSize   = Qtensor.size(0);
  int maxSeqLenQ  = Qtensor.size(1);
  int maxSeqLenKv = Ktensor.size(1);
  int numHeadsQPerKv = numHeadsQ / numHeadsKv; // GQA?

  /// TODO: Add softmaxMax/Sum

  // Preprocess sequence
  int* cumSeqLensQPtr;
  int* cumSeqLensKvPtr;
  int* seqLensQPtr = static_cast<int*>(seqLensQ.data_ptr());
  int* seqLensKvPtr = seqLensKv.numel() > 0 ? static_cast<int*>(seqLensKv.data_ptr()) : nullptr;
  int constSeqKv = seqLensKvPtr == nullptr;
  if (constSeqKv) {
    FLASHINFER_CUDA_RUNTIME_CALL(cudaMalloc(&seqLensKvPtr, batchSize * sizeof(int)));
  }
  FLASHINFER_CUDA_RUNTIME_CALL(cudaMalloc(&cumSeqLensQPtr, (batchSize + 1) * sizeof(int)));
  FLASHINFER_CUDA_RUNTIME_CALL(cudaMalloc(&cumSeqLensKvPtr, (batchSize + 1) * sizeof(int)));
  int sumOfSeqLensQ = 0, sumOfSeqLensKv = 0;
  /// FIXME: Verify: Is this filling consistent with varlen?
  FLASHINFER_CUDA_RUNTIME_CALL(sequencePreprocess(
      batchSize, maxSeqLenQ, maxSeqLenKv, constSeqKv,
      seqLensQPtr, seqLensKvPtr, cumSeqLensQPtr, cumSeqLensKvPtr,
      &sumOfSeqLensQ, &sumOfSeqLensKv, stream));

  // Options
  TllmGenFmhaRunnerParams options{};

  // Set options
  // Some are default options. Currently only context maskless separate-qkv is needed as a vision extension
  options.mQkvLayout = QkvLayout::SeparateQkv;
  options.mMaskType = TrtllmGenAttentionMaskType::Dense;
  options.mKernelType = FmhaKernelType::Context;
  options.mTileScheduler = persistentScheduler ? TileScheduler::Persistent : TileScheduler::Static;
  options.mMultiCtasKvMode = false;
  // options.mUseBlockSparseAttention = false; // [TRTLLM]
  options.mHeadDimQk = headDimQk;
  options.mHeadDimV = headDimV;
  // options.mHeadDimQkNope = 0; // [TRTLLM] MLA. Won't use
  options.mNumHeadsQ = numHeadsQ;
  options.mNumHeadsKv = numHeadsKv;
  options.mNumHeadsQPerKv = numHeadsQPerKv;
  options.mBatchSize = batchSize;
  options.mMaxSeqLenCacheKv = 0; // KV cache. Won't use
  options.mMaxSeqLenQ = maxSeqLenQ;
  options.mMaxSeqLenKv = maxSeqLenKv;
  options.mAttentionWindowSize = INT_MAX; // Sliding window. Won't use
  options.mChunkedAttentionSize = INT_MAX; // Sliding window. Won't use
  options.mSumOfSeqLensQ = sumOfSeqLensQ;
  options.mSumOfSeqLensKv = sumOfSeqLensKv;
  options.mMaxNumPagesPerSeqKv = 0; // Paged KV. Won't use
  options.mNumTokensPerPage = 0; // Paged KV. Won't use
  options.mNumPagesInMemPool = 0; // Paged KV. Won't use
  options.mMultiProcessorCount = 0;
  options.mSfStartTokenIdx = 0; // SF starts at zero index
  // options.mSparseMlaTopK = 0; // [TRTLLM] MLA. Won't use
  options.mScaleQ = scaleQ;
  options.mScaleOutput = 1.0; // [FlashInfer]: No additional output scaling
  options.outputScale = 1.0; // [FlashInfer]: This argument is meant to be kept default
  options.scaleSoftmaxLog2 = ::pow(headDimQk, -.5) / ::log(2.); // [FlashInfer]: This argument is meant to be kept default
  options.kStrideKeysValues = Ktensor.stride(1); // [FlashInfer]
  options.kStrideHeads = Ktensor.stride(2); // [FlashInfer]
  options.kStrideBatch = Ktensor.stride(0); // [FlashInfer]
  options.vStrideKeysValues = Vtensor.stride(1); // [FlashInfer]
  options.vStrideHeads = Vtensor.stride(2); // [FlashInfer]
  options.vStrideBatch = Vtensor.stride(0); // [FlashInfer]

  // Tensor buffers
  options.qPtr = Qtensor.data_ptr();
  options.kPtr = Ktensor.data_ptr();
  options.vPtr = Vtensor.data_ptr();
  options.seqLensKvPtr = seqLensKvPtr;
  options.cumSeqLensQPtr = cumSeqLensQPtr;
  options.cumSeqLensKvPtr = cumSeqLensKvPtr;
  options.scaleSoftmaxLog2Ptr = scaleSoftmaxLog2.numel() > 0 ? static_cast<float*>(scaleSoftmaxLog2.data_ptr()) : nullptr;
  options.outputScalePtr = outputScale.numel() > 0 ? static_cast<float*>(outputScale.data_ptr()) : nullptr;
  options.oPtr = Otensor.data_ptr();

  // Device props
  int deviceIndex;
  cudaDeviceProp deviceProps;
  FLASHINFER_CUDA_RUNTIME_CALL(cudaGetDevice(&deviceIndex));
  FLASHINFER_CUDA_RUNTIME_CALL(cudaGetDeviceProperties(&deviceProps, deviceIndex));
  options.mMultiProcessorCount = deviceProps.multiProcessorCount;
  options.stream = stream;

  // Find the kernel.
  auto kernelStoreItem = kernelStore.find_kernel(
    options,
    Qtensor.dtype(), Ktensor.dtype(), Otensor.dtype(),
    numToksPerSageAttnBlkQ,
    numToksPerSageAttnBlkK,
    numToksPerSageAttnBlkP,
    numToksPerSageAttnBlkV,
    deviceProps
  );

  // Found the kernel
  auto const &kernelMeta = *kernelStoreItem.mKernelMeta;

  // Compute Cta size
  // The number of Ctas per Q sequence.
  int numCtasPerSeqQ = (options.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
  // Take the number of headDim CTAs.
  int numHeadsPerCta = kernelMeta.mGroupsHeadsQ ? std::min(options.mNumHeadsQPerKv, kernelMeta.mMaxNumHeadsQPerKvInCta) : 1;
  int numCtasForAllHeadsQ = options.mNumHeadsQ / numHeadsPerCta;
  FLASHINFER_CHECK(numHeadsPerCta * numCtasForAllHeadsQ == options.mNumHeadsQ, "The numHeadsQ/numHeadsKv is not supported.");
  FLASHINFER_CHECK(kernelMeta.mHeadDimV == options.mHeadDimV, "We don't support multiple CTAs in headDimV yet.");
  int numCtasPerHeadDim = 1;
  // Compute the current numCtasX.
  int numCtasX = numCtasPerSeqQ;
  // Update the numCtasY.
  int numCtasY = numCtasForAllHeadsQ * numCtasPerHeadDim;
  // Compute the grid dimension Z.
  int numCtasZ = options.mBatchSize;

  // Consider the multiCtasKvMode for better GPU utilization.
  int numCtasPerSeqKv = 1;
  if (kernelMeta.mMultiCtasKvMode) {
    // The maximum attention window (the maximum number of tokensKv that will be attended to).
    int maxAttentionWindow{options.mMaxSeqLenKv};

    // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
    int const maxNumCtasPerSeqKv = (maxAttentionWindow + kernelMeta.mStepKv - 1) / kernelMeta.mStepKv;
    // Compute numCtasPerSeqKv.
    numCtasPerSeqKv = std::min(maxNumCtasPerSeqKv,
        std::max(1, int32_t(options.mMultiProcessorCount / (numCtasX * numCtasY * numCtasZ))));
    // Update the numCtasX.
    numCtasX *= numCtasPerSeqKv;
    // Add the debug info when multiCtasKvMode is enabled.
    if (numCtasPerSeqKv > 1)
    {
        TLLM_LOG_DEBUG(
            "TRTLLM-Gen launch info: multiCtasKvMode is enabled with tileSizeKv = %d, numCtasPerSeqKv = %d, "
            "numCtasPerSeqQ = "
            "%d, numCtasY = %d, numCtasZ = %d",
            kernelMeta.mTileSizeKv, numCtasPerSeqKv, numCtasPerSeqQ, numCtasY, numCtasZ);
    }
  }

  // The cluster size in the X dimension.
  int clusterDimX = kernelMeta.m2CtaMma ? 2 : 1;
  // Compute the current number of CTAs in total.
  int totalNumCtas = numCtasX * numCtasZ * numCtasY;

  // Get launch parameters
  auto kernelParams = KernelParams::setKernelParams(options, kernelMeta, numCtasPerSeqQ, numCtasPerSeqKv);

  // Write additional parameters for SageAttention
  if (numToksPerSageAttnBlkQ > 0) {
    /// TODO: Perform size check
    CHECK_CUDA(sageAttnSfQ);
    kernelParams.mLogNumEltsPerSageAttnBlkQ = std::bit_width(numToksPerSageAttnBlkQ) - 1;
    kernelParams.ptrSageAttnSfsQ = static_cast<float*>(sageAttnSfQ.data_ptr());
  }
  if (numToksPerSageAttnBlkK > 0) {
    CHECK_CUDA(sageAttnSfK);
    kernelParams.mLogNumEltsPerSageAttnBlkK = std::bit_width(numToksPerSageAttnBlkK) - 1;
    kernelParams.ptrSageAttnSfsK = static_cast<float*>(sageAttnSfK.data_ptr());
  }
  if (numToksPerSageAttnBlkP > 0) {
    CHECK_CUDA(sageAttnSfP);
    kernelParams.mLogNumEltsPerSageAttnBlkP = std::bit_width(numToksPerSageAttnBlkP) - 1;
    kernelParams.ptrSageAttnSfsP = static_cast<float*>(sageAttnSfP.data_ptr());
  }
  if (numToksPerSageAttnBlkV > 0) {
    CHECK_CUDA(sageAttnSfV);
    kernelParams.mLogNumEltsPerSageAttnBlkV = std::bit_width(numToksPerSageAttnBlkV) - 1;
    kernelParams.ptrSageAttnSfsV = static_cast<float*>(sageAttnSfV.data_ptr());
  }

  // Prepare kernel parameters list for cuLaunchKernelEx.
  void* kernelParamsList[] = {&kernelParams};
  CUlaunchConfig launch_config;
  launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
  launch_config.blockDimY = 1;
  launch_config.blockDimZ = 1;
  launch_config.gridDimX = numCtasX;
  launch_config.gridDimY = numCtasY;
  launch_config.gridDimZ = numCtasZ;
  launch_config.hStream = options.stream;
  launch_config.sharedMemBytes = kernelMeta.mSharedMemBytes;

  // Debug info.
  TLLM_LOG_DEBUG("TRTLLM-Gen launch info: kernelName = %s", kernelMeta.mFuncName);
  TLLM_LOG_DEBUG(
      "TRTLLM-Gen launch info: maxSeqLenQ = %d, "
      "maxSeqLenKv = %d, "
      "numHeadsQ = %d, "
      "numHeadsKv = %d, batchSize = %d, kernelType = %d",
      options.mMaxSeqLenQ, options.mMaxSeqLenKv, options.mNumHeadsQ, options.mNumHeadsKv, options.mBatchSize,
      static_cast<int>(options.mKernelType));
  TLLM_LOG_DEBUG("TRTLLM-Gen launch info: numCtasX = %d, numCtasY = %d, numCtasZ = %d, clusterDimX = %d",
      numCtasX, numCtasY, numCtasZ, clusterDimX);

  FLASHINFER_CHECK(clusterDimX <= 2, "Currently we allow maximal cluster size of 2.");
  CUlaunchAttribute launch_attribute[3];
  launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  launch_attribute[0].value.clusterDim.x = clusterDimX;
  launch_attribute[0].value.clusterDim.y = 1;
  launch_attribute[0].value.clusterDim.z = 1;
  launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
  launch_attribute[1].value.clusterSchedulingPolicyPreference
      = clusterDimX > 1 ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 2;

  // Launch the kernel
  FLASHINFER_CUDA_DRIVER_CALL(cuLaunchKernelEx(&launch_config, kernelStoreItem.mFunc, kernelParamsList, nullptr));

  /// FIXME: Is it save to dealloc?
  if (constSeqKv) {
    FLASHINFER_CUDA_RUNTIME_CALL(cudaFree(seqLensKvPtr));
  }
  FLASHINFER_CUDA_RUNTIME_CALL(cudaFree(cumSeqLensQPtr));
  FLASHINFER_CUDA_RUNTIME_CALL(cudaFree(cumSeqLensKvPtr));
}

} // impl

int64_t load_kernels() {
  using namespace flashinfer_vx::impl;
  auto* kernel_store = new FmhaKernelStore();
  return reinterpret_cast<int64_t>(kernel_store);
}

void unload_kernels(int64_t kernel_store_handle) {
  using namespace flashinfer_vx::impl;
  auto* kernel_store = reinterpret_cast<FmhaKernelStore*>(kernel_store_handle);
  delete kernel_store;
}

void run_context_attn(
  int64_t kernel_store_handle,
  TensorView Otensor,
  TensorView Qtensor,
  TensorView Ktensor,
  TensorView Vtensor,
  TensorView seqLensQ,
  TensorView seqLensKv,
  TensorView scaleSoftmaxLog2,
  TensorView outputScale,
  TensorView sageAttnSfQ,
  TensorView sageAttnSfK,
  TensorView sageAttnSfP,
  TensorView sageAttnSfV,
  int64_t numToksPerSageAttnBlkQ,
  int64_t numToksPerSageAttnBlkK,
  int64_t numToksPerSageAttnBlkP,
  int64_t numToksPerSageAttnBlkV,
  double scaleQ,
  bool persistentScheduler,
  int64_t stream
) {
  using namespace flashinfer_vx::impl;
  auto* kernel_store = reinterpret_cast<FmhaKernelStore*>(kernel_store_handle);

  run_context_attn(
    *kernel_store,
    Otensor, Qtensor, Ktensor, Vtensor,
    seqLensQ, seqLensKv,
    scaleSoftmaxLog2, outputScale,
    sageAttnSfQ, sageAttnSfK, sageAttnSfP, sageAttnSfV,
    static_cast<unsigned>(numToksPerSageAttnBlkQ),
    static_cast<unsigned>(numToksPerSageAttnBlkK),
    static_cast<unsigned>(numToksPerSageAttnBlkP),
    static_cast<unsigned>(numToksPerSageAttnBlkV),
    static_cast<float>(scaleQ),
    persistentScheduler,
    reinterpret_cast<cudaStream_t>(stream)
  );
}

}  // namespace flashinfer_vx

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_vx_load_kernels, flashinfer_vx::load_kernels);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_vx_unload_kernels, flashinfer_vx::unload_kernels);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_vx_run_context_attn, flashinfer_vx::run_context_attn);
