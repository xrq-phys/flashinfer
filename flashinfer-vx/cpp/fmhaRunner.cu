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

#include "torchCommon.hpp"
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
  Data_type convert_dtype_to_tllm(caffe2::TypeMeta dtype) {
    using namespace tensorrt_llm::kernels;
    if (dtype == torch::kFloat32) {
      return DATA_TYPE_FP32;
    }
    else if (dtype == torch::kFloat16) {
      return DATA_TYPE_FP16;
    }
    else if (dtype == torch::kBFloat16) {
      return DATA_TYPE_BF16;
    }
    else if (dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e4m3fnuz) {
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
    caffe2::TypeMeta dtypeQ,
    caffe2::TypeMeta dtypeKv,
    caffe2::TypeMeta dtypeO,
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
    c10::DeviceIndex deviceIndex;
    cudaDeviceProp deviceProps;
    c10::cuda::GetDevice(&deviceIndex);
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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_context_attn(
  FmhaKernelStore const &kernelStore,
  torch::Tensor Qtensor,
  torch::Tensor Ktensor,
  torch::Tensor Vtensor,
  torch::Tensor seqLensQ,
  torch::Tensor seqLensKv,
  torch::Tensor scaleSoftmaxLog2,
  torch::Tensor outputScale,
  torch::Tensor sageAttnSfQ,
  torch::Tensor sageAttnSfK,
  torch::Tensor sageAttnSfP,
  torch::Tensor sageAttnSfV,
  unsigned numToksPerSageAttnBlkQ,
  unsigned numToksPerSageAttnBlkK,
  unsigned numToksPerSageAttnBlkP,
  unsigned numToksPerSageAttnBlkV,
  float scaleQ,
  bool persistentScheduler,
  bool debugSetup
) {
  // Debug logging
  char logMessage[512];
  #define TLLM_LOG_DEBUG(...) { snprintf(logMessage, sizeof(logMessage), __VA_ARGS__); FLASHINFER_LOG_DEBUG(logMessage); }
  if (debugSetup) {
    spdlog::set_level(spdlog::level::debug);
  } else {
    spdlog::set_level(spdlog::level::info);
  }

  // CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Storage type
  CHECK_CUDA(Qtensor);
  CHECK_CUDA(Ktensor);
  CHECK_CUDA(Vtensor);

  // Check input sizes
  TORCH_CHECK_EQ(Qtensor.size(3), Ktensor.size(3)); // headDimQk
  TORCH_CHECK_EQ(Ktensor.size(2), Vtensor.size(2)); // numHeadsKv
  TORCH_CHECK_EQ(Ktensor.size(0), Qtensor.size(0)); // batchSize
  TORCH_CHECK_EQ(Ktensor.size(0), Vtensor.size(0)); // batchSize
  TORCH_CHECK_EQ(Ktensor.size(1), Vtensor.size(1)); // maxSeqLenKv

  // Tensor sizes
  int headDimQk   = Qtensor.size(3);
  int headDimV    = Vtensor.size(3);
  int numHeadsQ   = Qtensor.size(2);
  int numHeadsKv  = Ktensor.size(2);
  int batchSize   = Qtensor.size(0);
  int maxSeqLenQ  = Qtensor.size(1);
  int maxSeqLenKv = Ktensor.size(1);
  int numHeadsQPerKv = numHeadsQ / numHeadsKv; // GQA?

  // Output datatype
  auto outputDtype = Qtensor.dtype();
  if (outputDtype == torch::kFloat8_e4m3fn && outputScale.numel() < 1) {
    outputDtype = torch::kBFloat16;
  }

  // Alloc output buffers
  /// TODO: Add softmaxMax/Sum
  auto tensorDevice = Qtensor.device();
  torch::Tensor Otensor;
  if (debugSetup) {
    Otensor = at::zeros({batchSize, maxSeqLenQ, numHeadsQ, headDimV}, torch::TensorOptions().dtype(outputDtype).device(tensorDevice));
  } else {
    Otensor = at::empty({batchSize, maxSeqLenQ, numHeadsQ, headDimV}, torch::TensorOptions().dtype(outputDtype).device(tensorDevice));
  }

  // Preprocess sequence
  auto seqTensorOpts = torch::TensorOptions().dtype(torch::kInt32).device(tensorDevice);
  int constSeqKv = seqLensKv.data_ptr() == nullptr;
  if (constSeqKv) {
    seqLensKv = at::empty({batchSize}, seqTensorOpts);
  }
  torch::Tensor cumSeqLensQ = at::empty({batchSize + 1}, seqTensorOpts);
  torch::Tensor cumSeqLensKv = at::empty({batchSize + 1}, seqTensorOpts);
  int sumOfSeqLensQ = 0, sumOfSeqLensKv = 0;
  /// FIXME: Verify: Is this filling consistent with varlen?
  FLASHINFER_CUDA_RUNTIME_CALL(sequencePreprocess(
      batchSize, maxSeqLenQ, maxSeqLenKv, constSeqKv,
      seqLensQ.data_ptr<int>(), seqLensKv.data_ptr<int>(),
      cumSeqLensQ.data_ptr<int>(), cumSeqLensKv.data_ptr<int>(),
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
  options.seqLensKvPtr = seqLensKv.numel() > 0 ? seqLensKv.data_ptr<int>() : nullptr;
  options.cumSeqLensQPtr = cumSeqLensQ.numel() > 0 ? cumSeqLensQ.data_ptr<int>() : nullptr;
  options.cumSeqLensKvPtr = cumSeqLensKv.numel() > 0 ? cumSeqLensKv.data_ptr<int>() : nullptr;
  options.scaleSoftmaxLog2Ptr = scaleSoftmaxLog2.numel() > 0 ? scaleSoftmaxLog2.data_ptr<float>() : nullptr;
  options.outputScalePtr = outputScale.numel() > 0 ? outputScale.data_ptr<float>() : nullptr;
  options.oPtr = Otensor.data_ptr();

  // Device props
  c10::DeviceIndex deviceIndex;
  cudaDeviceProp deviceProps;
  c10::cuda::GetDevice(&deviceIndex);
  FLASHINFER_CUDA_RUNTIME_CALL(cudaGetDeviceProperties(&deviceProps, deviceIndex));
  options.mMultiProcessorCount = deviceProps.multiProcessorCount;
  options.stream = stream;

  // Find the kernel.
  auto kernelStoreItem = kernelStore.find_kernel(
    options,
    Qtensor.dtype(), Ktensor.dtype(), outputDtype,
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
    kernelParams.ptrSageAttnSfsQ = sageAttnSfQ.data_ptr<float>();
  }
  if (numToksPerSageAttnBlkK > 0) {
    CHECK_CUDA(sageAttnSfK);
    kernelParams.mLogNumEltsPerSageAttnBlkK = std::bit_width(numToksPerSageAttnBlkK) - 1;
    kernelParams.ptrSageAttnSfsK = sageAttnSfK.data_ptr<float>();
  }
  if (numToksPerSageAttnBlkP > 0) {
    CHECK_CUDA(sageAttnSfP);
    kernelParams.mLogNumEltsPerSageAttnBlkP = std::bit_width(numToksPerSageAttnBlkP) - 1;
    kernelParams.ptrSageAttnSfsP = sageAttnSfP.data_ptr<float>();
  }
  if (numToksPerSageAttnBlkV > 0) {
    CHECK_CUDA(sageAttnSfV);
    kernelParams.mLogNumEltsPerSageAttnBlkV = std::bit_width(numToksPerSageAttnBlkV) - 1;
    kernelParams.ptrSageAttnSfsV = sageAttnSfV.data_ptr<float>();
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

  return std::make_tuple(Otensor, at::empty({0}), at::empty({0}));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<FmhaKernelStore>(m, "FmhaKernelStore", py::module_local());
  m.def("run_context_attn", &run_context_attn, "Run context attention");
  m.def("load_kernels", []() { return std::make_unique<FmhaKernelStore>(); }, "Load all kernels from the embedded cubin");
}
