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

#include "flashinfer/trtllm/fmha/kernelParams.h"

namespace tensorrt_llm
{
namespace kernels
{
// clang-format off

#define TLLM_GEN_VERSION "070bf75d"
#ifndef EXCLUDE_SM_100
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin[];
extern unsigned char FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin[];
extern unsigned char FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin[];
#endif // EXCLUDE_SM_100

#ifndef EXCLUDE_SM_100
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len;
extern unsigned int FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len;
extern unsigned int FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len;
#endif // EXCLUDE_SM_100


struct TllmGenFmhaKernelMetaInfoVx
{
    Data_type mDataTypeQ;
    Data_type mDataTypeKv;
    Data_type mDataTypeO;
    int mTileSizeQ;
    int mTileSizeKv;
    int mStepQ;
    int mStepKv;
    int mHeadDimPerCtaV;
    int mHeadDimQk;
    int mHeadDimV;
    int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    int mSharedMemBytes;
    int mThreadsPerCTA;
    int mQkvLayout;
    int mNumTokensPerPage;
    int mMaskType;
    int mKernelType;
    int mMaxNumHeadsQPerKvInCta;
    int mTileScheduler;
    int mMultiCtasKvMode;
    int mNumEltsPerSageAttnBlkQ;
    int mNumEltsPerSageAttnBlkK;
    int mNumEltsPerSageAttnBlkP;
    int mNumEltsPerSageAttnBlkV;
    bool mGroupsHeadsQ;
    bool mReuseSmemKForV;
    bool m2CtaMma;
    bool mSparseMla;
    bool mSkipsSoftmaxWhenPossible;
    const char* sha256;
};

static const TllmGenFmhaKernelMetaInfoVx sTllmGenFmhaKernelMetaInfosVx[] = {
#ifndef EXCLUDE_SM_100
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 82256, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "4d70756f2685148cc649d48e159162f47c8e11e9297becbb21e6a670a8db929c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext", 84592, 512, 0, 0, 0, 0, 1, 1, 0, 1, 16, 0, 1, false, false, false, false, false, "1028546ed9fae60b0c7dc6633060f0a09e95eb606035b71cd4fa21faa1b5ac1a"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext", 84496, 512, 0, 0, 0, 0, 1, 0, 0, 1, 16, 0, 1, false, false, false, false, false, "113f4f25ff1993bb3e05e2150303a22f0c2735532b5545931effc7fbda96bd89"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext", 84448, 512, 0, 0, 0, 0, 1, 1, 0, 1, 64, 0, 1, false, false, false, false, false, "ef2a7a53cdd71910c3a37a243898e1975046b86cc4aabfb09a0743a9edd94b5c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext", 84352, 512, 0, 0, 0, 0, 1, 0, 0, 1, 64, 0, 1, false, false, false, false, false, "9d38402ea34834d1ccb86de6f9383ca771e65be3a159220503e702b11253fd30"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext", 84448, 512, 0, 0, 0, 0, 1, 1, 0, 64, 64, 0, 64, false, false, false, false, false, "f16a0a6c154d9260d7401beb900d42a50b9c8113ff9f7839d03f31aaf4968da7"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext", 84352, 512, 0, 0, 0, 0, 1, 0, 0, 64, 64, 0, 64, false, false, false, false, false, "5e57979c1f15841ccc3d47434a5e2a2ad1f069440bb63f1802d0a3caf3d26bf0"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128StaticContext", 82160, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "ba91d50b2f2b25452b6e70291d7fbed63f4eeeb1acf1f4c6871046b1dcb75530"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 213408, 384, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "f5b234f4c0e9fb96e8aed9b82f7eca3574fe778b17b6a5430f88f8606e08cd9c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext", 215232, 384, 0, 0, 0, 0, 1, 1, 0, 1, 16, 0, 1, false, false, false, false, false, "1511f93818dac3c8a5a1f0bf3f120ea196f7073d86c80b32b01d7fd6dc731bc6"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext", 215136, 384, 0, 0, 0, 0, 1, 0, 0, 1, 16, 0, 1, false, false, false, false, false, "98109e653e6646ed82e1f663f962c778ddaed7a60636bad99461d5546007243a"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext", 215088, 384, 0, 0, 0, 0, 1, 1, 0, 1, 64, 0, 1, false, false, false, false, false, "d9840199beaa248b1582a075d020116c47a01cceabf7bfaaccc8c94410e957dd"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext", 214992, 384, 0, 0, 0, 0, 1, 0, 0, 1, 64, 0, 1, false, false, false, false, false, "92548c98e08d7fd0a1ee070e5218d9b2db085b13f79e6cd41ff881eafd6e78d4"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext", 215088, 384, 0, 0, 0, 0, 1, 1, 0, 64, 64, 0, 64, false, false, false, false, false, "170e98ae93386be4e092a0615a0da1aa6d8e5641e26a569d53b79b389c12837b"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext", 214992, 384, 0, 0, 0, 0, 1, 0, 0, 64, 64, 0, 64, false, false, false, false, false, "ef95bf61737b1a75097cfc425b32bfc073cbb2f3943d7f41a4f799853f734111"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H256SeparateQkvDenseVarSeqQ128Kv128StaticContext", 213312, 384, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "7bde7435a45e722e750655bb87920b23ca1ac142044b682164842f0eb9054b4c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 41296, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "5f837c8f4b20aefc359b09b5ce47b7336dec06178a3523b1dd74215231cc03eb"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext", 43120, 512, 0, 0, 0, 0, 1, 1, 0, 1, 16, 0, 1, false, false, false, false, false, "11a82ccf8238981078855d207b475b0744a5dc38234aa2d13ac46bc6595c77e9"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext", 43024, 512, 0, 0, 0, 0, 1, 0, 0, 1, 16, 0, 1, false, false, false, false, false, "1bd4e15973d29a6d8acaf81196800e3589fefbb24c494716d4388508f0ef3691"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext", 42976, 512, 0, 0, 0, 0, 1, 1, 0, 1, 64, 0, 1, false, false, false, false, false, "94cf59ad24abf4debd96afc1cd69a39cff885009aa5805da99ebadc04d5138a5"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext", 42880, 512, 0, 0, 0, 0, 1, 0, 0, 1, 64, 0, 1, false, false, false, false, false, "20afaf4b90ffa37d1adea8bf78c4a2a0f411ce6080b75fb6f829a8b672d707ff"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext", 42976, 512, 0, 0, 0, 0, 1, 1, 0, 64, 64, 0, 64, false, false, false, false, false, "a3c980b5692f3a2112a8970cb79170c22ad55bd4efb3c24ba49b783d3d95b2d5"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext", 42880, 512, 0, 0, 0, 0, 1, 0, 0, 64, 64, 0, 64, false, false, false, false, false, "e0a8fce65fce2ed694508df45070b253f6556c12def07d94881d8775a4e3d73b"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OBfloat16H64SeparateQkvDenseVarSeqQ128Kv128StaticContext", 41200, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "55915c4059674ea45ac805fbea671acee747b7b59beb648be4f13a26a67aa37d"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 82256, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "cf43b66459cda16abe09f1abe9a38c222988dcaf6dc62f89d0ee443924dcea56"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext", 84592, 512, 0, 0, 0, 0, 1, 1, 0, 1, 16, 0, 1, false, false, false, false, false, "786b96939e1698b25a70de2c6bf1df6f15d06f21bc2b799df7c8fb681151efe5"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext", 84496, 512, 0, 0, 0, 0, 1, 0, 0, 1, 16, 0, 1, false, false, false, false, false, "c2c646b49898724588d941eecd60ac9050d8e74fdfd3335bdbe835ba0913f800"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext", 84448, 512, 0, 0, 0, 0, 1, 1, 0, 1, 64, 0, 1, false, false, false, false, false, "67ddbacb7dafed4d70a300d9f847f81712fe3fcc7949a26afd098c7ec3e083ba"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext", 84352, 512, 0, 0, 0, 0, 1, 0, 0, 1, 64, 0, 1, false, false, false, false, false, "52269935a690bf63e2231e0094f8796e594f47b2364e1f76d3ca55e3e4a46d8d"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext", 84448, 512, 0, 0, 0, 0, 1, 1, 0, 64, 64, 0, 64, false, false, false, false, false, "bea657af997ddc6f17b940330c89cd5b2e88ad0d49d3d0c7dac44bbfc58bd45e"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext", 84352, 512, 0, 0, 0, 0, 1, 0, 0, 64, 64, 0, 64, false, false, false, false, false, "8db7d0c72b27cf73efc3476b853631cbe021a0dbb326635144f20af4132710a4"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H128SeparateQkvDenseVarSeqQ128Kv128StaticContext", 82160, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "192b083e28a829caf1e2e402ec6d94ed37af389ac2c80baecb9dbeca1030b5b5"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 213408, 384, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "749e37827d66799643ee50938b5b64e46b794ccdbf918a087a7c7aba856a7164"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext", 215232, 384, 0, 0, 0, 0, 1, 1, 0, 1, 16, 0, 1, false, false, false, false, false, "5068f8785adae2cb8cb3e7d2147a91e8b5e61f46e005fe4288fad2181558da81"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext", 215136, 384, 0, 0, 0, 0, 1, 0, 0, 1, 16, 0, 1, false, false, false, false, false, "b70be7c5a124a1a767ffed58e9afbce6059e6867ffbd48611df1b4829402b47c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext", 215088, 384, 0, 0, 0, 0, 1, 1, 0, 1, 64, 0, 1, false, false, false, false, false, "03bfd80645b0c93e9f168f783e8834fc1675d328fa4e1b8f698cb990a05ffdef"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext", 214992, 384, 0, 0, 0, 0, 1, 0, 0, 1, 64, 0, 1, false, false, false, false, false, "25642da050f6fc7274c5a48264a304970845146882018f0f316be7c0bad884cd"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext", 215088, 384, 0, 0, 0, 0, 1, 1, 0, 64, 64, 0, 64, false, false, false, false, false, "6f1ccc655f244e0a84daf3677d438180ef6af49cbe9ae8878d6a04439a561536"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext", 214992, 384, 0, 0, 0, 0, 1, 0, 0, 64, 64, 0, 64, false, false, false, false, false, "1e79ccd1e732ecd69fedb38b1adc4a5d4cb5f61468467b188010b77dad6d58f7"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H256SeparateQkvDenseVarSeqQ128Kv128StaticContext", 213312, 384, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "5b6638ef2f1171faa3fff07142340f87b8f101372f8894ba8578a74eb913b912"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 41296, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "e4bcf315dd02d1a1e1c7b35d7d9107024debfd895484b6cc55cfaf23f92a1c7c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1PersistentContext", 43120, 512, 0, 0, 0, 0, 1, 1, 0, 1, 16, 0, 1, false, false, false, false, false, "c90bca91fee0024c6a89177ba78496332b10ec1d7393a79550f6f96223330e25"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK16SageV1StaticContext", 43024, 512, 0, 0, 0, 0, 1, 0, 0, 1, 16, 0, 1, false, false, false, false, false, "f8d58847f36a2f2858a2ba8e2630046aba3065a4a952dceab2de84ae12326377"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1PersistentContext", 42976, 512, 0, 0, 0, 0, 1, 1, 0, 1, 64, 0, 1, false, false, false, false, false, "a79b589e03fc5335f7b430f0c132c9eb9c8981eb34e1138f69bea289891ee7f1"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ1SageK64SageV1StaticContext", 42880, 512, 0, 0, 0, 0, 1, 0, 0, 1, 64, 0, 1, false, false, false, false, false, "4074deb550d4aa72fe54a6eb9a880de894b7f8911292bbd5ff41392ea239e6f1"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64PersistentContext", 42976, 512, 0, 0, 0, 0, 1, 1, 0, 64, 64, 0, 64, false, false, false, false, false, "b867a23a0ab0e44daeab9b939e5e6333658c6fbcb5bb77d48edd19558f49cdb0"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128SageQ64SageK64SageV64StaticContext", 42880, 512, 0, 0, 0, 0, 1, 0, 0, 64, 64, 0, 64, false, false, false, false, false, "324c1755499a8c819a3a179aa1ca0ef7534ed6736d99ae24e6c1617b0b31aaa9"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_100, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm100aKernel_QkvE4m3OE4m3H64SeparateQkvDenseVarSeqQ128Kv128StaticContext", 41200, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "53fbe575f343699a02881cf3d05acc0a343776aac0aefdbaca75b4b68d40c46f"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_103, FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 82256, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "3f7af89a5849e91142f81f6edc8df47f993bdf54da012ef9e82bb2d3a1be4f6e"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 128, 128, 128, kSM_103, FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OBfloat16H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext", 82160, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "7b90bc321cc2bf2906cc8c54cd6970fd3f1df691ef36d1cf0e744c06fcb285f4"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_103, FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 213408, 384, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "91989f2af007fd2365e984e69bb9fda0d70914c6241db629e60be87827c87b32"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 128, 128, 256, 256, 256, kSM_103, FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OBfloat16H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext", 213312, 384, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "786408aee15dad60afeb852194477593784dac7f5c1604b52a22e12c67daa0cf"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_103, FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 41296, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "6085ae2b202a6d303f4739022cf9ec9189bef2926c4bdeabb2059233785a29d0"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, 128, 128, 256, 128, 64, 64, 64, kSM_103, FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OBfloat16H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext", 41200, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "56d477852aeccdf45a54ad37884a8ef6b168838dd01cf8652db9b55a48098ad1"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_103, FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 82256, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "a5d20c50bf7cd9ab00b977b0c0072e884dd05e0fab60f36f24abbc7d32d161d4"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 128, 128, 128, kSM_103, FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OE4m3H128HVPerCta128SeparateQkvDenseVarSeqQ128Kv128StaticContext", 82160, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "fd695651ef8975546a03d4d87bd7c7db4260824a8603e6b3ef280e9067fcb24b"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_103, FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 213408, 384, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "b632bfbea9da5e59ccdf4467a9adbec0e110b395604244a690eac5ff2fd6cd53"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 128, 128, 256, 256, 256, kSM_103, FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OE4m3H256HVPerCta256SeparateQkvDenseVarSeqQ128Kv128StaticContext", 213312, 384, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "009862f4ba444beeb77f6cd5597717ebce6b12faf982b39b402d61baa8842ff9"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_103, FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin, FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128PersistentContext", 41296, 512, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, false, false, false, false, false, "f8d02fc46acd7a086c86b0cb762d6621dac579a1927d8a1ca66c9b8efd123e2c"},
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, 128, 128, 256, 128, 64, 64, 64, kSM_103, FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin, FmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext_cubin_len, "fmhaSm103aKernel_QkvE4m3OE4m3H64HVPerCta64SeparateQkvDenseVarSeqQ128Kv128StaticContext", 41200, 512, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, false, false, false, false, false, "f1045ebbd27220f75652ea283ce0ad62376573a4332236eaedc083b27b81429e"},
#endif // EXCLUDE_SM_100
};
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
