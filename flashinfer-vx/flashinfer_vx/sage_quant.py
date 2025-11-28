"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _sage_quant_qk_kernel(
    QkInputPtr,
    QkQuantPtr,
    QkScalePtr,
    KMeanPtr,
    QkInputStrideB,
    QkQuantStrideB,
    QkScaleStrideB,
    KMeanStrideB,
    QkInputStrideSeq,
    QkQuantStrideSeq,
    QkInputStrideH,
    QkQuantStrideH,
    QkScaleStrideH,
    KMeanStrideH,
    seqLen,
    HEAD_DIM: tl.constexpr,
    K_SMOOTH: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
):
    headId = tl.program_id(1)
    batchId = tl.program_id(2)
    seqBlkId = tl.program_id(0)

    QkInputPtr = QkInputPtr + QkInputStrideB * batchId + QkInputStrideH * headId
    QkQuantPtr = QkQuantPtr + QkQuantStrideB * batchId + QkQuantStrideH * headId
    QkScalePtr = QkScalePtr + QkScaleStrideB * batchId + QkScaleStrideH * headId
    if K_SMOOTH:
        KMeanPtr = KMeanPtr + KMeanStrideB * batchId + KMeanStrideH * headId

    rangeD = tl.arange(0, HEAD_DIM)
    rangeS = seqBlkId * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    maskS = rangeS < seqLen
    QkInputG = QkInputPtr + rangeS[:, None] * QkInputStrideSeq + rangeD[None, :]
    QkQuantG = QkQuantPtr + rangeS[:, None] * QkQuantStrideSeq + rangeD[None, :]
    QkScaleG = QkScalePtr + seqBlkId
    if K_SMOOTH:
        KMeanG = KMeanPtr + rangeD[None, :]

    Qk = tl.load(QkInputG, mask=maskS[:, None], other=0.0)
    if K_SMOOTH:
        km = tl.load(KMeanG)
        Qk = Qk - km

    QkScale = tl.max(tl.abs(Qk)) / 448 + 1e-3
    QkQuant = (Qk / QkScale).to(tl.float8e4nv)
    tl.store(QkQuantG, QkQuant, mask=maskS[:, None])
    tl.store(QkScaleG, QkScale)

@torch.compile
def _k_mean_fn(k):
    return k.to(torch.float32).mean(dim=1)

@torch.compile
def _sage_quant_v_fn(v, hidden_block_size):
    v_sage_ch = v.flatten(-2).unflatten(-1, (-1, hidden_block_size))
    v_sfs = v_sage_ch.abs().amax(dim=0).amax(dim=0).amax(dim=-1).to(torch.float32) / 448 + 1e-5 # (single mode: hidden // sage)
    v_fp8 = v_sage_ch / v_sfs[None, None, :, None]
    v_fp8 = v_fp8.reshape(v.shape).to(torch.float8_e4m3fn)
    return v_fp8, v_sfs

def quantize(q, k, v,
             sage_block_size_q=64,
             sage_block_size_k=64,
             sage_block_size_v=1,
             smooth=True):

    # allocate quantization results
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)

    # optional K-smoothing
    if smooth:
        k_mean = _k_mean_fn(k) # B, H, D
    else:
        k_mean = torch.empty((0, 0, 0), device=k.device)

    # repartition Q & K for SFs
    num_sage_sfs_q = triton.cdiv(q.shape[1], sage_block_size_q)
    num_sage_sfs_k = triton.cdiv(k.shape[1], sage_block_size_k)
    q_sfs = torch.empty((q.shape[0], q.shape[2], num_sage_sfs_q), device=q.device, dtype=torch.float32)
    k_sfs = torch.empty((k.shape[0], k.shape[2], num_sage_sfs_k), device=k.device, dtype=torch.float32)

    # launch grid: sequence, 1, batch
    grid_q = [num_sage_sfs_q, q.shape[2], q.shape[0]]
    grid_k = [num_sage_sfs_k, k.shape[2], k.shape[0]]

    _sage_quant_qk_kernel[grid_q](
        q,
        q_fp8,
        q_sfs,
        None,
        q.stride(0),
        q_fp8.stride(0),
        q_sfs.stride(0),
        0,
        q.stride(1),
        q_fp8.stride(1),
        q.stride(2),
        q_fp8.stride(2),
        q_sfs.stride(1),
        0,
        q.shape[1],
        HEAD_DIM=q.shape[-1],
        K_SMOOTH=False,
        SEQ_BLOCK=sage_block_size_q,
    )

    _sage_quant_qk_kernel[grid_k](
        k,
        k_fp8,
        k_sfs,
        k_mean,
        k.stride(0),
        k_fp8.stride(0),
        k_sfs.stride(0),
        k_mean.stride(0),
        k.stride(1),
        k_fp8.stride(1),
        k.stride(2),
        k_fp8.stride(2),
        k_sfs.stride(1),
        k_mean.stride(1),
        k.shape[1],
        HEAD_DIM=k.shape[-1],
        K_SMOOTH=smooth,
        SEQ_BLOCK=sage_block_size_k,
    )

    v_fp8, v_sfs = _sage_quant_v_fn(v, sage_block_size_v)

    return q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs, k_mean
