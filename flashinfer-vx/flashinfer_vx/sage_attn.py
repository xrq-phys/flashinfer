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

from typing import Optional
import torch

from flashinfer_vx import sage_quant


class _FlashInferVxCubin:
    kernel_store = None
    runner_module = None

    @classmethod
    def load_module_lazy(cls):
        "Lazy-loading for the runner module"
        if _FlashInferVxCubin.runner_module is None:
            import flashinfer_vx.flashinfer_vx_cubin
            _FlashInferVxCubin.runner_module = flashinfer_vx.flashinfer_vx_cubin
        return _FlashInferVxCubin.runner_module

    @classmethod
    def load_kernels_lazy(cls):
        "Lazy-loading for the cubin kernels to the driver"
        runner_module = _FlashInferVxCubin.load_module_lazy()
        if _FlashInferVxCubin.kernel_store is None:
            _FlashInferVxCubin.kernel_store = runner_module.load_kernels()
        return _FlashInferVxCubin.kernel_store

def _check_attn_input(Q: torch.Tensor,
                      K: torch.Tensor,
                      V: torch.Tensor,
                      seqLensQ: Optional[torch.Tensor] = None,
                      seqLensKv: Optional[torch.Tensor] = None):

    batchSizeQ, maxSeqLenQ, numHeadsQ, nChannelQ = Q.shape
    batchSizeK, maxSeqLenK, numHeadsK, nChannelK = K.shape
    batchSizeV, maxSeqLenV, numHeadsV, nChannelV = V.shape

    if nChannelQ != nChannelK:
        raise RuntimeError(f"nChannelQk mismatch: {nChannelQ} != {nChannelK}")
    if (maxSeqLenK, numHeadsK) != (maxSeqLenV, numHeadsV):
        raise RuntimeError(f"seqKv/headKv mismatch: {numHeadsK} != {numHeadsV}")
    if (batchSizeQ, batchSizeK) != (batchSizeK, batchSizeV):
        raise RuntimeError(f"batchSizeQkv mismatch: {(batchSizeQ, batchSizeK)} != {(batchSizeK, batchSizeV)}")

    if seqLensQ is None:
        seqLensQ = torch.empty(0, dtype=torch.int32)
    if seqLensKv is None:
        seqLensKv = torch.empty(0, dtype=torch.int32)

    return seqLensQ, seqLensKv

def qattn(Q: torch.Tensor,
          K: torch.Tensor,
          V: torch.Tensor,
          seqLensQ: Optional[torch.Tensor] = None,
          seqLensKv: Optional[torch.Tensor] = None):

    seqLensQ, seqLensKv = _check_attn_input(Q, K, V, seqLensQ, seqLensKv)

    scaleSoftmaxLog2 = torch.empty(0, dtype=torch.float32, device=Q.device)
    outScale = torch.empty(0, dtype=torch.float32, device=Q.device)

    module = _FlashInferVxCubin()

    out, _, _ = module.load_module_lazy().run_context_attn(
        module.load_kernels_lazy(),
        Q.to(torch.float8_e4m3fn),
        K.to(torch.float8_e4m3fn),
        V.to(torch.float8_e4m3fn),
        seqLensQ,
        seqLensKv,
        scaleSoftmaxLog2,
        outScale,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        0, # SfBlkQ
        0, # SfBlkK
        0, # SfBlkP
        0, # SfBlkV
        1.0, # scaleQ
        False, # persistent
        False # debug
    )

    return out

def sageattn(Q: torch.Tensor,
             K: torch.Tensor,
             V: torch.Tensor,
             seqLensQ: Optional[torch.Tensor] = None,
             seqLensKv: Optional[torch.Tensor] = None,
             sage_block_size_q: int = 1,
             sage_block_size_k: int = 16,
             sage_block_size_v: int = 1,
             smooth: bool = True):

    seqLensQ, seqLensKv = _check_attn_input(Q, K, V, seqLensQ, seqLensKv)

    scaleSoftmaxLog2 = torch.empty(0, dtype=torch.float32, device=Q.device)
    outScale = torch.empty(0, dtype=torch.float32, device=Q.device)

    QFp8, KFp8, VFp8, SfQ, SfK, SfV, KMean = sage_quant.quantize(
        Q, K, V,
        sage_block_size_q=sage_block_size_q,
        sage_block_size_k=sage_block_size_k,
        sage_block_size_v=sage_block_size_v,
        smooth=smooth,
    )

    SfQ = SfQ.flatten(0, 1)
    SfK = SfK.flatten(0, 1)
    SfP = torch.empty(0)

    module = _FlashInferVxCubin()

    out, _, _ = module.load_module_lazy().run_context_attn(
        module.load_kernels_lazy(),
        QFp8,
        KFp8,
        VFp8,
        seqLensQ,
        seqLensKv,
        scaleSoftmaxLog2,
        outScale,
        SfQ,
        SfK,
        SfP,
        SfV,
        sage_block_size_q, # SfBlkQ
        sage_block_size_k, # SfBlkK
        0, # SfBlkP
        sage_block_size_v, # SfBlkV
        1.0, # scaleQ
        False, # persistent
        False # debug
    )

    return out
