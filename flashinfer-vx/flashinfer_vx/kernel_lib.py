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

import ctypes
from pathlib import Path
from typing import Tuple
import tvm_ffi as ffi

import torch
from torch.utils.dlpack import to_dlpack


def _load_extension():
    """Load the compiled CUDA extension."""

    # Find the shared library file
    package_dir = Path(__file__).parent
    so_files = list(package_dir.glob('flashinfer_vx_cubin*.so'))

    if not so_files:
        raise RuntimeError(
            f"Could not find flashinfer_vx_cubin.so in {package_dir}. "
            "The package may not have been built correctly."
        )
    so_file = so_files[0]

    # Load the handle to register TVM FFI functions
    ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)

    return so_file

class FmhaKernelLib:
    """Wrapper for the TVM FFI library."""
    _so_file = None
    _load_kernels_func = None
    _unload_kernels_func = None
    _run_context_attn_func = None

    @classmethod
    def _lazy_init_class(cls):
        # Load the extension
        if FmhaKernelLib._so_file is None:
            _so_file = _load_extension()
        # Get TVM FFI functions
        if FmhaKernelLib._load_kernels_func is None:
            _load_kernels_func = ffi.get_global_func("flashinfer_vx_load_kernels")
            _unload_kernels_func = ffi.get_global_func("flashinfer_vx_unload_kernels")
            _run_context_attn_func = ffi.get_global_func("flashinfer_vx_run_context_attn")

    def _lazy_init(self):
        FmhaKernelLib._lazy_init_class()
        if self.handle is None:
            self.handle = FmhaKernelLib._load_kernels_func()

    def __init__(self):
        self.handle = None

    def __del__(self):
        if self.handle is not None:
            FmhaKernelLib._unload_kernels_func(self.handle)

    def __call__(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        seqLensQ: torch.Tensor,
        seqLensKv: torch.Tensor,
        scaleSoftmaxLog2: torch.Tensor,
        outputScale: torch.Tensor,
        sageAttnSfQ: torch.Tensor,
        sageAttnSfK: torch.Tensor,
        sageAttnSfP: torch.Tensor,
        sageAttnSfV: torch.Tensor,
        numToksPerSageAttnBlkQ: int,
        numToksPerSageAttnBlkK: int,
        numToksPerSageAttnBlkP: int,
        numToksPerSageAttnBlkV: int,
        scaleQ: float,
        persistentScheduler: bool,
        stream: torch.cuda.Stream,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run context attention kernel."""

        # Lazy init
        self._lazy_init()

        # Output shapes
        batchSize, maxSeqLenQ, numHeadsQ, _ = Q.shape
        _, _, _, headDimV = V.shape

        # Output dtype
        if Q.dtype == torch.float8_e4m3fn and outputScale.numel() > 0:
            output_dtype = torch.float8_e4m3fn
        else:
            output_dtype = torch.bfloat16

        # Allocate output tensor
        O = torch.empty(batchSize, maxSeqLenQ, numHeadsQ, headDimV, dtype=output_dtype, device=Q.device)

        # Convert all tensors to dlpack
        O_dlpack = to_dlpack(O)
        Q_dlpack = to_dlpack(Q)
        K_dlpack = to_dlpack(K)
        V_dlpack = to_dlpack(V)
        seqLensQ_dlpack = to_dlpack(seqLensQ)
        seqLensKv_dlpack = to_dlpack(seqLensKv)
        scaleSoftmaxLog2_dlpack = to_dlpack(scaleSoftmaxLog2)
        outputScale_dlpack = to_dlpack(outputScale)
        sageAttnSfQ_dlpack = to_dlpack(sageAttnSfQ)
        sageAttnSfK_dlpack = to_dlpack(sageAttnSfK)
        sageAttnSfP_dlpack = to_dlpack(sageAttnSfP)
        sageAttnSfV_dlpack = to_dlpack(sageAttnSfV)

        # Call the TVM FFI function
        self._run_context_attn_func(
            self.handle,
            O_dlpack,
            Q_dlpack,
            K_dlpack,
            V_dlpack,
            seqLensQ_dlpack,
            seqLensKv_dlpack,
            scaleSoftmaxLog2_dlpack,
            outputScale_dlpack,
            sageAttnSfQ_dlpack,
            sageAttnSfK_dlpack,
            sageAttnSfP_dlpack,
            sageAttnSfV_dlpack,
            numToksPerSageAttnBlkQ,
            numToksPerSageAttnBlkK,
            numToksPerSageAttnBlkP,
            numToksPerSageAttnBlkV,
            float(scaleQ),
            bool(persistentScheduler),
            int(stream.cuda_stream)
        )

        # Return the results
        ## TODO: Support globalMax & globalSum
        return O, torch.empty(0), torch.empty(0)
