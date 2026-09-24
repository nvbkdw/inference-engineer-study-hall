
import time
import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

# CuteDSL kernel compilation cache
COMPILED = {}
COMPILATIONS = []


def _tensor(x):
    return from_dlpack(x.detach()).mark_layout_dynamic(leading_dim=x.ndim - 1)


def _launch(name, fn, tensors, constants):
    args = tuple(_tensor(t) for t in tensors)
    # mark_layout_dynamic preserves ZERO strides as compile-time constants.
    # Broadcast positions [B,T] therefore have a different launch ABI from
    # an unbroadcast B=1 tensor. Never share those compiled argument packs.
    key = (
        name,
        tuple(
            (
                t.ndim,
                str(t.dtype),
                t.device.index,
                tuple(i for i, stride in enumerate(t.stride()) if stride == 0),
                tuple(i for i, stride in enumerate(t.stride()) if stride == 1),
            )
            for t in tensors
        ),
        constants,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(tensors[0].device).cuda_stream)
    if key not in COMPILED:
        begin = time.perf_counter()
        COMPILED[key] = cute.compile(fn, *args, *constants, stream)
        COMPILATIONS.append(
            dict(kernel=name, signature=str(key), seconds=time.perf_counter() - begin)
        )
    COMPILED[key](*args, stream)
    return COMPILED[key], args, stream

def _check_bf16(*tensors):
    if not tensors or any(
        t.device.type != "cuda" or t.dtype != torch.bfloat16 or t.stride(-1) != 1
        for t in tensors
    ):
        raise ValueError("CuTe kernels require CUDA BF16 tensors with unit last stride")
    if any(t.device != tensors[0].device for t in tensors):
        raise ValueError("All tensors must share a CUDA device")
    if any(any(s <= 0 for s in t.shape) for t in tensors):
        raise ValueError("Empty tensors are unsupported")