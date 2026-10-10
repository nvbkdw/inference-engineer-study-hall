"""Optional actual GPU copy check with and without probes (no PyTorch dependency)."""

import ctypes
import struct
import sys

import cutlass
import cutlass.cute as cute
from cuda.bindings import runtime as cuda

import cuteviz
from examples.inspect_copy import entry


def checked(result):
    if result[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) == 2 else result[1:]


def gemm_smoke(path):
    from examples.inspect_gemm import entry as gemm_entry

    values_a = [float(i % 7 - 3) for i in range(16 * 16)]
    values_b = [float(i % 5 - 2) for i in range(8 * 16)]
    host_a = ctypes.create_string_buffer(struct.pack("<256e", *values_a))
    host_b = ctypes.create_string_buffer(struct.pack("<128e", *values_b))
    host_c = (ctypes.c_float * 128)()
    pointers = []
    try:
        for host, length in [(host_a, 512), (host_b, 256), (host_c, 512)]:
            ptr = checked(cuda.cudaMalloc(length))
            pointers.append(ptr)
            checked(
                cuda.cudaMemcpy(
                    ptr, ctypes.addressof(host), length, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
            )
        args = [
            cute.runtime.make_ptr(dtype, ptr, cute.AddressSpace.gmem, assumed_align=16)
            for dtype, ptr in zip([cutlass.Float16, cutlass.Float16, cutlass.Float32], pointers)
        ]
        with cuteviz.capture(path):
            compiled = cute.compile(gemm_entry, *args)
        compiled(*args)
        checked(cuda.cudaDeviceSynchronize())
        checked(
            cuda.cudaMemcpy(
                ctypes.addressof(host_c),
                pointers[2],
                512,
                cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        expected = [
            sum(values_a[m * 16 + k] * values_b[n * 16 + k] for k in range(16))
            for m in range(16)
            for n in range(8)
        ]
        assert list(host_c) == expected
        print("GPU FP16 warp GEMM: all 128 FP32 outputs match independent host multiplication")
    finally:
        for ptr in pointers:
            checked(cuda.cudaFree(ptr))


if __name__ == "__main__":
    count = 64
    host = (ctypes.c_float * count)(*[float(i * 3 + 1) for i in range(count)])
    output = (ctypes.c_float * count)()
    src = checked(cuda.cudaMalloc(ctypes.sizeof(host)))
    dst = checked(cuda.cudaMalloc(ctypes.sizeof(host)))
    try:
        checked(
            cuda.cudaMemcpy(
                src,
                ctypes.addressof(host),
                ctypes.sizeof(host),
                cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )
        a = cute.runtime.make_ptr(cutlass.Float32, src, cute.AddressSpace.gmem, assumed_align=16)
        b = cute.runtime.make_ptr(cutlass.Float32, dst, cute.AddressSpace.gmem, assumed_align=16)
        for enabled in (False, True):
            if enabled:
                with cuteviz.capture(sys.argv[1]):
                    compiled = cute.compile(entry, a, b, enabled)
            else:
                compiled = cute.compile(entry, a, b, enabled)
            checked(cuda.cudaMemset(dst, 0, ctypes.sizeof(output)))
            compiled(a, b)
            checked(cuda.cudaDeviceSynchronize())
            checked(
                cuda.cudaMemcpy(
                    ctypes.addressof(output),
                    dst,
                    ctypes.sizeof(output),
                    cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            assert list(output) == list(host), (enabled, list(output))
        print(
            "GPU global → shared → global copy: identical results with probes enabled and disabled"
        )
    finally:
        checked(cuda.cudaFree(src))
        checked(cuda.cudaFree(dst))
    gemm_smoke(sys.argv[1])
