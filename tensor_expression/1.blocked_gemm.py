from concurrent.futures import thread
import tvm
from tvm import te
import numpy as np
import tvm.testing
import os

TASK = "gemm"
USE_MANUAL_CODE = False
_dtype = "float32"


def write_code(code, fname):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    with open(fname, "w") as f:
        f.write(code)


def test_gemm():
    # graph
    # M, K, N = 16384, 16384, 16384
    M, K, N = 4096, 4096, 4096
    A = te.placeholder((M, K), dtype=_dtype, name="A")
    B = te.placeholder((K, N), dtype=_dtype, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(
        A[i, k] * B[k, j], axis=k), name="C")

    # C = te.compute((M, N), lambda i, j: te.sum(
    #     A[k, j] * B[k, i], axis=k), name="C")

    # schedule
    s: tvm.te.Schedule = te.create_schedule(C.op)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/0.initial.py")

    # AA = s.cache_read(A, "shared", [C])
    # BB = s.cache_read(B, "shared", [C])
    # AL = s.cache_read(AA, "local", [C])
    # BL = s.cache_read(BB, "local", [C])

    # create a cache stage(store C in local memory before writing to global memory)
    CC = s.cache_write(C, "local")
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/1.cache.py")

    # grid_size 16384
    # block_size 256

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    block_h = 32
    block_w = 32

    bx, xi = s[C].split(C.op.axis[0], factor=(block_h))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/2.split_i.py")
    by, yi = s[C].split(C.op.axis[1], factor=(block_w))
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/3.split_j.py")

    # launch 128x128 thread blocks, and each threadblock processes (M/128)x(N/128) elements
    s[C].bind(bx, block_x)
    s[C].bind(by, block_y)
    s[C].reorder(bx, by, xi, yi)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/4.bind_block.py")

    s[C].bind(xi, thread_x)
    s[C].bind(yi, thread_y)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/5.bind_thread.py")

    s[CC].compute_at(s[C], yi)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/6.CC_compute_at.py")
    # s[AA].compute_at(s[C], yi)
    # write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/13.AA_compute_at_ko.cu")
    # s[BB].compute_at(s[C], yi)
    # write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/14.BB_compute_at_ko.cu")
    # s[AL].compute_at(s[C], yi)
    # s[BL].compute_at(s[C], yi)

    # s[AA].bind(AA.op.axis[0], thread_y)
    # s[BB].bind(BB.op.axis[0], thread_y)
    write_code(
        str(tvm.lower(s, [A, B, C], simple_mode=True)), "log/1.blocked_gemm/7.compute_at_done.py")

    device = "cuda"

    dev = tvm.device(device, 0)
    if not dev.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Device %s" % device)
    f = tvm.build(s, [A, B, C], device)
    write_code(f.imported_modules[0].get_source(),
               "log/1.blocked_gemm/1.blocked_gemm.cu")
    # launch the kernel.
    # n, m, l = nn, nn, nn
    a_np = np.random.uniform(size=(M, K)).astype(A.dtype)
    b_np = np.random.uniform(size=(K, N)).astype(B.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), dev)
    for i in range(2):
        f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-3)

    num_flops = 2 * M * K * N
    num_runs = 2
    timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("average time cost of %d runs = %g ms, %g GFLOPS." %
          (num_runs, t * 1e3, GFLOPS))


if __name__ == "__main__":
    test_gemm()
