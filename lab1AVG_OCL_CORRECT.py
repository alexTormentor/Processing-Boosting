import numpy as np
import pyopencl as cl
import math
import time
from mpi4py import MPI

def f(a, b):
    res = 0.0
    for i in range(100):
        res += math.sqrt((a - i) ** 2 + (b + i) ** 2)
    return res

def add_serial(a, b):
    c = np.zeros_like(a)
    for i in range(len(a)):
        c[i] = f(a[i], b[i])
    return c


def add_numpy(a, b):
    vec_f = np.vectorize(f)
    return vec_f(a, b)

def add_with_opencl(a, b, group_size):
    platforms = cl.get_platforms()
    gpu_device = None
    for platform in platforms:
        devices = platform.get_devices(cl.device_type.GPU)
        if devices:
            gpu_device = devices[0]
            break

    if not gpu_device:
        raise RuntimeError("No GPU device found")

    context = cl.Context([gpu_device])
    queue = cl.CommandQueue(context, device=gpu_device)

    program_source = """
    __kernel void vecAdding(__global const float *a, __global const float *b, __global float *c) {
        int i = get_global_id(0);
        float res = 0.0;
        for (int j = 0; j < 100; j++) {
            res += sqrt((a[i] - j) * (a[i] - j) + (b[i] + j) * (b[i] + j));
        }
        c[i] = res;
    }
    """

    program = cl.Program(context, program_source).build()

    a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)

    kernel = program.vecAdding
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, c_buf)

    global_size = len(a)
    local_size = group_size

    cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (local_size,))
    c = np.empty_like(a)
    cl.enqueue_copy(queue, c, c_buf)

    return c


# MPI вычисление
def add_with_mpi(a, b, num_processes=8):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = num_processes  # Устанавливаем количество процессов вручную

    n = len(a)
    local_n = n // size
    start = rank * local_n
    end = (rank + 1) * local_n if rank != size - 1 else n

    local_a = a[start:end]
    local_b = b[start:end]

    local_c = np.array([f(local_a[i], local_b[i]) for i in range(local_a.shape[0])], dtype=np.float32)

    c = None
    if rank == 0:
        c = np.empty_like(a, dtype=np.float32)

    comm.Gather(local_c, c, root=0)


def main():
    array_size = int(input("Enter array size: "))
    group_size = int(input("Enter workgroup size: "))

    a = np.random.rand(array_size).astype(np.float32)
    b = np.random.rand(array_size).astype(np.float32)

    start_time = time.time()
    c_serial = add_serial(a, b)
    elapsed_serial = time.time() - start_time
    print(f"Elapsed time (serial): {elapsed_serial:.6f} seconds")

    start_time = time.time()
    c_numpy = add_numpy(a, b)
    elapsed_numpy = time.time() - start_time
    print(f"Elapsed time (numpy): {elapsed_numpy:.6f} seconds")

    start_time = time.time()
    c_mpi = add_with_mpi(a, b, num_processes=8)  # Указываем количество процессов
    elapsed_mpi = time.time() - start_time
    print(f"Elapsed time (MPI): {elapsed_mpi:.6f} seconds")

    start_time = time.time()
    c_opencl = add_with_opencl(a, b, group_size)
    elapsed_opencl = time.time() - start_time
    print(f"Elapsed time (OpenCL): {elapsed_opencl:.6f} seconds")

    # Проверка корректности
    if np.allclose(c_serial, c_opencl, atol=1e-2):
        print("Results are correct.")
    else:
        print("Results are incorrect.")


if __name__ == "__main__":
    main()
