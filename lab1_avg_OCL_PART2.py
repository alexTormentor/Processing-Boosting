from mpi4py import MPI
import numpy as np
import pyopencl as cl
import math
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def mpi_processing():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 8:
        if rank == 0:
            print("This program requires exactly 8 MPI processes.")
        return

    if rank == 0:
        array_size = 1000
        data = np.random.rand(array_size) * 100
        print(f"Original array (first 10 elements): {data[:10]}")
        chunk_size = len(data) // size
    else:
        data = None
        chunk_size = None

    chunk_size = comm.bcast(chunk_size, root=0)

    recvbuf = np.empty(chunk_size, dtype='d')
    comm.Scatter(data, recvbuf, root=0)

    sorted_chunk = bubble_sort(recvbuf)
    sqrt_chunk = np.sqrt(sorted_chunk)
    sin_chunk = np.sin(sqrt_chunk)
    final_sorted_chunk = bubble_sort(sin_chunk)

    result = None
    if rank == 0:
        result = np.empty_like(data)
    comm.Gather(final_sorted_chunk, result, root=0)

    if rank == 0:
        print(f"Processed array (first 10 elements): {result[:10]}")

def opencl_processing(array_size, device_type, group_size):
    data = np.random.rand(array_size) * 100  

    print(f"Original array (first 10 elements): {data[:10]}")

    platform = cl.get_platforms()[0]
    if device_type == "GPU":
        device = platform.get_devices(device_type=cl.device_type.GPU)[0]
    else:
        device = platform.get_devices(device_type=cl.device_type.CPU)[0]

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    kernel_code = """
    __kernel void process_data(__global float *data, int n) {
        int gid = get_global_id(0);
        if (gid < n) {
            data[gid] = sin(sqrt(data[gid]));
        }
    }
    """

    program = cl.Program(ctx, kernel_code).build()

    mf = cl.mem_flags
    data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

    start_time = time.time()
    program.process_data(queue, (array_size,), (group_size,), data_buf, np.int32(array_size))
    cl.enqueue_copy(queue, data, data_buf).wait()
    end_time = time.time()

    sorted_data = bubble_sort(data)

    print(f"Processed array (first 10 elements): {sorted_data[:10]}")
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["MPI", "OpenCL"], required=True, help="Choose the mode of execution")
    parser.add_argument("--array_size", type=int, default=1024, help="Size of the array (for OpenCL only)")
    parser.add_argument("--device_type", choices=["CPU", "GPU"], default="GPU", help="Device type for OpenCL")
    parser.add_argument("--group_size", type=int, default=8, help="Work group size for OpenCL")
    args = parser.parse_args()

    if args.mode == "MPI":
        mpi_processing()
    elif args.mode == "OpenCL":
        opencl_processing(args.array_size, args.device_type, args.group_size)
