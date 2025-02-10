import pyopencl as cl
import numpy as np
import math
import time

def create_opencl_context():
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    context = cl.Context([devices[0]])
    queue = cl.CommandQueue(context, devices[0])
    return context, queue, devices[0]

def matrix_multiply_opencl(A, B, group_size):
    N = A.shape[0]
    A_cl = np.array(A, dtype=np.float32)
    B_cl = np.array(B, dtype=np.float32)
    result_cl = np.empty_like(A_cl, dtype=np.float32)
    context, queue, device = create_opencl_context()

    program_src = """
    __kernel void matrix_multiply(
        __global const float *A,
        __global const float *B,
        __global float *result,
        const unsigned int N) 
    {
        int row = get_global_id(1);
        int col = get_global_id(0);

        if (row < N && col < N) {
            float sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * N + col];
            }
            result[row * N + col] = sin(sum);  // Применяем синус
        }
    }
    """

    program = cl.Program(context, program_src).build()

    # Создаем буферы для данных
    mf = cl.mem_flags
    A_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_cl)
    B_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_cl)
    result_buffer = cl.Buffer(context, mf.WRITE_ONLY, result_cl.nbytes)

    global_size = (N, N)
    local_size = (group_size, group_size)  # Размер рабочей группы
    program.matrix_multiply(queue, global_size, local_size, A_buffer, B_buffer, result_buffer, np.int32(N))

    cl.enqueue_copy(queue, result_cl, result_buffer).wait()

    return result_cl


def main():
    N = 8192 
    group_size = 16 

    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    start_time = time.time()

    result = matrix_multiply_opencl(A, B, group_size)

    end_time = time.time()

    print(f"Время выполнения: {end_time - start_time} секунд")


if __name__ == "__main__":
    main()
