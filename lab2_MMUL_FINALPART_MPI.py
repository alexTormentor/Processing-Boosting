import numpy as np
from mpi4py import MPI
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

def apply_sine(matrix):
    return np.sin(matrix)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 8192 
    A = np.random.rand(N, N) 
    B = np.random.rand(N, N) 

    rows_per_process = N // size

    start_time = time.time()
    local_A = np.array_split(A, size, axis=0)[rank]
    B = comm.bcast(B, root=0)
    local_result = matrix_multiply(local_A, B)

    # Собираем результаты с всех процессов
    result = None
    if rank == 0:
        result = np.empty((N, N), dtype=np.float64)

    comm.Gather(local_result, result, root=0)

    if rank == 0:
        result = apply_sine(result)
        end_time = time.time()
        print(f"Время выполнения: {end_time - start_time} секунд")

if __name__ == "__main__":
    main()
