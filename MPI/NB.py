from mpi4py import MPI
import numpy as np


neighbors = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3, 4],
    3: [1, 2, 4, 5],
    4: [2, 3, 5, 6],
    5: [3, 4, 6],
    6: [4, 5],
}


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 4


np.random.seed(rank)
A = np.random.rand(N, N)
B = np.random.rand(N, N)


C = np.zeros((N, N))


start_time = MPI.Wtime()


for neighbor_rank in neighbors[rank]:
    # буферизация матрицы в биты
    send_buffer = A.tobytes()
    recv_buffer = bytearray(B.tobytes())  

    # синхронная рассылка/приемка
    send_request = comm.Isend(send_buffer, dest=neighbor_rank)
    recv_request = comm.Irecv(recv_buffer, source=neighbor_rank)

    # синхрон
    send_request.Wait()
    recv_request.Wait()

    # битовый массив дешифруем в обычный
    B = np.frombuffer(recv_buffer, dtype=float).reshape(N, N)

    # выполняем умножение с полученными данными
    C += np.dot(A, B)


end_time = MPI.Wtime()
print(f"Process {rank}: Result C =\n{C}")
print(f"Process {rank}: Execution Time = {end_time - start_time} seconds")

if rank == 0:
    all_results = np.zeros((size, N, N))
else:
    all_results = None

comm.Gather(C, all_results, root=0) # сбор результатов

if rank == 0:
    final_result = np.sum(all_results, axis=0)
    print("Final Result (Sum of All Matrices):\n", final_result)
