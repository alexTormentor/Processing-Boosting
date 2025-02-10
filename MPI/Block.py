from mpi4py import MPI
import numpy as np


N = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

neighbors = {
    0: [1, 2],
    1: [0, 4],
    2: [0, 3],
    3: [2, 5, 6],
    4: [1, 5],
    5: [3, 4],
    6: [3]
}


def matrix_multiply(A, B):
    return np.dot(A, B)


A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))

# отправка и прием процессов
for neighbor in neighbors[rank]:
    comm.send(A, dest=neighbor)
    comm.send(B, dest=neighbor)

for neighbor in neighbors[rank]:
    received_A = comm.recv(source=neighbor)
    received_B = comm.recv(source=neighbor)
    C += matrix_multiply(received_A, received_B)


print(f"Process {rank} Result:")
print(C)

# сбор промежуточных результатов
result = np.zeros((N, N))
comm.Reduce(C, result, op=MPI.SUM, root=0)

if rank == 0:
    print("Final Result:")
    print(result)



