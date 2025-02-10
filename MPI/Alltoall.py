from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nnodes = 8
index = [2, 6, 9, 12, 14, 17, 19, 22]
edges = [1, 7, 0, 2, 3, 4, 1, 4, 5, 1, 5, 6, 1, 2, 2, 3, 7, 3, 7, 0, 6, 5]
reorder = 1

comm_graph = comm.Create_graph(index, edges, reorder)

if comm_graph != MPI.COMM_NULL:
    graph_rank = comm_graph.Get_rank()

    matrix_size = (6, 16) 
    A = np.random.rand(matrix_size[0], matrix_size[1])  
    B = np.random.rand(matrix_size[1], matrix_size[1])  

    local_A_columns = matrix_size[1] // size
    local_B_rows = matrix_size[1] // size

    local_A = np.zeros((matrix_size[0], local_A_columns), dtype=np.float64)
    local_B = np.zeros((local_B_rows, matrix_size[1]), dtype=np.float64)

    comm.Scatter(A, local_A, root=0)
    comm.Scatter(B, local_B, root=0)

    local_C = np.dot(local_A, local_B)

    sendbuf = np.zeros((size, local_C.size), dtype=np.float64)
    recvbuf = np.zeros((size, local_C.size), dtype=np.float64)

    sendbuf[graph_rank] = local_C.flatten()

    comm_graph.Alltoall(sendbuf, recvbuf)

    recvbuf = recvbuf.reshape((size, local_A.shape[0], local_B.shape[1]))

    global_C = np.sum(recvbuf, axis=0)


    if rank == 0:
        print("Matrix C:")
        print(global_C)

else:
    print(f"Process {rank} does not belong to the graph communicator.")

MPI.Finalize()
