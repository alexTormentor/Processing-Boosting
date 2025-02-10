#include <iostream>
#include <vector>
#include "mpi.h"
#include <chrono>
#include <thread>


void delay(int n)
{
	unsigned long long fact = 1;
	for (int i = 1; i < n + 1; i++)
		for (int j = 1; j < 500000; j++) fact *= i;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int pRank, pNumber;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
	MPI_Comm_size(MPI_COMM_WORLD, &pNumber);

	double start = MPI_Wtime(), end = 0;

	if (pRank == 0)
	{
		printf("Start\n");
		for (int i = 0; i < 9; i++)
		{
			MPI_Recv(&end, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			printf("Process %d finished Time: %f\n", status.MPI_SOURCE, end - start);
			delay(500);
		}
		end = MPI_Wtime();
		printf("Work is finished %f\n", end - start);
		end = MPI_Wtime();
		printf("All process finished Time: %f\n", end - start);
	}
	else if (pRank == 1)
	{
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 3, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 4, 1, MPI_COMM_WORLD);

		MPI_Recv(&end, 1, MPI_DOUBLE, /*2*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_DOUBLE, /*4*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_DOUBLE, /*5*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		//message = 2;
		end = MPI_Wtime();
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}

	else if (pRank == 2)
	{
		MPI_Recv(&end, 1, MPI_DOUBLE, /*1*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);

		MPI_Recv(&end, 1, MPI_DOUBLE, /*4*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_DOUBLE, /*5*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		//message = 2;
		end = MPI_Wtime();
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}

	else if (pRank == 3)
	{
		MPI_Recv(&end, 1, MPI_DOUBLE, /*1*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 5, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 6, 1, MPI_COMM_WORLD);
	}

	else if (pRank == 4)
	{
		MPI_Recv(&end, 1, MPI_DOUBLE, /*1*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD);
	}

	else if (pRank == 5)
	{
		MPI_Recv(&end, 1, MPI_DOUBLE, /*3*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_DOUBLE, /*7*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 2, 1, MPI_COMM_WORLD);
	}

	else if (pRank == 6)
	{
		MPI_Recv(&end, 1, MPI_DOUBLE, /*3*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_DOUBLE, /*7*/ MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}

	else if (pRank == 7)
	{
		delay(1000);
		end = MPI_Wtime();
		//message = 1;
		MPI_Send(&end, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 5, 1, MPI_COMM_WORLD);
		MPI_Send(&end, 1, MPI_DOUBLE, 6, 1, MPI_COMM_WORLD);
	}

	//p2p();

	MPI_Finalize();
	return 0;
}
