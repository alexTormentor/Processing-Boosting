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

	//p2p();
	//bcast();

	int pRank, pNumber;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
	MPI_Comm_size(MPI_COMM_WORLD, &pNumber);

	double start = MPI_Wtime(), end = 0;

	double times[8] = { 0 };
	if (pRank == 0)
		printf("Start\n");

	// шаг 1 - 1 и 7 потоки
	else if (pRank == 1)
	{
		delay(1000); end = MPI_Wtime();
	}
	else if (pRank == 7)
	{
		delay(5000); end = MPI_Wtime();
	}

	MPI_Allgather(&end, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	end = 0;

	if (pRank == 0)
	{
		for (int i = 0; i < 8; i++)
			if (times[i] != 0)
			{
				printf("Process %d finished Time: %f\n", i, times[i] - start);
				times[i] = 0;
			}
	}

	// шаг 2 - 2, 3, 4 потоки
	else if (pRank == 2)
	{
		delay(1800);
		end = MPI_Wtime();
	}
	else if (pRank == 3)
	{
		delay(1500);
		end = MPI_Wtime();
	}
	else if (pRank == 4)
	{
		delay(2000);
		end = MPI_Wtime();
	}

	MPI_Allgather(&end, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	end = 0;

	if (pRank == 0)
	{
		for (int i = 0; i < 8; i++)
			if (times[i] != 0)
			{
				printf("Process %d finished Time: %f\n", i, times[i] - start);
				times[i] = 0;
			}
	}

	// шаг 3 - 5, 6 потоки
	else if (pRank == 5)
	{
		delay(500);
		end = MPI_Wtime();
	}
	else if (pRank == 6)
	{
		delay(1000);
		end = MPI_Wtime();
	}

	MPI_Allgather(&end, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, MPI_COMM_WORLD);
	end = 0;

	if (pRank == 0)
	{
		for (int i = 0; i < 8; i++)
			if (times[i] != 0)
			{
				printf("Process %d finished Time: %f\n", i, times[i] - start);
				times[i] = 0;
			}
	}

	// шаг 4 - 1, 2 потоки
	else if (pRank == 1)
	{
		delay(500);
		end = MPI_Wtime();
	}
	else if (pRank == 2)
	{
		delay(1000);
		end = MPI_Wtime();
	}

	MPI_Allgather(&end, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, MPI_COMM_WORLD);

	if (pRank == 0)
	{
		for (int i = 0; i < 8; i++)
			if (times[i] != 0)
			{
				printf("Process %d finished Time: %f\n", i, times[i] - start);
				times[i] = 0;
			}
		end = MPI_Wtime();
		printf("All process finished Time: %f\n", end - start);
	}

	MPI_Finalize();
	return 0;
}
