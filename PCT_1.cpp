#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Определение типа для локальной матрицы
typedef struct {
    int rows;
    int cols;
    double* data;
} LOCAL_MATRIX_TYPE;

// Функция для выделения памяти под локальную матрицу
LOCAL_MATRIX_TYPE* Local_matrix_allocate(int rows, int cols) {
    LOCAL_MATRIX_TYPE* local_matrix = (LOCAL_MATRIX_TYPE*)malloc(sizeof(LOCAL_MATRIX_TYPE));
    local_matrix->rows = rows;
    local_matrix->cols = cols;
    local_matrix->data = (double*)malloc(rows * cols * sizeof(double));
    return local_matrix;
}

// Функция для инициализации локальной матрицы нулями
void Set_to_zero(LOCAL_MATRIX_TYPE* local_matrix) {
    int i, j;
    for (i = 0; i < local_matrix->rows; i++) {
        for (j = 0; j < local_matrix->cols; j++) {
            local_matrix->data[i * local_matrix->cols + j] = 0.0;
        }
    }
}

// Функция для умножения двух локальных матриц
void Local_matrix_multiply(LOCAL_MATRIX_TYPE* local_A, LOCAL_MATRIX_TYPE* local_B, LOCAL_MATRIX_TYPE* local_C) {
    int i, j, k;
    for (i = 0; i < local_A->rows; i++) {
        for (j = 0; j < local_B->cols; j++) {
            for (k = 0; k < local_A->cols; k++) {
                local_C->data[i * local_C->cols + j] += local_A->data[i * local_A->cols + k] * local_B->data[k * local_B->cols + j];
            }
        }
    }
}

// Определение структуры для информации о сетке
typedef struct {
    int p; // Общее число процессов
    MPI_Comm comm; // Коммуникатор для сетки
    MPI_Comm row_comm; // Коммуникатор строки
    MPI_Comm col_comm; // Коммуникатор столбца
    int q; // Порядок сетки
    int my_row; // Номер строки
    int my_col; // Номер столбца
    int my_rank; // Ранг процесса в коммуникаторе сетки
} GRID_INFO_TYPE;

// Функция для настройки сетки
void Setup_grid(GRID_INFO_TYPE* grid) {
    int old_rank;
    int dimensions[2];
    int periods[2];
    int coordinates[2];
    int varying_coords[2];

    // Настройка глобальной информации о сетке
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);
    grid->q = (int)sqrt((double)grid->p);
    dimensions[0] = dimensions[1] = grid->q;
    periods[0] = periods[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    // Настройка коммуникаторов для строк и столбцов
    varying_coords[0] = 0;
    varying_coords[1] = 1;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->row_comm));
    varying_coords[0] = 1;
    varying_coords[1] = 0;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->col_comm));
}

// Функция для умножения матриц с использованием алгоритма Фокса
void Fox(int n, GRID_INFO_TYPE* grid, LOCAL_MATRIX_TYPE* local_A, LOCAL_MATRIX_TYPE* local_B, LOCAL_MATRIX_TYPE* local_C) {
    LOCAL_MATRIX_TYPE* temp_A;
    int step;
    int bcast_root;
    int n_bar; // Порядок подматрицы = n/q
    int source;
    int dest;
    int tag = 43;
    MPI_Status status;

    n_bar = n / grid->q;
    Set_to_zero(local_C);

    // Вычисление адресов для циклического сдвига B
    source = (grid->my_row + 1) % grid->q;
    dest = (grid->my_row + grid->q - 1) % grid->q;

    // Выделение памяти для рассылки блоков A
    temp_A = Local_matrix_allocate(n_bar, n_bar);

    for (step = 0; step < grid->q; step++) {
        bcast_root = (grid->my_row + step) % grid->q;
        if (bcast_root == grid->my_col) {
            MPI_Bcast(local_A->data, local_A->rows * local_A->cols, MPI_DOUBLE, bcast_root, grid->row_comm);
            Local_matrix_multiply(local_A, local_B, local_C);
        }
        else {
            MPI_Bcast(temp_A->data, temp_A->rows * temp_A->cols, MPI_DOUBLE, bcast_root, grid->row_comm);
            Local_matrix_multiply(temp_A, local_B, local_C);
        }

        MPI_Send(local_B->data, local_B->rows * local_B->cols, MPI_DOUBLE, dest, tag, grid->col_comm);
        MPI_Recv(local_B->data, local_B->rows * local_B->cols, MPI_DOUBLE, source, tag, grid->col_comm, &status);
    }

    // Освобождение выделенной памяти
    free(temp_A->data);
    free(temp_A);
}


int main(int argc, char* argv[]) {
    int n = 4; // Размер матрицы (пример: 4x4)
    MPI_Init(&argc, &argv);

    int p; // Общее число процессов
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p < 4) {
        fprintf(stderr, "Ошибка: Необходимо как минимум 4 процесса\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    GRID_INFO_TYPE grid;
    Setup_grid(&grid);

    // Создание и инициализация локальных матриц
    LOCAL_MATRIX_TYPE* local_A = Local_matrix_allocate(n / grid.q, n / grid.q);
    LOCAL_MATRIX_TYPE* local_B = Local_matrix_allocate(n / grid.q, n / grid.q);
    LOCAL_MATRIX_TYPE* local_C = Local_matrix_allocate(n / grid.q, n / grid.q);

    // Заполнение local_A данными (пример)
    for (int i = 0; i < local_A->rows; i++) {
        for (int j = 0; j < local_A->cols; j++) {
            local_A->data[i * local_A->cols + j] = 1.0; // Замените на свои данные
        }
    }

    // Заполнение local_B данными (пример)
    for (int i = 0; i < local_B->rows; i++) {
        for (int j = 0; j < local_B->cols; j++) {
            local_B->data[i * local_B->cols + j] = 1.0; // Замените на свои данные
        }
    }


    Fox(n, &grid, local_A, local_B, local_C);


    // Вычисление суммы на 0-м блоке строк
    double local_sum = 0.0;
    for (int i = 0; i < local_C->rows; i++) {
        for (int j = 0; j < local_C->cols; j++) {
            local_sum += local_C->data[i * local_C->cols + j];
        }
    }

    double global_sum = 0.0;

    // Используйте MPI_Reduce для суммирования на 0-м блоке строк
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, grid.row_comm);

    // global_sum теперь содержит сумму значений на 0-м блоке строк всех процессов

    // Здесь вы можете вывести global_sum только на корневом процессе (процессе с рангом 0)
    if (grid.my_rank == 0) {
        printf("Sum on Process 0: %.2f\n", global_sum);
    }


    // Вывод содержимого local_C на каждом процессе после операции суммирования
    for (int proc_row = 0; proc_row < grid.q; proc_row++) {
        for (int proc_col = 0; proc_col < grid.q; proc_col++) {
            if (grid.my_row == proc_row && grid.my_col == proc_col) {
                // Текущий процесс, выводим содержимое local_C
                printf("Local_C on Process [%d][%d]:\n", grid.my_row, grid.my_col);
                for (int i = 0; i < local_C->rows; i++) {
                    for (int j = 0; j < local_C->cols; j++) {
                        printf("%.2f ", local_C->data[i * local_C->cols + j]);
                    }
                    printf("\n");
                }
            }

            MPI_Barrier(grid.comm); // Синхронизация процессов
        }
    }



    // Освобождение памяти
    free(local_A->data);
    free(local_A);
    free(local_B->data);
    free(local_B);
    free(local_C->data);
    free(local_C);

    MPI_Finalize();
    return 0;
}
