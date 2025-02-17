#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <emmintrin.h>

void sumUsual(double* a, double* b, double* c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            c[i * N + j] = a[i * N + j] + b[j * N + i];
}

void sumBlocking(double* a, double* b, double* c, int N, int block_size)
{
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < std::min(i + block_size, N); ii++)
                for (int jj = j; jj < std::min(j + block_size, N); jj++)
                    c[ii * N + jj] = a[ii * N + jj] + b[jj * N + ii];
}

void sumUsual2(double* a, double* b, double* c, int N)
{
    a[0] = 0;
    a[1] = 1;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            c[i * N + j] = (a[i * N + j] + b[j * N + i]) * b[j * N + i] / ((a[i * N + j]) * (a[i
                * N + j]));
            if ((i * N + j) < N * N - 2)
                a[i * N + j + 2] = (a[i * N + j] + b[j * N + i]) / b[j * N + i];
        }
}

void sumBlocking2(double* a, double* b, double* c, int N, int block_size)
{
    a[0] = 0;
    a[1] = 1;
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < i + block_size; ii++)
                for (int jj = j; jj < j + block_size; jj++)
                {
                    c[ii * N + jj] = (a[ii * N + jj] + b[jj * N + ii]) * b[jj * N + ii] / ((a[ii *
                        N + jj]) * (a[ii * N + jj]));
                    if ((ii * N + jj) < N * N - 2)
                        a[ii * N + jj + 2] = (a[ii * N + jj] + b[jj * N + ii]) / b[jj * N + ii];
                }
}

void sumBlockingIntr(double* a, double* b, double* c, int N, int block_size)
{
    a[0] = 0;
    a[1] = 1;
    __m128d Aij, Aijtemp, Bij, Cij;
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < i + block_size; ii++)
                for (int jj = j; jj < j + block_size; jj += 2) {
                    Aij = _mm_load_pd((double const*)&a[ii * N + jj]);
                    Bij = _mm_load_sd((double const*)&b[jj * N + ii]);
                    Bij = _mm_loadh_pd(Bij, (double const*)&b[(jj + 1) * N + ii]);
                    Cij = _mm_add_pd(Aij, Bij);
                    Cij = _mm_mul_pd(Cij, Bij);
                    Aijtemp = _mm_mul_pd(Aij, Aij);
                    Cij = _mm_div_pd(Cij, Aijtemp);
                    _mm_store_pd(&c[ii * N + jj], Cij);
                    if ((ii * N + jj) < N * N - 2) {
                        Aijtemp = _mm_add_pd(Aij, Bij);
                        Aijtemp = _mm_div_pd(Aijtemp, Bij);
                        _mm_store_pd(&a[ii * N + jj + 2], Aijtemp);
                    }
                }
}

void sumUsualIntr(double* a, double* b, double* c, int N)
{
    a[0] = 0;
    a[1] = 1;
    __m128d Aij, Aijtemp, Bij, Cij;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 2) { // Обрабатываем два элемента за раз
            // Загружаем два элемента из массивов a и b
            Aij = _mm_load_pd(&a[i * N + j]);
            Bij = _mm_set_pd(b[(j + 1) * N + i], b[j * N + i]); // Создаем 2-элементный вектор

            // Вычисляем выражение (a + b) * b / (a * a)
            Cij = _mm_add_pd(Aij, Bij); // (a + b)
            Cij = _mm_mul_pd(Cij, Bij); // (a + b) * b
            Aijtemp = _mm_mul_pd(Aij, Aij); // a * a
            Cij = _mm_div_pd(Cij, Aijtemp); // (a + b) * b / (a * a)

            // Сохраняем результат в c
            _mm_store_pd(&c[i * N + j], Cij);

            // Обновляем значения в массиве a
            if ((i * N + j) < N * N - 2) {
                Aijtemp = _mm_add_pd(Aij, Bij); // (a + b)
                Aijtemp = _mm_div_pd(Aijtemp, Bij); // (a + b) / b
                _mm_store_pd(&a[i * N + j + 2], Aijtemp); // обновление a
            }
        }
    }
}

void measureAndRun(void (*func)(double*, double*, double*, int), double* a, double* b, double* c, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    func(a, b, c, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds.\n";
}

void measureAndRunWithBlock(void (*func)(double*, double*, double*, int, int), double* a, double* b, double* c, int N, int block_size) {
    auto start = std::chrono::high_resolution_clock::now();
    func(a, b, c, N, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds.\n";
}


inline double f(double x) {
    return x * x; // Пример функции f(x), можно изменить при необходимости
}

void sumUsual3(double* a, double* b, double* c, int N) {
    a[0] = 0;
    a[1] = 1;
    for (int i = 0; i < N; i += 2) // Шаг 2 для i
        for (int j = 0; j < N; j += 2) // Шаг 2 для j
        {
            double temp = f(a[i * N + j] * b[j * N + i]);
            c[i * N + j] = temp * b[j * N + i] / (a[i * N + j] * a[i * N + j]);
            if ((i * N + j) < N * N - 2)
                a[i * N + j + 2] = (a[i * N + j] + b[j * N + i]) / b[j * N + i];
        }
}

void sumBlocking3(double* a, double* b, double* c, int N, int block_size) {
    a[0] = 0;
    a[1] = 1;
    for (int i = 0; i < N; i += block_size)
        for (int j = 0; j < N; j += block_size)
            for (int ii = i; ii < i + block_size; ii += 2) // Шаг 2 для ii
                for (int jj = j; jj < j + block_size; jj += 2) // Шаг 2 для jj
                {
                    double temp = f(a[ii * N + jj] * b[jj * N + ii]);
                    c[ii * N + jj] = temp * b[jj * N + ii] / (a[ii * N + jj] * a[ii * N + jj]);
                    if ((ii * N + jj) < N * N - 2)
                        a[ii * N + jj + 2] = (a[ii * N + jj] + b[jj * N + ii]) / b[jj * N + ii];
                }
}

int main() {
    int N = 8192;
    int block_size = 128;
    std::vector<double> a(N * N);
    std::vector<double> b(N * N);

    // Генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister для генерации случайных чисел
    std::uniform_real_distribution<> dis(0.0, 100.0); // Диапазон от 0.0 до 1.0

    for (int i = 0; i < N * N; ++i) {
        a[i] = dis(gen); // Случайное число от 0 до 1
        b[i] = dis(gen);
    }
    std::vector<double> c(N * N, 0.0);

    // Benchmarking sumUsual
    /*auto start = std::chrono::high_resolution_clock::now();
    sumUsual(a.data(), b.data(), c.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "sumUsual: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";*/

    // Benchmarking sumBlocking
    auto start = std::chrono::high_resolution_clock::now();
    sumBlocking(a.data(), b.data(), c.data(), N, block_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "sumBlocking: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    

    /*const int N = 1024; // Размер матриц
    const int block_size = 16; // Размер блока
    double* a = new double[N * N];
    double* b = new double[N * N];
    double* c = new double[N * N];

    // Инициализация массивов a и b случайными значениями для корректного тестирования
    for (int i = 0; i < N * N; ++i) {
        a[i] = static_cast<double>(rand()) / RAND_MAX;
        b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    std::cout << "sumUsual2:\n";
    measureAndRun(sumUsual2, a, b, c, N);

    std::cout << "sumBlocking2:\n";
    measureAndRunWithBlock(sumBlocking2, a, b, c, N, block_size);

    std::cout << "sumUsual3:\n";
    measureAndRun(sumUsual3, a, b, c, N);

    std::cout << "sumBlocking3:\n";
    measureAndRunWithBlock(sumBlocking3, a, b, c, N, block_size);

    std::cout << "sumBlockingIntr:\n";
    measureAndRunWithBlock(sumBlockingIntr, a, b, c, N, block_size);

    std::cout << "sumUsualIntr:\n";
    measureAndRun(sumUsualIntr, a, b, c, N);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;*/

    

    return 0;
}
