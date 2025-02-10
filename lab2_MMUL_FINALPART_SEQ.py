import numpy as np
import math
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

def apply_sine(matrix):
    return np.sin(matrix)

def main():
    N = 8192  
    A = np.random.rand(N, N)  # Матрица A
    B = np.random.rand(N, N)  # Матрица B

    start_time = time.time()
    result = matrix_multiply(A, B)
    result = apply_sine(result)
    end_time = time.time()

    print(f"Время выполнения: {end_time - start_time} секунд")

if __name__ == "__main__":
    main()
