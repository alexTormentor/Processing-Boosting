#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

#include <emmintrin.h>

#define _FLOATING_POINT_
#ifdef _FLOATING_POINT_
//#define _DOUBLE_
#define _FLOAT_
#endif

//#define _INTEGER_
#ifdef _INTEGER_
//#define _int8_
//#define _int16_
//#define _int32_
#define _int64_
#endif

#define N 1000000 // 5000*5000

void main(int argc, char** argv)
{
#ifdef _DOUBLE_
	double* a = (double*)_aligned_malloc(N * N * sizeof(double), 16);
	double* b = (double*)_aligned_malloc(N * N * sizeof(double), 16);
	double* c = (double*)_aligned_malloc(N * N * sizeof(double), 16);
#endif
#ifdef _FLOAT_
	float* a = (float*)_aligned_malloc(N * sizeof(float), 16);
	float* b = (float*)_aligned_malloc(N * sizeof(float), 16);
	float* c = (float*)_aligned_malloc(N * sizeof(float), 16);
#endif
#ifdef _int8_
	unsigned char* a = (unsigned char*)_aligned_malloc(N * sizeof(unsigned char), 16);
	unsigned char* b = (unsigned char*)_aligned_malloc(N * sizeof(unsigned char), 16);
	unsigned char* c = (unsigned char*)_aligned_malloc(N * sizeof(unsigned char), 16);
#endif
#ifdef _int16_
	short* a = (short*)_aligned_malloc(N * sizeof(short), 16);
	short* b = (short*)_aligned_malloc(N * sizeof(short), 16);
	short* c = (short*)_aligned_malloc(N * sizeof(short), 16);
#endif
#ifdef _int32_
	int* a = (int*)_aligned_malloc(N * sizeof(int), 16);
	int* b = (int*)_aligned_malloc(N * sizeof(int), 16);
	int* c = (int*)_aligned_malloc(N * sizeof(int), 16);
#endif
#ifdef _int64_
	__int64* a = (__int64*)_aligned_malloc(N * sizeof(__int64), 16);
	__int64* b = (__int64*)_aligned_malloc(N * sizeof(__int64), 16);
	__int64* c = (__int64*)_aligned_malloc(N * sizeof(__int64), 16);
#endif

	for (int i = 0; i < N; i++)
	{
		a[i] = 1;
		b[i] = 2;
		c[i] = 0;
	}

	double start = omp_get_wtime();
	for (int i = 0; i < N; i++)
		c[i] = a[i] * b[i] * a[i];
	double end = omp_get_wtime();
	printf("\n%f\n", end - start);

	int step = 16;
	start = omp_get_wtime();
#ifdef _DOUBLE_
	int num_iter = N / 2;
#endif
#ifdef _FLOAT_
	int num_iter = N / 4;
#endif
#ifdef _int8_
	int num_iter = N / 16;
#endif
#ifdef _int16_
	int num_iter = N / 8;
#endif
#ifdef _int32_
	int num_iter = N / 4;
#endif
#ifdef _int64_
	int num_iter = N / 2;
#endif
	_asm
	{
		mov ESI, a // ТЕКУЩАЯ ПОЗИЦИЯ В а
		mov EDI, b // ТЕКУЩАЯ ПОЗИЦИЯ В b
		mov EDX, c // ТЕКУЩАЯ ПОЗИЦИЯ В c
		mov ECX, num_iter
		loop_vect :
		movapd xmm0, xmmword ptr[ESI] // ТЕКУЩИЕ ЭЛЕМЕНТЫ a
			movapd xmm1, xmmword ptr[EDI] // ТЕКУЩИЕ ЭЛЕМЕНТЫ b
#ifdef _DOUBLE_
			mulpd xmm1, xmm0 // умножаем
			mulpd xmm1, xmm0 // умножаем
#endif
#ifdef _FLOAT_
			mulps xmm1, xmm0
			mulps xmm1, xmm0
#endif
#ifdef _int64_
			paddq xmm1, xmm0
			paddq xmm1, xmm0
#endif 
#ifdef _int32_
			pmulld xmm1, xmm0
			pmulld xmm1, xmm0
#endif 
#ifdef _int16_
			pmullw xmm1, xmm0
			pmullw xmm1, xmm0
#endif 
#ifdef _int8_
			paddb xmm1, xmm0
			paddb xmm1, xmm0
#endif 
			movapd xmmword ptr[EDX], xmm1 // СОХРАНЯЕМ в c
			add ESI, step
			add EDI, step
			add EDX, step
			loop loop_vect
	}
	end = omp_get_wtime();
	printf("\nasm:\n");
	/*for (int i = 0; i < N; i++)
		printf("%d ", c[i]);*/
	printf("\n%f\n", end - start);

	start = omp_get_wtime();
#ifdef _DOUBLE_
	int step_i = 2;
#endif
#ifdef _FLOAT_
	int step_i = 4;
#endif
#ifdef _int8_
	int step_i = 16;
#endif
#ifdef _int16_
	int step_i = 8;
#endif
#ifdef _int32_
	int step_i = 4;
#endif
#ifdef _int64_
	int step_i = 2;
#endif
#ifdef _DOUBLE_
	__m128d Aij, Bij, Cij;
	for (int i = 0; i < N; i += step_i)
	{
		Aij = _mm_load_pd(&a[i]);
		Bij = _mm_load_pd(&b[i]);
		Cij = _mm_mul_pd(Aij, Bij);
		Cij = _mm_mul_pd(Cij, Aij);
		_mm_store_pd(&c[i], Cij);
	}
#endif
#ifdef _FLOAT_
	__m128 Aij, Bij, Cij;
	for (int i = 0; i < N; i += step_i)
	{
		Aij = _mm_load_ps(&a[i]);
		Bij = _mm_load_ps(&b[i]);
		Cij = _mm_mul_ps(Aij, Bij);
		Cij = _mm_mul_ps(Cij, Aij);
		_mm_store_ps(&c[i], Cij);
	}
#endif
#ifdef _INTEGER_
	__m128i Aij, Bij, Cij;
	for (int i = 0; i < N; i += step_i)
	{
		Aij = _mm_load_si128((__m128i*) & a[i]);
		Bij = _mm_load_si128((__m128i*) & b[i]);
#ifdef _int8_
		Cij = _mm_add_epi8(Aij, Bij);
		Cij = _mm_add_epi8(Cij, Aij);
#endif
#ifdef _int16_
		Cij = _mm_mullo_epi16(Aij, Bij);
		Cij = _mm_mullo_epi16(Cij, Aij);
#endif
#ifdef _int32_
		Cij = _mm_mullo_epi32(Aij, Bij);
		Cij = _mm_mullo_epi32(Cij, Aij);
#endif
#ifdef _int64_
		Cij = _mm_add_epi64(Aij, Bij);
		Cij = _mm_add_epi64(Cij, Aij);
#endif
		_mm_store_si128((__m128i*) & c[i], Cij);
	}
#endif

	end = omp_get_wtime();
	printf("\nintr:\n");
	/*for (int i = 0; i < N; i++)
		printf("%d ", c[i]);*/
	printf("\n%f\n", end - start);

	start = omp_get_wtime();
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < N; i++)
		c[i] = a[i] * b[i] * a[i];
	end = omp_get_wtime();
	printf("\nomp:\n");
	printf("\n%f\n", end - start);
}