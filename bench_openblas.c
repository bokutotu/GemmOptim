/****************************************************************************
 * 2025.01.31 作成
 * 作者：近藤 輝
 *
 * OpenBLAS (cblas) を用いた GEMM ベンチマーク・サンプルコード
 * 
 * Homebrew でインストールした OpenBLAS を使用する前提で、cblas_sgemm を使って
 * 「C = A x B」(Row-Major) を行い、処理時間・チェックサムを計測する例です。
 *
 * コンパイル例 (brew インストールパスは環境に合わせて変更してください):
 *   clang -o cblas_gemm cblas_gemm.c \
 *         -I/opt/homebrew/include -L/opt/homebrew/lib -lopenblas
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

int main(void) {
    int m = 1024;
    int k = 1024;
    int n = 1024;

    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));

    for (int i = 0; i < m*k; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < k*n; i++) {
        B[i] = 1.0f;
    }
    for (int i = 0; i < m*n; i++) {
        C[i] = 0.0f;
    }

    int iterations = 100;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < iterations; i++) {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            1.0f,
            A,
            k,
            B,
            n,
            0.0f,
            C,
            n
        );
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double checksum = 0.0;
    for (int i = 0; i < m*n; i++) {
        checksum += C[i];
    }

    double time_spent = (end.tv_sec - start.tv_sec);
    time_spent += (end.tv_nsec - start.tv_nsec) / 1e9;
    time_spent /= iterations;

    double flops = 2.0 * (double)m * (double)k * (double)n;
    double gflops = flops / (time_spent * 1.0e9);

    printf("OpenBLAS (cblas_sgemm) GEMM\n");
    printf("Matrix size: %dx%dx%d\n", m, k, n);
    printf("Time: %.6f seconds (average over %d iterations)\n", time_spent, iterations);
    printf("Performance: %.3f GFLOP/s\n", gflops);
    printf("Checksum: %f\n", checksum);

    free(A);
    free(B);
    free(C);

    return 0;
}

