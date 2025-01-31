/****************************************************************************
 * 2025.01.31 作成
 * 作者：近藤 輝
 *
 * このサンプルコードは「GEMM（行列乗算）の最適化をステップバイステップで学ぶ」ための
 * ベースライン実装からループ順序を変更したサンプルです。
 * 
 * =========================================================================
 * 【最適化の目的】
 *  - ナイーブな3重ループでは、内側のループが“Bの列”に対して飛び飛びのアクセスをするため、
 *    キャッシュ効率が悪くなることが多い。
 *  - そこで、ループ順序を変え、一番内側のループで連続アクセスを発生させるように変更し、
 *    キャッシュヒット率を高めています。
 *
 * =========================================================================
 * 【行列A, B, C のサイズとメモリ配置】
 *   - A: (m x k) 行列 (Row-major)
 *   - B: (k x n) 行列 (Row-major)
 *   - C: (m x n) 行列 (Row-major)
 *
 * 図解: たとえば m=4, k=3, n=5 の場合 (行と列をイメージ的に)
 *
 *   行列 A (4x3): indices = A[i, l]
 *       i\l   0   1   2
 *       ----------------
 *       0    A[0,0] ...
 *       1
 *       2
 *       3
 *
 *   行列 B (3x5): indices = B[l, j]
 *       l\j   0   1   2   3   4
 *       ------------------------
 *       0    B[0,0] ...
 *       1
 *       2
 *
 *   行列 C (4x5): indices = C[i, j]
 *       i\j   0   1   2   3   4
 *       ------------------------
 *       0    C[0,0] ...
 *       1
 *       2
 *       3
 *
 * このプログラムでは、
 *   for i in [0, m):
 *     for l in [0, k):
 *       a_il = A[i*k + l]
 *       for j in [0, n):
 *         C[i*n + j] += a_il * B[l*n + j]
 *
 * という順序でループを回します。
 * B[l*n + j] で j が連続変化するため、B へのアクセスが連続的になりやすく、
 * キャッシュ効率が向上します。
 *
 * =========================================================================
 * 【計算式】
 *   C[i,j] = Σ_{l=0..k-1} ( A[i,l] * B[l,j] )
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief 行列乗算（C = A * B）を行う関数
 * @note Row-Major 行列を想定
 *       最も、ナイーブな3重ループの順番を変更する
 *       一番内側のループがキャッシュに対して連続なアクセスをする。
 *       キャッシュヒット率が高くなる
 *
 * @param m 行列 A の行数
 * @param n 行列 B の列数
 * @param k 行列 A の列数、行列 B の行数
 * @param A 行列 A の先頭アドレス size: m x k
 * @param B 行列 B の先頭アドレス size: k x n
 * @param C 行列 C の先頭アドレス size: m x n
 */
void optimize_loop_order_gemm(int m, int n, int k, const float *A, const float *B, float *C) {
    // 例：ループ順序を i -> l -> j に変える
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            float a_il = A[i*k + l];
            for (int j = 0; j < n; j++) {
                C[i*n + j] += a_il * B[l*n + j];
            }
        }
    }
}

int main(void) {
    int n = 1024, k = 1024, m = 1024;
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));

    for (int i = 0; i < n*k; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < k*m; i++) {
        B[i] = 1.0f;
    }
    for (int i = 0; i < n*m; i++) {
        C[i] = 0.0f;
    }

    clock_t start = clock();

    for (int i = 0; i < 100; ++i) {
        optimize_loop_order_gemm(n, k, m, A, B, C);
    }

    clock_t end = clock();

    double checksum = 0.0;
    for (int i = 0; i < n*m; i++) {
        checksum += C[i];
    }

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC / 100;
    double flops = 2.0 * n * k * m;
    double gflops = flops / (time_spent * 1e9);

    printf("Naive GEMM\n");
    printf("Matrix size: %dx%dx%d\n", n, k, m);
    printf("Time: %.3f seconds\n", time_spent);
    printf("Performance: %.3f GFLOP/s\n", gflops);
    printf("Checksum: %f\n", checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}

