/****************************************************************************
 * 2025.01.31 作成
 * 作者：近藤 輝
 *
 * このサンプルコードは「GEMM（行列乗算）の最適化をステップバイステップで学ぶ」ための
 * ベースライン実装（ナイーブな3重ループ）です。
 * 
 * ・計算結果が最適化で削除されないよう、最終的な行列 C の要素を用いて
 *   チェックサムを計算・表示する処理を入れています。
 * ・まずはこのナイーブ実装の性能を確認し、ブロッキングやSIMDベクトル化、プリフェッチ
 *   などを段階的に導入していく形でGEMMの性能向上を学習していきます。
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief 行列乗算（C = A * B）を行う関数
 * @note Row-Major 行列を想定
 *       最も、ナイーブな3重ループによる実装
 *       Aのi行目とBのj列目の内積をCのi行j列目に格納
 *
 * @param m 行列 A の行数
 * @param n 行列 B の列数
 * @param k 行列 A の列数、行列 B の行数
 * @param A 行列 A の先頭アドレス size: m x k
 * @param B 行列 B の先頭アドレス size: k x n
 * @param C 行列 C の先頭アドレス size: m x n
 */
void naive_gemm(int m, int n, int k, const float *A, const float *B, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
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
        naive_gemm(n, k, m, A, B, C);
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

