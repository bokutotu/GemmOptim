/****************************************************************************
 * 2025.01.31 作成
 * 作者：近藤 輝
 *
 * 【次の最適化：ブロッキング（タイル化）】
 * 
 * これは「GEMM（行列乗算）の最適化をステップバイステップで学ぶ」流れにおいて、
 * ループ順序を工夫した後、さらに大きな行列に対してキャッシュ効率を高めるために
 * 「ブロッキング（タイル化）」を導入したサンプルコードです。
 *
 * =========================================================================
 * ■ ブロッキング（タイル化）とは？
 *    大きな行列を小さなブロック（タイル）に分割し、ブロック単位で計算することで
 *    キャッシュミスを減らし、キャッシュの再利用を高める最適化手法です。
 *
 * ＜イメージ図＞
 *    m×k, k×n の行列を、BM×BK, BK×BN といったブロックに分割して計算します。
 * 
 *    例: m=8, k=6, n=8 とし、BM=4, BK=3, BN=4 の場合
 *
 *          行列A (8×6)
 *          +----------+----------+
 *          |  ブロック|  ブロック|
 *          |  (4×3)  |  (4×3)  |
 *          +----------+----------+
 *          |  ブロック|  ブロック|
 *          |  (4×3)  |  (4×3)  |
 *          +----------+----------+
 *
 *          行列B (6×8)
 *          +--------------+--------------+
 *          |   (3×4)     |   (3×4)      |
 *          +--------------+--------------+
 *          |   (3×4)     |   (3×4)      |
 *          +--------------+--------------+
 *
 *    1) 外側ループで (i0, l0, j0) を BM, BK, BN 刻みでまわす
 *    2) 内側でブロック (i, l, j) を計算
 * 
 *    こうすることで、ブロック内の要素をキャッシュに載せたまま集中的に処理でき、
 *    メモリ帯域負荷やキャッシュミスを大幅に削減できます。
 *
 * =========================================================================
 * ■ 今後の発展
 *    - SIMD（ベクトル化）を組み合わせたマイクロカーネル最適化
 *    - パッキング（行列要素をブロックごとに連続配置する） 
 *    - プリフェッチ命令の活用
 *    - マルチスレッド化
 * などを組み合わせることで、さらに高い性能を狙うことができます。
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief ブロッキング（タイル化）したGEMM（C = A * B）
 * @note  Row-Major 行列を想定
 *
 * ループ構造：
 *    for i0 in 0..m..BM
 *      for l0 in 0..k..BK
 *        for j0 in 0..n..BN
 *          // タイル (i0..i0+BM, l0..l0+BK, j0..j0+BN) を計算
 *          for i in i0..iMax
 *            for l in l0..lMax
 *              float a_il = A[i*k + l];
 *              for j in j0..jMax
 *                C[i*n + j] += a_il * B[l*n + j];
 *
 * @param m 行列 A の行数
 * @param n 行列 B の列数
 * @param k 行列 A の列数、行列 B の行数
 * @param A [m x k] 行列
 * @param B [k x n] 行列
 * @param C [m x n] 結果行列
 */
void blocked_gemm(int m, int n, int k,
                  const float *A, const float *B, float *C)
{
    const int BM = 8;
    const int BK = 8;
    const int BN = 8;

    for (int i0 = 0; i0 < m; i0 += BM) {
        int iMax = (i0 + BM < m) ? (i0 + BM) : m;
        for (int l0 = 0; l0 < k; l0 += BK) {
            int lMax = (l0 + BK < k) ? (l0 + BK) : k;
            for (int j0 = 0; j0 < n; j0 += BN) {
                int jMax = (j0 + BN < n) ? (j0 + BN) : n;

                for (int i = i0; i < iMax; i++) {
                    for (int l = l0; l < lMax; l++) {
                        float a_il = A[i*k + l];
                        for (int j = j0; j < jMax; j++) {
                            C[i*n + j] += a_il * B[l*n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(void) {
    int m = 1024, k = 1024, n = 1024;

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

    clock_t start = clock();

    for (int rep = 0; rep < 100; rep++) {
        blocked_gemm(m, n, k, A, B, C);
    }

    clock_t end = clock();

    double checksum = 0.0;
    for (int i = 0; i < m*n; i++) {
        checksum += C[i];
    }

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC / 100;
    double flops = 2.0 * (double)m * (double)k * (double)n;
    double gflops = flops / (time_spent * 1e9);

    printf("Blocked GEMM (Tiling)\n");
    printf("Matrix size: %dx%dx%d (m x k x n)\n", m, k, n);
    printf("Time (avg): %.3f seconds\n", time_spent);
    printf("Performance: %.3f GFLOP/s\n", gflops);
    printf("Checksum: %f\n", checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}

