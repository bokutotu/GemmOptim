/****************************************************************************
 * 2025.01.31 作成
 * 作者：近藤 輝
 *
 * 【NEON を使った GEMM の簡易ベクトル化デモ】
 *
 * ポイント：
 *   - 行列 A (m x k), B (k x n), C (m x n) を row-major で格納
 *   - 内側ループを "j" 方向に 4 ずつ進め、NEON ベクトル命令で同時に 4 要素を演算
 *   - 端数があるときはスカラー処理でフォロー
 *   - まだブロッキング等は行っておらず、あくまで「ナイーブな3重ループを部分的にNEON化」した例
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>  // NEON ヘッダ

/**
 * @brief NEON を用いて (m x k) * (k x n) = (m x n) の行列乗算を行う
 *
 * @param m 行列 A の行数
 * @param n 行列 B の列数
 * @param k 行列 A の列数 = 行列 B の行数
 * @param A [m*k] サイズの配列 (row-major)
 * @param B [k*n] サイズの配列 (row-major)
 * @param C [m*n] サイズの配列 (row-major)
 */
void gemm_neon(int m, int n, int k, const float *A, const float *B, float *C)
{
    // C = A * B (すべて足し合わせる -> 累積）
    // 事前に C を 0.0 初期化しておく or += にするかは用途による
    // ここでは「C を上書きする」形を想定し、最初に 0.0 で埋める。
    for (int i = 0; i < m*n; i++) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            float a_il = A[i*k + l];

            // 内側ループ: j を 4 ずつ進める
            int j = 0;
            for (; j <= n - 4; j += 4) {
                // B[l, j..j+3] をロード (4要素)
                float32x4_t b_vec = vld1q_f32(&B[l*n + j]);
                // C[i, j..j+3] をロード
                float32x4_t c_vec = vld1q_f32(&C[i*n + j]);
                // c_vec += a_il * b_vec
                float32x4_t tmp = vmlaq_n_f32(c_vec, b_vec, a_il);
                // 結果をストア
                vst1q_f32(&C[i*n + j], tmp);
            }

            // 端数処理 (n が 4 の倍数でない場合の残り列)
            for (; j < n; j++) {
                C[i*n + j] += a_il * B[l*n + j];
            }
        }
    }
}


int main(void)
{
    // 例として 1024 x 1024 の行列
    int m = 1024, k = 1024, n = 1024;
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));

    // 初期化 (単純に 1.0f で埋める）
    for (int i = 0; i < m*k; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < k*n; i++) {
        B[i] = 1.0f;
    }

    // 計測開始
    clock_t start = clock();

    // ここでは 10 回程度回して時間を測定
    for (int rep = 0; rep < 100; rep++) {
        gemm_neon(m, n, k, A, B, C);
    }

    // 計測終了
    clock_t end = clock();

    // チェックサムを計算して結果が最適化除去されないようにする
    double checksum = 0.0;
    for (int i = 0; i < m*n; i++) {
        checksum += C[i];
    }

    // 実行時間と FLOPS を計算
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC / 100; // 10回平均
    double flops = 2.0 * (double)m * (double)k * (double)n;  // (乗算+加算) * m*k*n
    double gflops = flops / (time_spent * 1e9);

    // 結果表示
    printf("NEON GEMM (Naive partial vectorization)\n");
    printf("Matrix size: %dx%dx%d\n", m, k, n);
    printf("Time (avg): %.3f s\n", time_spent);
    printf("Perf: %.3f GFLOP/s\n", gflops);
    printf("Checksum: %f\n", checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}

