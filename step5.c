#include <arm_neon.h> // ARM NEON SIMD 命令用のヘッダーを追加
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h> // OpenBLAS用のヘッダーを追加

#define N 1024         // 行列サイズ
#define BLOCK_SIZE 128 // 外側のブロックサイズ（調整可能）
#define TILE_SIZE 32   // タイルサイズ（適宜調整可）

// ナイーブなGEMM実装: C = A * B
void gemm(double *A, double *B, double *C) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// 1段階のタイル化を行ったGEMM実装: C = A * B
// ※ C は事前にゼロ初期化されていることを前提としています
void gemm_tiled(double *A, double *B, double *C) {
  int i, j, k, i0, j0, k0;
  for (i0 = 0; i0 < N; i0 += TILE_SIZE) {
    for (j0 = 0; j0 < N; j0 += TILE_SIZE) {
      for (k0 = 0; k0 < N; k0 += TILE_SIZE) {
        int i_max = (i0 + TILE_SIZE < N) ? (i0 + TILE_SIZE) : N;
        int j_max = (j0 + TILE_SIZE < N) ? (j0 + TILE_SIZE) : N;
        int k_max = (k0 + TILE_SIZE < N) ? (k0 + TILE_SIZE) : N;
        for (i = i0; i < i_max; i++) {
          for (j = j0; j < j_max; j++) {
            for (k = k0; k < k_max; k++) {
              C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

// SIMDを活用したタイル化GEMM実装: C = A * B
// ARM NEONを使用して内部ループをベクトル化
void gemm_tiled_simd(double *A, double *B, double *C) {
  int i, j, k, i0, j0, k0;
  for (i0 = 0; i0 < N; i0 += TILE_SIZE) {
    for (j0 = 0; j0 < N; j0 += TILE_SIZE) {
      for (k0 = 0; k0 < N; k0 += TILE_SIZE) {
        int i_max = (i0 + TILE_SIZE < N) ? (i0 + TILE_SIZE) : N;
        int j_max = (j0 + TILE_SIZE < N) ? (j0 + TILE_SIZE) : N;
        int k_max = (k0 + TILE_SIZE < N) ? (k0 + TILE_SIZE) : N;

        for (i = i0; i < i_max; i++) {
          for (j = j0; j < j_max; j += 2) { // 2要素ずつ処理（double2ベクトル）
            // j+1が範囲を超える可能性があるため、チェック
            float64x2_t c_vec = vdupq_n_f64(0.0); // 累積用ベクトルを0で初期化

            for (k = k0; k < k_max; k++) {
              // Aから1つの要素をロードして複製
              float64x2_t a_vec = vdupq_n_f64(A[i * N + k]);

              // Bから連続する2要素をロード
              float64x2_t b_vec = {B[k * N + j], B[k * N + j + 1]};

              // ベクトル積和演算
              c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
            }

            // 結果をCにストア
            C[i * N + j] += vgetq_lane_f64(c_vec, 0);
            C[i * N + j + 1] += vgetq_lane_f64(c_vec, 1);
          }
        }
      }
    }
  }
}

void gemm_multi_tiled_simd(double *restrict A, double *restrict B,
                           double *restrict C) {
  int i, j, k;
  int i0, j0, k0;
  int i1, j1, k1;

  // 外側のブロックループ
  for (i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
    for (j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
      for (k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
        int i0_max = (i0 + BLOCK_SIZE < N) ? (i0 + BLOCK_SIZE) : N;
        int j0_max = (j0 + BLOCK_SIZE < N) ? (j0 + BLOCK_SIZE) : N;
        int k0_max = (k0 + BLOCK_SIZE < N) ? (k0 + BLOCK_SIZE) : N;

        // 内側のタイルループ
        for (i1 = i0; i1 < i0_max; i1 += TILE_SIZE) {
          for (j1 = j0; j1 < j0_max; j1 += TILE_SIZE) {
            for (k1 = k0; k1 < k0_max; k1 += TILE_SIZE) {
              int i1_max =
                  (i1 + TILE_SIZE < i0_max) ? (i1 + TILE_SIZE) : i0_max;
              int j1_max =
                  (j1 + TILE_SIZE < j0_max) ? (j1 + TILE_SIZE) : j0_max;
              int k1_max =
                  (k1 + TILE_SIZE < k0_max) ? (k1 + TILE_SIZE) : k0_max;

              // 内部計算（SIMD処理）
              for (i = i1; i < i1_max; i++) {
                for (j = j1; j < j1_max;
                     j += 2) { // 2要素ずつ処理（double2ベクトル）
                  // SIMD用の累積レジスタを初期化
                  float64x2_t c_vec = vdupq_n_f64(0.0);

                  for (k = k1; k < k1_max; k++) {
                    // Aの1要素を複製してベクトルに
                    float64x2_t a_vec = vdupq_n_f64(A[i * N + k]);

                    // Bから連続する2要素をロード
                    float64x2_t b_vec = {B[k * N + j], B[k * N + j + 1]};

                    // ベクトル乗算・加算（FMA的な処理）
                    c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
                  }

                  // 結果をCにストア（元の値に加算）
                  C[i * N + j] += vgetq_lane_f64(c_vec, 0);
                  C[i * N + j + 1] += vgetq_lane_f64(c_vec, 1);
                }
              }
            }
          }
        }
      }
    }
  }
}

void gemm_multi_tiled_simd_unrolled(double *restrict A, double *restrict B,
                                    double *restrict C) {
  int i, j, k;
  int i0, j0, k0;
  int i1, j1, k1;

  // 外側のブロックループ
  for (i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
    for (j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
      for (k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
        int i0_max = (i0 + BLOCK_SIZE < N) ? (i0 + BLOCK_SIZE) : N;
        int j0_max = (j0 + BLOCK_SIZE < N) ? (j0 + BLOCK_SIZE) : N;
        int k0_max = (k0 + BLOCK_SIZE < N) ? (k0 + BLOCK_SIZE) : N;

        // 内側のタイルループ
        for (i1 = i0; i1 < i0_max; i1 += TILE_SIZE) {
          for (j1 = j0; j1 < j0_max; j1 += TILE_SIZE) {
            for (k1 = k0; k1 < k0_max; k1 += TILE_SIZE) {
              int i1_max =
                  (i1 + TILE_SIZE < i0_max) ? (i1 + TILE_SIZE) : i0_max;
              int j1_max =
                  (j1 + TILE_SIZE < j0_max) ? (j1 + TILE_SIZE) : j0_max;
              int k1_max =
                  (k1 + TILE_SIZE < k0_max) ? (k1 + TILE_SIZE) : k0_max;

              // 内部計算（SIMD処理＋ループアンローリング）
              for (i = i1; i < i1_max; i++) {
                for (j = j1; j < j1_max;
                     j += 2) { // 2要素ずつ処理（double2ベクトル）
                  float64x2_t c_vec = vdupq_n_f64(0.0);

                  // kループを4ループ分アンローリング
                  int k_unroll_end = k1_max - ((k1_max - k1) % 4);
                  for (k = k1; k < k_unroll_end; k += 4) {
                    // Aの値を複製してベクトルに
                    float64x2_t a_vec0 = vdupq_n_f64(A[i * N + k]);
                    float64x2_t a_vec1 = vdupq_n_f64(A[i * N + k + 1]);
                    float64x2_t a_vec2 = vdupq_n_f64(A[i * N + k + 2]);
                    float64x2_t a_vec3 = vdupq_n_f64(A[i * N + k + 3]);

                    // Bから連続する2要素をロード
                    float64x2_t b_vec0 = {B[k * N + j], B[k * N + j + 1]};
                    float64x2_t b_vec1 = {B[(k + 1) * N + j],
                                          B[(k + 1) * N + j + 1]};
                    float64x2_t b_vec2 = {B[(k + 2) * N + j],
                                          B[(k + 2) * N + j + 1]};
                    float64x2_t b_vec3 = {B[(k + 3) * N + j],
                                          B[(k + 3) * N + j + 1]};

                    // 複数のFMAを適用
                    c_vec = vmlaq_f64(c_vec, a_vec0, b_vec0);
                    c_vec = vmlaq_f64(c_vec, a_vec1, b_vec1);
                    c_vec = vmlaq_f64(c_vec, a_vec2, b_vec2);
                    c_vec = vmlaq_f64(c_vec, a_vec3, b_vec3);
                  }
                  // アンローリングで処理しきれなかった余りループ
                  for (; k < k1_max; k++) {
                    float64x2_t a_vec = vdupq_n_f64(A[i * N + k]);
                    float64x2_t b_vec = {B[k * N + j], B[k * N + j + 1]};
                    c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
                  }
                  // 計算結果をCに加算
                  C[i * N + j] += vgetq_lane_f64(c_vec, 0);
                  C[i * N + j + 1] += vgetq_lane_f64(c_vec, 1);
                }
              }
            }
          }
        }
      }
    }
  }
}

void gemm_multi_tiled_simd_unrolled_prefetch(double *restrict A,
                                             double *restrict B,
                                             double *restrict C) {
  int i, j, k;
  int i0, j0, k0;
  int i1, j1, k1;

  // 外側のブロックループ
  for (i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
    for (j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
      for (k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
        int i0_max = (i0 + BLOCK_SIZE < N) ? (i0 + BLOCK_SIZE) : N;
        int j0_max = (j0 + BLOCK_SIZE < N) ? (j0 + BLOCK_SIZE) : N;
        int k0_max = (k0 + BLOCK_SIZE < N) ? (k0 + BLOCK_SIZE) : N;

        // 内側のタイルループ
        for (i1 = i0; i1 < i0_max; i1 += TILE_SIZE) {
          for (j1 = j0; j1 < j0_max; j1 += TILE_SIZE) {
            for (k1 = k0; k1 < k0_max; k1 += TILE_SIZE) {
              int i1_max =
                  (i1 + TILE_SIZE < i0_max) ? (i1 + TILE_SIZE) : i0_max;
              int j1_max =
                  (j1 + TILE_SIZE < j0_max) ? (j1 + TILE_SIZE) : j0_max;
              int k1_max =
                  (k1 + TILE_SIZE < k0_max) ? (k1 + TILE_SIZE) : k0_max;

              // 内部計算（SIMD処理＋ループアンローリング＋プリフェッチ）
              for (i = i1; i < i1_max; i++) {
                for (j = j1; j < j1_max;
                     j += 2) { // 2要素ずつ処理（double2ベクトル）
                  // C行列の先読み（書き込み先）
                  __builtin_prefetch(C + i * N + j + 16, 1, 1);

                  float64x2_t c_vec = vdupq_n_f64(0.0);

                  // kループのアンローリング用準備
                  int k_unroll_end = k1_max - ((k1_max - k1) % 4);
                  for (k = k1; k < k_unroll_end; k += 4) {
                    // A, Bの先読み（次のループ反復分）
                    if (k + 4 < k1_max) {
                      __builtin_prefetch(A + i * N + k + 4, 0, 1);
                      __builtin_prefetch(B + (k + 4) * N + j, 0, 1);
                    }

                    // アンローリングされた4回分の演算
                    float64x2_t a_vec0 = vdupq_n_f64(A[i * N + k]);
                    float64x2_t a_vec1 = vdupq_n_f64(A[i * N + k + 1]);
                    float64x2_t a_vec2 = vdupq_n_f64(A[i * N + k + 2]);
                    float64x2_t a_vec3 = vdupq_n_f64(A[i * N + k + 3]);

                    float64x2_t b_vec0 = {B[k * N + j], B[k * N + j + 1]};
                    float64x2_t b_vec1 = {B[(k + 1) * N + j],
                                          B[(k + 1) * N + j + 1]};
                    float64x2_t b_vec2 = {B[(k + 2) * N + j],
                                          B[(k + 2) * N + j + 1]};
                    float64x2_t b_vec3 = {B[(k + 3) * N + j],
                                          B[(k + 3) * N + j + 1]};

                    c_vec = vmlaq_f64(c_vec, a_vec0, b_vec0);
                    c_vec = vmlaq_f64(c_vec, a_vec1, b_vec1);
                    c_vec = vmlaq_f64(c_vec, a_vec2, b_vec2);
                    c_vec = vmlaq_f64(c_vec, a_vec3, b_vec3);
                  }
                  // 残りの余りループ
                  for (; k < k1_max; k++) {
                    float64x2_t a_vec = vdupq_n_f64(A[i * N + k]);
                    float64x2_t b_vec = {B[k * N + j], B[k * N + j + 1]};
                    c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
                  }
                  // 結果をCに加算
                  C[i * N + j] += vgetq_lane_f64(c_vec, 0);
                  C[i * N + j + 1] += vgetq_lane_f64(c_vec, 1);
                }
              }
            }
          }
        }
      }
    }
  }
}

// 4x4マイクロカーネル：C[i:i+4, j:j+4] に対して、
// k方向の範囲 [k_start, k_end) で部分積を計算し、Cに加算する。
void gemm_microkernel_4x4(double *restrict A, double *restrict B,
                          double *restrict C, int i, int j, int k_start,
                          int k_end) {
  // 4行×4列分の累積レジスタ（各レジスタは2要素を持つので、4列=2ベクトル）
  float64x2_t c0_0 = vdupq_n_f64(0.0), c0_1 = vdupq_n_f64(0.0);
  float64x2_t c1_0 = vdupq_n_f64(0.0), c1_1 = vdupq_n_f64(0.0);
  float64x2_t c2_0 = vdupq_n_f64(0.0), c2_1 = vdupq_n_f64(0.0);
  float64x2_t c3_0 = vdupq_n_f64(0.0), c3_1 = vdupq_n_f64(0.0);

  for (int k = k_start; k < k_end; k++) {
    // Aの各行から1要素をロードしてベクトルに複製
    float64x2_t a0 = vdupq_n_f64(A[(i + 0) * N + k]);
    float64x2_t a1 = vdupq_n_f64(A[(i + 1) * N + k]);
    float64x2_t a2 = vdupq_n_f64(A[(i + 2) * N + k]);
    float64x2_t a3 = vdupq_n_f64(A[(i + 3) * N + k]);

    // Bから、連続する4要素を2ベクトルでロード（条件分岐なしで4要素）
    float64x2_t b0 = vld1q_f64(B + k * N + j);     // B[k][j]～B[k][j+1]
    float64x2_t b1 = vld1q_f64(B + k * N + j + 2); // B[k][j+2]～B[k][j+3]

    // FMA的な積和演算
    c0_0 = vmlaq_f64(c0_0, a0, b0);
    c0_1 = vmlaq_f64(c0_1, a0, b1);
    c1_0 = vmlaq_f64(c1_0, a1, b0);
    c1_1 = vmlaq_f64(c1_1, a1, b1);
    c2_0 = vmlaq_f64(c2_0, a2, b0);
    c2_1 = vmlaq_f64(c2_1, a2, b1);
    c3_0 = vmlaq_f64(c3_0, a3, b0);
    c3_1 = vmlaq_f64(c3_1, a3, b1);
  }

  // 加算結果をCに書き戻す（Cは事前に初期化されている前提）
  float64x2_t c_val;
  // 行0
  c_val = vld1q_f64(C + (i + 0) * N + j);
  c_val = vaddq_f64(c_val, c0_0);
  vst1q_f64(C + (i + 0) * N + j, c_val);
  c_val = vld1q_f64(C + (i + 0) * N + j + 2);
  c_val = vaddq_f64(c_val, c0_1);
  vst1q_f64(C + (i + 0) * N + j + 2, c_val);
  // 行1
  c_val = vld1q_f64(C + (i + 1) * N + j);
  c_val = vaddq_f64(c_val, c1_0);
  vst1q_f64(C + (i + 1) * N + j, c_val);
  c_val = vld1q_f64(C + (i + 1) * N + j + 2);
  c_val = vaddq_f64(c_val, c1_1);
  vst1q_f64(C + (i + 1) * N + j + 2, c_val);
  // 行2
  c_val = vld1q_f64(C + (i + 2) * N + j);
  c_val = vaddq_f64(c_val, c2_0);
  vst1q_f64(C + (i + 2) * N + j, c_val);
  c_val = vld1q_f64(C + (i + 2) * N + j + 2);
  c_val = vaddq_f64(c_val, c2_1);
  vst1q_f64(C + (i + 2) * N + j + 2, c_val);
  // 行3
  c_val = vld1q_f64(C + (i + 3) * N + j);
  c_val = vaddq_f64(c_val, c3_0);
  vst1q_f64(C + (i + 3) * N + j, c_val);
  c_val = vld1q_f64(C + (i + 3) * N + j + 2);
  c_val = vaddq_f64(c_val, c3_1);
  vst1q_f64(C + (i + 3) * N + j + 2, c_val);
}

// 外側のタイル化はそのまま維持し、内側の計算は4x4マイクロカーネルで処理する
void gemm_multi_tiled_simd_microkernel(double *restrict A, double *restrict B,
                                       double *restrict C) {
  int i, j;
  int i0, j0, k0;
  int i1, j1, k1;

  // 外側のブロックループ
  for (i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
    for (j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
      for (k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
        int i0_max = (i0 + BLOCK_SIZE < N) ? (i0 + BLOCK_SIZE) : N;
        int j0_max = (j0 + BLOCK_SIZE < N) ? (j0 + BLOCK_SIZE) : N;
        int k0_max = (k0 + BLOCK_SIZE < N) ? (k0 + BLOCK_SIZE) : N;

        // 内側のタイルループ
        for (i1 = i0; i1 < i0_max; i1 += TILE_SIZE) {
          for (j1 = j0; j1 < j0_max; j1 += TILE_SIZE) {
            for (k1 = k0; k1 < k0_max; k1 += TILE_SIZE) {
              int i1_max =
                  (i1 + TILE_SIZE < i0_max) ? (i1 + TILE_SIZE) : i0_max;
              int j1_max =
                  (j1 + TILE_SIZE < j0_max) ? (j1 + TILE_SIZE) : j0_max;
              int k1_max =
                  (k1 + TILE_SIZE < k0_max) ? (k1 + TILE_SIZE) : k0_max;

              // 内部計算：マイクロカーネルで4x4ブロックごとに処理
              for (i = i1; i < i1_max; i += 4) {
                for (j = j1; j < j1_max; j += 4) {
                  // ここではプリフェッチも必要に応じて実施できますが、
                  // マイクロカーネル内部はすでに高速化されているため、
                  // 外側のループでのプリフェッチ（例：C行列）で十分とする。
                  __builtin_prefetch(C + i * N + j + 16, 1, 1);

                  // k軸方向はタイル単位でループして部分積を加算
                  gemm_microkernel_4x4(A, B, C, i, j, k1, k1_max);
                }
              }
            }
          }
        }
      }
    }
  }
}

// 外側のタイル化はそのまま維持し、内側の計算は4x4マイクロカーネルで処理する
void gemm_multi_tiled_simd_microkernel_multi_thread(double *restrict A,
                                                    double *restrict B,
                                                    double *restrict C) {
  int i, j;
  int i0, j0, k0;
  int i1, j1, k1;

#pragma omp parallel for schedule(dynamic, 1) collapse(2) private(i0,j0,k0,i1,j1,k1,i,j)
  for (i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
    for (j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
      for (k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
        int i0_max = (i0 + BLOCK_SIZE < N) ? (i0 + BLOCK_SIZE) : N;
        int j0_max = (j0 + BLOCK_SIZE < N) ? (j0 + BLOCK_SIZE) : N;
        int k0_max = (k0 + BLOCK_SIZE < N) ? (k0 + BLOCK_SIZE) : N;

        // 内側のタイルループ
        for (i1 = i0; i1 < i0_max; i1 += TILE_SIZE) {
          for (j1 = j0; j1 < j0_max; j1 += TILE_SIZE) {
            for (k1 = k0; k1 < k0_max; k1 += TILE_SIZE) {
              int i1_max =
                  (i1 + TILE_SIZE < i0_max) ? (i1 + TILE_SIZE) : i0_max;
              int j1_max =
                  (j1 + TILE_SIZE < j0_max) ? (j1 + TILE_SIZE) : j0_max;
              int k1_max =
                  (k1 + TILE_SIZE < k0_max) ? (k1 + TILE_SIZE) : k0_max;

              // 内部計算：マイクロカーネルで4x4ブロックごとに処理
              for (i = i1; i < i1_max; i += 4) {
                for (j = j1; j < j1_max; j += 4) {
                  // C行列の先読み（書き込み先）
                  __builtin_prefetch(C + i * N + j + 16, 1, 1);
                  // k軸方向はタイル単位でループして部分積を加算
                  gemm_microkernel_4x4(A, B, C, i, j, k1, k1_max);
                }
              }
            }
          }
        }
      }
    }
  }
}

// OpenBLAS GEMM実装のラッパー: C = A * B
void gemm_openblas(double *A, double *B, double *C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);
}

// 行列メモリを確保する関数
double *allocate_matrix(void) {
  double *M = (double *)malloc(N * N * sizeof(double));
  if (M == NULL) {
    printf("メモリ確保に失敗しました\n");
    exit(1);
  }
  return M;
}

// 行列をランダム値で初期化する関数
void initialize_random_matrix(double *M) {
  for (int i = 0; i < N * N; i++) {
    M[i] = (double)rand() / RAND_MAX;
  }
}

// 行列をゼロクリアする補助関数
void zero_matrix(double *M) {
  for (int i = 0; i < N * N; i++) {
    M[i] = 0.0;
  }
}

// 行列間の最大誤差を計算する関数
double calculate_max_error(double *A, double *B) {
  double max_error = 0.0;
  for (int i = 0; i < N * N; i++) {
    double diff = fabs(A[i] - B[i]);
    if (diff > max_error) {
      max_error = diff;
    }
  }
  return max_error;
}

// 精度検証を行う関数
void verify_accuracy(double *C_naive, double *C_tiled, double *C_tiled_simd,
                     double *C_multi_tiled, double *C_multi_tiled_unrolled,
                     double *C_multi_tiled_unrolled_prefetch,
                     double *C_multi_tiled_microkernel) {
  double max_error_tiled = calculate_max_error(C_naive, C_tiled);
  printf("最大絶対誤差 (gemm vs gemm_tiled): %e\n", max_error_tiled);

  double max_error_simd = calculate_max_error(C_naive, C_tiled_simd);
  printf("最大絶対誤差 (gemm vs gemm_tiled_simd): %e\n", max_error_simd);

  double max_error_multi = calculate_max_error(C_naive, C_multi_tiled);
  printf("最大絶対誤差 (gemm vs gemm_multi_tiled_simd): %e\n", max_error_multi);

  double max_error_unrolled =
      calculate_max_error(C_naive, C_multi_tiled_unrolled);
  printf("最大絶対誤差 (gemm vs gemm_multi_tiled_simd_unrolled): %e\n",
         max_error_unrolled);

  double max_error_prefetch =
      calculate_max_error(C_naive, C_multi_tiled_unrolled_prefetch);
  printf("最大絶対誤差 (gemm vs gemm_multi_tiled_simd_unrolled_prefetch): %e\n",
         max_error_prefetch);

  double max_error_microkernel =
      calculate_max_error(C_naive, C_multi_tiled_microkernel);
  printf("最大絶対誤差 (gemm vs gemm_multi_tiled_simd_microkernel): %e\n",
         max_error_microkernel);
}

// 指定の関数でGEMMを実行し、平均実行時間を計測する関数
double measure_performance(void (*gemm_func)(double *, double *, double *),
                           double *A, double *B, double *C, int iterations) {
  double total_time = 0.0;
  clock_t start, end;

  for (int iter = 0; iter < iterations; iter++) {
    zero_matrix(C);
    start = clock();
    gemm_func(A, B, C);
    end = clock();
    total_time += (double)(end - start) / CLOCKS_PER_SEC;
  }

  return total_time / iterations;
}

// 高速化率を表示する関数
void print_speedup(const char *label, double baseline_time,
                   double optimized_time) {
  printf("高速化率 (%s): %.2f倍\n", label, baseline_time / optimized_time);
}

int main(void) {
  // 行列の確保
  double *A = allocate_matrix();
  double *B = allocate_matrix();
  double *C_naive = allocate_matrix();
  double *C_tiled = allocate_matrix();
  double *C_tiled_simd = allocate_matrix();
  double *C_multi_tiled = allocate_matrix();
  double *C_multi_tiled_unrolled = allocate_matrix();
  double *C_multi_tiled_unrolled_prefetch = allocate_matrix();
  double *C_multi_tiled_microkernel = allocate_matrix(); // マイクロカーネル用
  double *C_multi_tiled_microkernel_multi_thread =
      allocate_matrix(); // マルチスレッド用
  double *C_openblas = allocate_matrix(); // OpenBLAS用

  // 乱数初期化
  srand((unsigned int)time(NULL));
  initialize_random_matrix(A);
  initialize_random_matrix(B);

  // 精度比較のため、各手法を1回ずつ実行して結果を比較
  zero_matrix(C_naive);
  gemm(A, B, C_naive);

  zero_matrix(C_tiled);
  gemm_tiled(A, B, C_tiled);

  zero_matrix(C_tiled_simd);
  gemm_tiled_simd(A, B, C_tiled_simd);

  zero_matrix(C_multi_tiled);
  gemm_multi_tiled_simd(A, B, C_multi_tiled);

  zero_matrix(C_multi_tiled_unrolled);
  gemm_multi_tiled_simd_unrolled(A, B, C_multi_tiled_unrolled);

  zero_matrix(C_multi_tiled_unrolled_prefetch);
  gemm_multi_tiled_simd_unrolled_prefetch(A, B,
                                          C_multi_tiled_unrolled_prefetch);

  // マイクロカーネル実装を実行
  zero_matrix(C_multi_tiled_microkernel);
  gemm_multi_tiled_simd_microkernel(A, B, C_multi_tiled_microkernel);

  // マルチスレッドマイクロカーネル実装を実行
  zero_matrix(C_multi_tiled_microkernel_multi_thread);
  gemm_multi_tiled_simd_microkernel_multi_thread(
      A, B, C_multi_tiled_microkernel_multi_thread);

  // OpenBLAS実装を実行
  zero_matrix(C_openblas);
  gemm_openblas(A, B, C_openblas);

  // 精度の比較
  verify_accuracy(C_naive, C_tiled, C_tiled_simd, C_multi_tiled,
                  C_multi_tiled_unrolled, C_multi_tiled_unrolled_prefetch,
                  C_multi_tiled_microkernel);

  // マルチスレッドマイクロカーネル実装の精度検証
  double max_error_microkernel_multi_thread =
      calculate_max_error(C_naive, C_multi_tiled_microkernel_multi_thread);
  printf("最大絶対誤差 (gemm vs "
         "gemm_multi_tiled_simd_microkernel_multi_thread): %e\n",
         max_error_microkernel_multi_thread);

  // OpenBLASの精度検証
  double max_error_openblas = calculate_max_error(C_naive, C_openblas);
  printf("最大絶対誤差 (gemm vs gemm_openblas): %e\n", max_error_openblas);

  // パフォーマンス計測
  int iterations = 30;
  double avg_time_naive = measure_performance(gemm, A, B, C_naive, iterations);
  double avg_time_tiled =
      measure_performance(gemm_tiled, A, B, C_tiled, iterations);
  double avg_time_tiled_simd =
      measure_performance(gemm_tiled_simd, A, B, C_tiled_simd, iterations);
  double avg_time_multi_tiled = measure_performance(gemm_multi_tiled_simd, A, B,
                                                    C_multi_tiled, iterations);
  double avg_time_multi_tiled_unrolled = measure_performance(
      gemm_multi_tiled_simd_unrolled, A, B, C_multi_tiled_unrolled, iterations);
  double avg_time_multi_tiled_unrolled_prefetch =
      measure_performance(gemm_multi_tiled_simd_unrolled_prefetch, A, B,
                          C_multi_tiled_unrolled_prefetch, iterations);
  double avg_time_multi_tiled_microkernel =
      measure_performance(gemm_multi_tiled_simd_microkernel, A, B,
                          C_multi_tiled_microkernel, iterations);
  double avg_time_multi_tiled_microkernel_multi_thread =
      measure_performance(gemm_multi_tiled_simd_microkernel_multi_thread, A, B,
                          C_multi_tiled_microkernel_multi_thread, iterations);
  double iterations_openblas = 10; // OpenBLASは高速なので少ない反復回数で十分
  double avg_time_openblas = measure_performance(gemm_openblas, A, B, C_openblas, iterations_openblas);

  printf("\n--- 実行時間の比較 ---\n");
  printf("平均実行時間 (gemm): %f 秒\n", avg_time_naive);
  printf("平均実行時間 (gemm_tiled): %f 秒\n", avg_time_tiled);
  printf("平均実行時間 (gemm_tiled_simd): %f 秒\n", avg_time_tiled_simd);
  printf("平均実行時間 (gemm_multi_tiled_simd): %f 秒\n", avg_time_multi_tiled);
  printf("平均実行時間 (gemm_multi_tiled_simd_unrolled): %f 秒\n",
         avg_time_multi_tiled_unrolled);
  printf("平均実行時間 (gemm_multi_tiled_simd_unrolled_prefetch): %f 秒\n",
         avg_time_multi_tiled_unrolled_prefetch);
  printf("平均実行時間 (gemm_multi_tiled_simd_microkernel): %f 秒\n",
         avg_time_multi_tiled_microkernel);
  printf(
      "平均実行時間 (gemm_multi_tiled_simd_microkernel_multi_thread): %f 秒\n",
      avg_time_multi_tiled_microkernel_multi_thread);
  printf("平均実行時間 (gemm_openblas): %f 秒\n", avg_time_openblas);

  printf("\n--- 高速化率の比較 ---\n");
  // 既存の高速化率表示
  print_speedup("naive → tiled", avg_time_naive, avg_time_tiled);
  print_speedup("naive → tiled_simd", avg_time_naive, avg_time_tiled_simd);
  print_speedup("naive → multi_tiled_simd", avg_time_naive,
                avg_time_multi_tiled);
  print_speedup("naive → multi_tiled_simd_unrolled", avg_time_naive,
                avg_time_multi_tiled_unrolled);
  print_speedup("naive → multi_tiled_simd_unrolled_prefetch", avg_time_naive,
                avg_time_multi_tiled_unrolled_prefetch);
  print_speedup("naive → multi_tiled_simd_microkernel", avg_time_naive,
                avg_time_multi_tiled_microkernel);
  print_speedup("naive → multi_tiled_simd_microkernel_multi_thread",
                avg_time_naive, avg_time_multi_tiled_microkernel_multi_thread);
  print_speedup("tiled → tiled_simd", avg_time_tiled, avg_time_tiled_simd);
  print_speedup("tiled → multi_tiled_simd", avg_time_tiled,
                avg_time_multi_tiled);
  print_speedup("tiled → multi_tiled_simd_unrolled", avg_time_tiled,
                avg_time_multi_tiled_unrolled);
  print_speedup("tiled → multi_tiled_simd_unrolled_prefetch", avg_time_tiled,
                avg_time_multi_tiled_unrolled_prefetch);
  print_speedup("tiled_simd → multi_tiled_simd", avg_time_tiled_simd,
                avg_time_multi_tiled);
  print_speedup("tiled_simd → multi_tiled_simd_unrolled", avg_time_tiled_simd,
                avg_time_multi_tiled_unrolled);
  print_speedup("tiled_simd → multi_tiled_simd_unrolled_prefetch",
                avg_time_tiled_simd, avg_time_multi_tiled_unrolled_prefetch);
  print_speedup("multi_tiled_simd → multi_tiled_simd_unrolled",
                avg_time_multi_tiled, avg_time_multi_tiled_unrolled);
  print_speedup("multi_tiled_simd → multi_tiled_simd_unrolled_prefetch",
                avg_time_multi_tiled, avg_time_multi_tiled_unrolled_prefetch);
  print_speedup(
      "multi_tiled_simd_unrolled → multi_tiled_simd_unrolled_prefetch",
      avg_time_multi_tiled_unrolled, avg_time_multi_tiled_unrolled_prefetch);
  print_speedup(
      "multi_tiled_simd_unrolled_prefetch → multi_tiled_simd_microkernel",
      avg_time_multi_tiled_unrolled_prefetch, avg_time_multi_tiled_microkernel);

  // 新たにマルチスレッドマイクロカーネルとの比較を追加
  print_speedup("multi_tiled_simd_microkernel → "
                "multi_tiled_simd_microkernel_multi_thread",
                avg_time_multi_tiled_microkernel,
                avg_time_multi_tiled_microkernel_multi_thread);
  print_speedup("naive → openblas", avg_time_naive, avg_time_openblas);
  print_speedup("multi_tiled_simd_microkernel_multi_thread → openblas", 
                avg_time_multi_tiled_microkernel_multi_thread, avg_time_openblas);

  // メモリ解放
  free(A);
  free(B);
  free(C_naive);
  free(C_tiled);
  free(C_tiled_simd);
  free(C_multi_tiled);
  free(C_multi_tiled_unrolled);
  free(C_multi_tiled_unrolled_prefetch);
  free(C_multi_tiled_microkernel);              // マイクロカーネル用メモリ解放
  free(C_multi_tiled_microkernel_multi_thread); // マルチスレッド用メモリ解放
  free(C_openblas); // OpenBLAS用メモリ解放
  return 0;
}
