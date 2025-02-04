CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic -O3

build: step1_naive.c step2_loop_order.c step3_tiling.c step4_simd_neon.c bench_openblas.c
	$(CC) $(CFLAGS) -o step1_naive step1_naive.c
	$(CC) $(CFLAGS) -o step2_loop_order step2_loop_order.c
	$(CC) $(CFLAGS) -o step3_tiling step3_tiling.c -march=native
	$(CC) $(CFLAGS) -o step4_simd_neon step4_simd_neon.c
	$(CC) $(CFLAGS) -o bench_openblas bench_openblas.c -L/opt/homebrew/Cellar/openblas/0.3.29/lib -lopenblas -I/opt/homebrew/Cellar/openblas/0.3.29/include 

benchmark: step1_naive step2_loop_order step3_tiling
	./step1_naive

	./step2_loop_order

	./step3_tiling

	./step4_simd_neon

	./bench_openblas

clean:
	rm -f step1_naive step2_loop_order step3_tiling step4_simd_neon bench_openblas

commit:
	git add Makefile step1_naive.c step2_loop_order.c step3_tiling.c step4_simd_neon.c bench_openblas.c
	git commit
