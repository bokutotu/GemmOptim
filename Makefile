CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic -O3

build: step1_naive.c step2_loop_order.c step3_tiling.c
	$(CC) $(CFLAGS) -o step1_naive step1_naive.c
	$(CC) $(CFLAGS) -o step2_loop_order step2_loop_order.c
	$(CC) $(CFLAGS) -o step3_tiling step3_tiling.c

benchmark: step1_naive step2_loop_order
	./step1_naive
	./step2_loop_order
	./step3_tiling

clean:
	rm -f step1_naive step2_loop_order step3_tiling

commit:
	git add Makefile step1_naive.c step2_loop_order.c step3_tiling.c
	git commit
