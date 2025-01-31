CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic -O3

build: step1_naive.c step2_loop_order.c
	$(CC) $(CFLAGS) -o step1_naive step1_naive.c
	$(CC) $(CFLAGS) -o step2_loop_order step2_loop_order.c

benchmark: step1_naive step2_loop_order
	./step1_naive
	./step2_loop_order

