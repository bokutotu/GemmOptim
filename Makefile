CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic -O3

build: step1_naive.c
	$(CC) $(CFLAGS) -o step1_naive step1_naive.c

benchmark: step1_naive
	./step1_naive

