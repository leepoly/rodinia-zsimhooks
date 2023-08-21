LOCAL_CC = gcc -g -O3 -Wall
CC = gcc
CFLAGS = -lm -fopenmp -Wall
HOOKS_CFLAGS = -I../../common -DENABLE_RODINIA_HOOKS


all : nn

clean :
	rm -rf *.o nn

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) $(HOOKS_CFLAGS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

